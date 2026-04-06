"""
Memory Slot Routing Visualization for MoM-Driver.

Usage:
    python tools/visualize_memory_slots.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_root /path/to/navsim_data \
        --output_dir ./viz_output \
        --num_samples 8

Produces:
    1. Camera overlay: each scene token colored by its dominant memory slot
    2. Temporal activation: per-slot activation over 10 frames
    3. Camera preference: which cameras each slot prefers
    4. Summary grid: all 4 views × 10 frames in one figure
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

# ─────────────────────────────────────────────────────────────────────────────
# Hook infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class RouterLogitsHook:
    """Forward hook that captures router_logits from every MomBlock."""

    def __init__(self):
        self.logits_per_layer = []   # list[Tensor[B, T, num_memories]]
        self._handles = []

    def register(self, mom_blocks):
        """Register hooks on all MomBlock instances."""
        for block in mom_blocks:
            # MomBlock.forward returns (hidden, attn, cache, router_logits)
            # We intercept via the attn sub-module to get router_logits before
            # it's discarded at the backbone level.
            h = block.register_forward_hook(self._hook_fn)
            self._handles.append(h)

    def _hook_fn(self, module, inputs, outputs):
        # outputs = (hidden_states, attentions, past_key_values, router_logits)
        router_logits = outputs[3]  # [B*T, num_memories]  (view(-1, M))
        if router_logits is not None:
            self.logits_per_layer.append(router_logits.detach().cpu())

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.logits_per_layer.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Routing analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_routing(
    router_logits_flat: torch.Tensor,   # [B*T, M]
    batch_size: int,
    seq_len: int,                        # total tokens T = num_frames * tpf
    num_frames: int,
    tokens_per_frame: int,
    num_cams: int,
    tokens_per_cam: int,                 # = num_scene_tokens = 16
) -> dict:
    """Decode flat router_logits into spatially structured routing weights.

    Returns a dict with keys:
      probs:         [B, T, M]          soft routing probabilities
      dominant:      [B, T]             argmax slot per token
      per_frame_cam: [B, F, C, M]      mean routing prob per frame/camera/slot
      per_frame:     [B, F, M]         mean routing prob per frame/slot
      per_cam:       [B, C, M]         mean routing prob per camera/slot
    """
    B, T, M = batch_size, seq_len, router_logits_flat.shape[-1]

    probs = F.softmax(router_logits_flat.float(), dim=-1)   # [B*T, M]
    probs = probs.reshape(B, T, M)                          # [B, T, M]
    dominant = probs.argmax(dim=-1)                          # [B, T]

    # Reshape to [B, num_frames, tokens_per_frame, M]
    p4d = probs.reshape(B, num_frames, tokens_per_frame, M)
    # Reshape tokens_per_frame → [num_cams, tokens_per_cam]
    p5d = p4d.reshape(B, num_frames, num_cams, tokens_per_cam, M)

    per_frame_cam = p5d.mean(dim=3)         # [B, F, C, M]
    per_frame     = per_frame_cam.mean(dim=2)  # [B, F, M]
    per_cam       = per_frame_cam.mean(dim=1)  # [B, C, M]

    return dict(
        probs=probs,
        dominant=dominant,
        per_frame_cam=per_frame_cam,
        per_frame=per_frame,
        per_cam=per_cam,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualization functions
# ─────────────────────────────────────────────────────────────────────────────

SLOT_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]   # slot 0-3
CAM_NAMES   = ["Front", "Back", "Left", "Right"]
FRAME_LABEL = [f"t-{9-i}" for i in range(10)]                 # t-9 … t-0

def _slot_color_rgba(slot_id, alpha=0.6):
    import matplotlib.colors as mc
    c = mc.to_rgba(SLOT_COLORS[slot_id])
    return (*c[:3], alpha)


def plot_temporal_activation(routing: dict, batch_idx: int, save_path: str):
    """Line plot: per-slot mean routing prob over 10 frames."""
    per_frame = routing["per_frame"][batch_idx].numpy()   # [F, M]
    num_frames, M = per_frame.shape

    fig, ax = plt.subplots(figsize=(8, 3))
    for m in range(M):
        ax.plot(
            range(num_frames), per_frame[:, m],
            color=SLOT_COLORS[m], linewidth=2,
            marker="o", markersize=5,
            label=f"Memory Slot {m}"
        )
    ax.set_xlabel("Frame (t-9 → t-0)")
    ax.set_ylabel("Mean Routing Probability")
    ax.set_title("Memory Slot Temporal Activation")
    ax.set_xticks(range(num_frames))
    ax.set_xticklabels(FRAME_LABEL, rotation=30, ha="right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_camera_preference(routing: dict, batch_idx: int, save_path: str):
    """Bar chart: which cameras each memory slot prefers."""
    per_cam = routing["per_cam"][batch_idx].numpy()   # [C, M]
    num_cams, M = per_cam.shape

    fig, axes = plt.subplots(1, M, figsize=(3 * M, 3), sharey=True)
    if M == 1:
        axes = [axes]
    for m, ax in enumerate(axes):
        values = per_cam[:, m]
        bars = ax.bar(CAM_NAMES[:num_cams], values, color=SLOT_COLORS[m], alpha=0.8)
        ax.set_title(f"Slot {m}", color=SLOT_COLORS[m], fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean Prob" if m == 0 else "")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Camera Preference per Memory Slot", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slot_heatmap_grid(
    routing: dict,
    images: torch.Tensor,     # [B, F, C, 3, H, W]  (uint8 or float)
    batch_idx: int,
    tokens_per_cam: int,      # 16 → 4×4 grid
    save_path: str,
):
    """Camera images (10 frames × 4 cameras) with memory slot color overlay.

    Each scene token (4×4 grid per camera image) is tinted with the color
    of its dominant memory slot.
    """
    B, num_frames, num_cams, _, H, W = images.shape
    grid_size = int(tokens_per_cam ** 0.5)              # 4 for 16 tokens

    # dominant: [B, T] → [B, F, C, tokens_per_cam]
    D = routing["dominant"][batch_idx]                   # [T]
    T = D.shape[0]
    F_  = num_frames
    C_  = num_cams
    tpc = tokens_per_cam
    D = D.reshape(F_, C_, tpc)                          # [F, C, tpc]

    # Build figure: rows=cameras, cols=frames
    fig, axes = plt.subplots(num_cams, num_frames,
                             figsize=(1.8 * num_frames, 2.5 * num_cams))

    for cam_i in range(num_cams):
        for frame_i in range(num_frames):
            ax = axes[cam_i][frame_i]

            # Get the camera image
            img = images[batch_idx, frame_i, cam_i]     # [3, H, W]
            if img.dtype == torch.uint8:
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = (img.permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
            ax.imshow(img_np)

            # Overlay 4×4 grid with slot color
            token_slots = D[frame_i, cam_i].numpy()     # [tpc]
            cell_h = H / grid_size
            cell_w = W / grid_size
            for tok_i, slot_id in enumerate(token_slots):
                row = tok_i // grid_size
                col = tok_i % grid_size
                rect = mpatches.Rectangle(
                    (col * cell_w, row * cell_h), cell_w, cell_h,
                    linewidth=0, facecolor=SLOT_COLORS[int(slot_id)], alpha=0.45
                )
                ax.add_patch(rect)

            ax.set_xticks([])
            ax.set_yticks([])
            if frame_i == 0:
                ax.set_ylabel(CAM_NAMES[cam_i], fontsize=8, rotation=90)
            if cam_i == 0:
                ax.set_title(FRAME_LABEL[frame_i], fontsize=8)

    # Legend
    patches = [mpatches.Patch(color=SLOT_COLORS[m], label=f"Slot {m}",
                              alpha=0.8) for m in range(routing["probs"].shape[-1])]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Memory Slot Assignment per Token (dominant slot)", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slot_probability_grid(
    routing: dict,
    batch_idx: int,
    num_frames: int,
    num_cams: int,
    save_path: str,
):
    """Heatmap: routing probability of each slot across [frames × cameras]."""
    per_frame_cam = routing["per_frame_cam"][batch_idx].numpy()  # [F, C, M]
    M = per_frame_cam.shape[-1]

    fig, axes = plt.subplots(1, M, figsize=(3 * M, 3))
    if M == 1:
        axes = [axes]
    for m, ax in enumerate(axes):
        data = per_frame_cam[:, :, m]   # [F, C]
        im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1, origin="upper")
        ax.set_title(f"Slot {m}", color=SLOT_COLORS[m], fontweight="bold")
        ax.set_yticks(range(num_cams))
        ax.set_yticklabels(CAM_NAMES[:num_cams], fontsize=8)
        ax.set_xticks(range(num_frames))
        ax.set_xticklabels(FRAME_LABEL, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Frame")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Routing Probability: Memory Slot × [Frame, Camera]",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization(model, dataloader, output_dir, num_samples, device):
    """Run the model with hooks to capture routing, then save all plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()

    # Attach hooks to the temporal MoM blocks
    hook = RouterLogitsHook()
    hook.register(model.backbone.mom_blocks)

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            if sample_idx >= num_samples:
                break

            hook.clear()

            # ---- forward pass ----
            image = batch["image"].to(device)           # [B, F, C, 3, H, W]
            valid_frame_len = batch["valid_frame_len"].to(device)
            _ = model.backbone(image, valid_frame_len)

            if not hook.logits_per_layer:
                print("WARNING: no router_logits captured — check hook registration")
                break

            # Use last layer's logits (most semantically rich)
            logits = hook.logits_per_layer[-1]          # [B*T, M]
            B = image.shape[0]
            T = image.shape[1] * model.backbone.tokens_per_frame
            M = logits.shape[-1]
            logits = logits.reshape(B, T, M)

            num_frames    = model.backbone.seq_len
            tokens_per_frame = model.backbone.tokens_per_frame
            num_cams      = model.backbone.num_cams
            tokens_per_cam = model.backbone.num_scene_tokens

            routing = decode_routing(
                logits.reshape(B * T, M),
                batch_size=B,
                seq_len=T,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                num_cams=num_cams,
                tokens_per_cam=tokens_per_cam,
            )

            for b in range(B):
                sid = f"sample_{sample_idx:04d}"
                print(f"  Saving visualizations for {sid} ...")

                # 1. Temporal activation line plot
                plot_temporal_activation(
                    routing, b,
                    save_path=str(output_dir / f"{sid}_temporal.png")
                )

                # 2. Camera preference bar chart
                plot_camera_preference(
                    routing, b,
                    save_path=str(output_dir / f"{sid}_cam_pref.png")
                )

                # 3. Slot probability heatmap [frame × camera]
                plot_slot_probability_grid(
                    routing, b, num_frames, num_cams,
                    save_path=str(output_dir / f"{sid}_slot_heatmap.png")
                )

                # 4. Camera image grid with slot overlay (if images available)
                try:
                    plot_slot_heatmap_grid(
                        routing, image.cpu(), b, tokens_per_cam,
                        save_path=str(output_dir / f"{sid}_overlay.png")
                    )
                except Exception as e:
                    print(f"    Overlay skipped: {e}")

                sample_idx += 1
                if sample_idx >= num_samples:
                    break

    hook.remove()
    print(f"\nDone. Saved {sample_idx} samples to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--data_root",  required=True, help="NavSim data root")
    p.add_argument("--output_dir", default="./viz_memory_slots")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- load model ----
    # Adjust import path to match your project structure
    from navsim.agents.MoM_driver.mom_model import MoMDriverModel
    model = MoMDriverModel.load_from_checkpoint(args.checkpoint)

    # ---- build dataloader ----
    # Replace with your actual NavSim dataloader
    from navsim.planning.training.dataset import CachingAgentDataset
    from torch.utils.data import DataLoader
    dataset = CachingAgentDataset(args.data_root, agent_name="mom_driver_agent")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    run_visualization(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
