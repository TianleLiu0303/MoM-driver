import torch

from fla.layers.gla import Mamba as GatedLinearAttention
from fla.models.utils import Cache


def run_chunked(layer, x, chunk_size, device):
    batch_size, seq_len, _ = x.shape
    cache = Cache()
    outputs = []

    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            x_chunk = x[:, start:end, :]
            attention_mask = torch.ones(batch_size, end - start, device=device, dtype=torch.long)
            y_chunk, _, cache = layer(
                hidden_states=x_chunk,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            outputs.append(y_chunk)

    return torch.cat(outputs, dim=1)


def main():
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_heads = 4
    chunk_size = 8

    device = "cuda:0"
    dtype = torch.float32

    torch.manual_seed(0)

    layer = GatedLinearAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        layer_idx=0,
        use_short_conv=False,
    ).to(device=device, dtype=dtype).eval()

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    y_single = run_chunked(layer, x, chunk_size=1, device=device)
    y_multi = run_chunked(layer, x, chunk_size=chunk_size, device=device)

    y_single_last = y_single[:, -1, :]
    y_multi_last = y_multi[:, -1, :]
    diff = (y_single_last - y_multi_last).abs()

    print("input shape              :", x.shape)
    print("single output shape      :", y_single.shape)
    print(f"chunk={chunk_size} output shape :", y_multi.shape)
    print("last token shape         :", y_single_last.shape)
    print("last token max abs diff  :", diff.max().item())
    print("last token mean abs diff :", diff.mean().item())


if __name__ == "__main__":
    main()
