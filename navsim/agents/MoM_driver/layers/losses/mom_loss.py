from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    gt_valid_expanded = gt_valid[:, :, None].detach().float()
    pred_logits_expanded = pred_logits[:, None, :].detach()

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    gt_states_expanded = gt_states[:, :, None, :2].detach()
    pred_states_expanded = pred_states[:, None, :, :2].detach()
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1)
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _agent_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    agent_class_weight: float,
    agent_box_weight: float,
):
    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    batch_dim, _ = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = agent_class_weight * ce_cost + agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for c in cost]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


def _three_to_two_classes(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[x == 0.5] = 0.0
    return x


class MoMDrivoRLoss(torch.nn.Module):
    def __init__(
        self,
        trajectory_weight: float = 1.0,
        inter_weight: float = 1.0,
        sub_score_weight: float = 1.0,
        final_score_weight: float = 1.0,
        pred_ce_weight: float = 1.0,
        pred_l1_weight: float = 1.0,
        pred_area_weight: float = 1.0,
        prev_weight: float = 1.0,
        agent_class_weight: float = 1.0,
        agent_box_weight: float = 1.0,
        bev_semantic_weight: float = 1.0,
    ):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.inter_weight = inter_weight
        self.sub_score_weight = sub_score_weight
        self.final_score_weight = final_score_weight
        self.pred_ce_weight = pred_ce_weight
        self.pred_l1_weight = pred_l1_weight
        self.pred_area_weight = pred_area_weight
        self.prev_weight = prev_weight
        self.agent_class_weight = agent_class_weight
        self.agent_box_weight = agent_box_weight
        self.bev_semantic_weight = bev_semantic_weight

    def score_loss(
        self,
        pred_logit,
        pred_logit2,
        agents_state,
        pred_area_logits,
        target_scores,
        gt_states,
        gt_valid,
        gt_ego_areas,
        l2_distance,
    ):
        pred_ce_loss = 0
        pred_l1_loss = 0
        pred_area_loss = 0

        comfort = pred_logit["comfort"]
        dtype = comfort.dtype

        no_at_fault_collisions = pred_logit["no_at_fault_collisions"]
        drivable_area_compliance = pred_logit["drivable_area_compliance"]
        time_to_collision_within_bound = pred_logit["time_to_collision_within_bound"]
        ego_progress = pred_logit["ego_progress"]
        driving_direction_compliance = pred_logit["driving_direction_compliance"]

        (
            gt_no_at_fault_collisions,
            gt_drivable_area_compliance,
            gt_ego_progress,
            gt_time_to_collision_within_bound,
            gt_comfort,
            gt_driving_direction_compliance,
            _,
        ) = torch.split(target_scores, 1, dim=-1)

        gt_no_at_fault_collisions = gt_no_at_fault_collisions.squeeze(-1)
        gt_drivable_area_compliance = gt_drivable_area_compliance.squeeze(-1)
        gt_ego_progress = gt_ego_progress.squeeze(-1)
        gt_time_to_collision_within_bound = gt_time_to_collision_within_bound.squeeze(-1)
        gt_driving_direction_compliance = gt_driving_direction_compliance.squeeze(-1)
        gt_comfort = gt_comfort.squeeze(-1)

        da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance, gt_drivable_area_compliance.to(dtype))

        mask_valid_ttc = (gt_time_to_collision_within_bound != 2.0).float()
        ttc_loss = (
            F.binary_cross_entropy_with_logits(
                time_to_collision_within_bound,
                gt_time_to_collision_within_bound.to(dtype),
                mask_valid_ttc,
                reduction="sum",
            )
            / mask_valid_ttc.sum().clamp(min=1.0)
        )

        noc_gt = _three_to_two_classes(gt_no_at_fault_collisions.to(dtype))
        noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, noc_gt)
        progress_loss = F.binary_cross_entropy_with_logits(ego_progress, gt_ego_progress.to(dtype))

        ddc_gt = _three_to_two_classes(gt_driving_direction_compliance.to(dtype))
        ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, ddc_gt)

        comfort_loss = F.binary_cross_entropy_with_logits(comfort, gt_comfort.to(dtype))

        sub_score_loss = [da_loss, ttc_loss, noc_loss, progress_loss, ddc_loss, comfort_loss]

        final_score_loss = da_loss + ttc_loss + noc_loss + progress_loss + ddc_loss + comfort_loss

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def diversity_loss(self, proposals: torch.Tensor) -> torch.Tensor:
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)
        dist = dist + (dist == 0)
        inter_loss = -dist.amin(1).amin(1).mean()
        return inter_loss

    def forward(self, targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config, scoring_function):
        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]

        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = scoring_function(
            targets, proposals, test=False
        )

        target_trajectory_long = targets.get("trajectory_long")

        trajectory_loss = 0
        min_loss_list = []
        inter_loss_list = []
        for proposals_i in proposal_list:
            min_loss = torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1).mean(-1).amin(1).mean()

            if target_trajectory_long is not None:
                min_loss = min_loss + torch.linalg.norm(
                    proposals_i - target_trajectory_long[:, None], dim=-1, ord=1
                ).mean(-1).amin(1).mean()

            inter_loss = self.diversity_loss(proposals_i)
            trajectory_loss = self.prev_weight * trajectory_loss + min_loss + inter_loss * self.inter_weight
            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        l2_distance = -((proposals.detach() - target_trajectory[:, None]) ** 2) / 0.5

        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"],
                pred["pred_logit2"],
                pred["pred_agents_states"],
                pred["pred_area_logit"],
                target_scores,
                gt_states,
                gt_valid,
                gt_ego_areas,
                l2_distance.detach(),
            )
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = _agent_loss(
                targets, pred, self.agent_class_weight, self.agent_box_weight
            )
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        loss = (
            self.trajectory_weight * trajectory_loss
            + self.final_score_weight * final_score_loss
            + self.pred_ce_weight * pred_ce_loss
            + self.pred_l1_weight * pred_l1_loss
            + self.pred_area_weight * pred_area_loss
            + self.agent_class_weight * agent_class_loss
            + self.agent_box_weight * agent_box_loss
            + self.bev_semantic_weight * bev_semantic_loss
        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        da_loss, ttc_loss, noc_loss, progress_loss, ddc_loss, comfort_loss = sub_score_loss

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            "da_loss": da_loss,
            "ttc_loss": ttc_loss,
            "noc_loss": noc_loss,
            "progress_loss": progress_loss,
            "ddc_loss": ddc_loss,
            "comfort_loss": comfort_loss,
            "final_score_loss": final_score_loss,
            "pred_ce_loss": pred_ce_loss,
            "pred_l1_loss": pred_l1_loss,
            "pred_area_loss": pred_area_loss,
            "inter_loss0": inter_loss0,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score,
        }

        return loss_dict
