from typing import Any

import os
import statistics

import torch

from navsim.agents.transfuser_mf.transfuser_agent import TransfuserAgent as TF_agent_mf
from navsim.agents.transfuser_mf.transfuser_config import TransfuserConfig as TF_config_mf
from navsim.agents.diffusiondrive_mf.transfuser_agent import DiffusionDrive as DD_agent_mf
from navsim.agents.diffusiondrive_mf.transfuser_config import TransfuserConfig as DD_config_mf
from navsim.agents.diffusiondrive.transfuser_agent import DiffusionDriveAgent as DD_agent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig as DD_config
from navsim.agents.rwkv7_mf.rwkv_agent import RWKVAgent as RWKV_agent_mf
from navsim.agents.rwkv7_mf.rwkv_config import RWKVConfig as RWKV_config_mf
from navsim.agents.MoM_driver.mom_agent import MoMAgent
from navsim.agents.MoM_driver.mom_config import MoMConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = "/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"


def measure_mem_and_time(fn, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    starter.record()
    with torch.no_grad():
        fn(*args, **kwargs)
    ender.record()
    torch.cuda.synchronize(device)
    elapsed = starter.elapsed_time(ender) / 1000.0
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return elapsed, peak_bytes


def compute_statistics(data: list[float]) -> tuple[float, float]:
    mean = statistics.mean(data)
    var = statistics.variance(data) if len(data) > 1 else 0.0
    return mean, var


def print_evaluation_summary(
    tf_time_mem_list,
    dd_mf_time_mem_list,
    dd_time_mem_list,
    rwkv_time_mem_list,
    mom_time_mem_list,
):
    avg_tf_time, var_tf_time = compute_statistics(tf_time_mem_list[0])
    avg_tf_mem, var_tf_mem = compute_statistics(tf_time_mem_list[1])

    avg_dd_mf_time, var_dd_mf_time = compute_statistics(dd_mf_time_mem_list[0])
    avg_dd_mf_mem, var_dd_mf_mem = compute_statistics(dd_mf_time_mem_list[1])

    avg_dd_time, var_dd_time = compute_statistics(dd_time_mem_list[0])
    avg_dd_mem, var_dd_mem = compute_statistics(dd_time_mem_list[1])

    avg_rw_time, var_rw_time = compute_statistics(rwkv_time_mem_list[0])
    avg_rw_mem, var_rw_mem = compute_statistics(rwkv_time_mem_list[1])

    avg_mom_time, var_mom_time = compute_statistics(mom_time_mem_list[0])
    avg_mom_mem, var_mom_mem = compute_statistics(mom_time_mem_list[1])

    print(f"Average TF time: {avg_tf_time:.6f} seconds")
    print(f"Variance TF time: {var_tf_time:.6f} seconds")
    print(f"Average TF memory: {avg_tf_mem:.6f} MB")
    print(f"Variance TF memory: {var_tf_mem:.6f} MB")

    print(f"Average Diffusion MF time: {avg_dd_mf_time:.6f} seconds")
    print(f"Variance Diffusion MF time: {var_dd_mf_time:.6f} seconds")
    print(f"Average Diffusion MF memory: {avg_dd_mf_mem:.6f} MB")
    print(f"Variance Diffusion MF memory: {var_dd_mf_mem:.6f} MB")

    print(f"Average Diffusion time: {avg_dd_time:.6f} seconds")
    print(f"Variance Diffusion time: {var_dd_time:.6f} seconds")
    print(f"Average Diffusion memory: {avg_dd_mem:.6f} MB")
    print(f"Variance Diffusion memory: {var_dd_mem:.6f} MB")

    print(f"Average RWKV time: {avg_rw_time:.6f} seconds")
    print(f"Variance RWKV time: {var_rw_time:.6f} seconds")
    print(f"Average RWKV memory: {avg_rw_mem:.6f} MB")
    print(f"Variance RWKV memory: {var_rw_mem:.6f} MB")

    print(f"Average MoM time: {avg_mom_time:.6f} seconds")
    print(f"Variance MoM time: {var_mom_time:.6f} seconds")
    print(f"Average MoM memory: {avg_mom_mem:.6f} MB")
    print(f"Variance MoM memory: {var_mom_mem:.6f} MB")
    print()


def random_evaluation(seq_len, eva_nums=100):
    tf_time_list = []
    diffusion_mf_time_list = []
    diffusion_time_list = []
    rwkv_time_list = []
    mom_time_list = []

    print(f"Start {seq_len} frame!")
    for i in range(eva_nums):
        if i % 10 == 0:
            print(f"Processing {i} / {eva_nums} ...")

        torch.cuda.synchronize()
        results = evaluate(seq_len)
        tf_time_list.append(results["tf_full"])
        diffusion_mf_time_list.append(results["diff_mf"])
        diffusion_time_list.append(results["diff"])
        rwkv_time_list.append(results["rwkv"])
        mom_time_list.append(results["mom"])

    print_evaluation_summary(
        list(zip(*tf_time_list)),
        list(zip(*diffusion_mf_time_list)),
        list(zip(*diffusion_time_list)),
        list(zip(*rwkv_time_list)),
        list(zip(*mom_time_list)),
    )


def build_mom_features(seq_len: int) -> dict[str, torch.Tensor]:
    mom_cfg = MoMConfig()
    mom_cfg.cam_seq_len = 1
    num_cams = mom_cfg.num_cams
    width, height = mom_cfg.image_size

    image = torch.rand((1, 1, num_cams, 3, height, width), device=device)
    ego_status = torch.rand((1, 1, 11), device=device)
    valid_frame_len = torch.tensor([1], device=device, dtype=torch.int32)

    return {
        "image": image,
        "ego_status": ego_status,
        "valid_frame_len": valid_frame_len,
    }


def evaluate(seq_len: int) -> dict[Any, Any]:
    camera_data = torch.rand((seq_len, 3, 256, 1024), device=device)
    lidar_data = torch.rand((seq_len, 1, 256, 256), device=device)
    status_data = torch.rand((seq_len, 8), device=device)
    valid_frame_data = torch.ones((seq_len, 1), device=device)

    total_features = {
        "camera_feature": camera_data,
        "lidar_feature": lidar_data,
        "status_feature": status_data[-1],
    }
    total_features = {k: v.unsqueeze(0) for k, v in total_features.items()}
    results = {}

    separate_features = []
    for i in range(seq_len):
        diff_feature = {
            "camera_feature": camera_data[i],
            "lidar_feature": lidar_data[i],
            "status_feature": status_data[i],
        }
        diff_feature = {k: v.unsqueeze(0) for k, v in diff_feature.items()}
        separate_features.append(diff_feature)

    rwkv_features = []
    for i in range(seq_len):
        rwkv_feature = {
            "camera_feature": camera_data[i].unsqueeze(0),
            "lidar_feature": lidar_data[i].unsqueeze(0),
            "status_feature": status_data[i],
            "valid_frame_len": valid_frame_data[i],
        }
        rwkv_feature = {k: v.unsqueeze(0) for k, v in rwkv_feature.items()}
        rwkv_features.append(rwkv_feature)

    tf_cfg = TF_config_mf()
    tf_cfg.cam_seq_len = seq_len
    tf_cfg.lidar_seq_len = seq_len
    tf_agent = TF_agent_mf(tf_cfg, lr=1e-4).to(device)
    tf_agent.eval()
    tf_time, tf_mem = measure_mem_and_time(tf_agent.forward, total_features, False)
    results["tf_full"] = (tf_time, tf_mem)
    tf_agent = None

    plan_anchor_path = os.path.join(base_path, "plan_anchors.npy")

    dd_mf_config = DD_config_mf()
    dd_mf_config.cam_seq_len = seq_len
    dd_mf_config.lidar_seq_len = seq_len
    dd_mf_config.plan_anchor_path = plan_anchor_path
    dd_mf_agent = DD_agent_mf(dd_mf_config, lr=1e-4).to(device)
    dd_mf_agent.eval()
    dd_mf_time, dd_mf_mem = measure_mem_and_time(dd_mf_agent.forward, total_features)
    results["diff_mf"] = (dd_mf_time, dd_mf_mem)
    dd_mf_agent = None

    dd_config = DD_config()
    dd_config.plan_anchor_path = plan_anchor_path
    dd_agent = DD_agent(dd_config, lr=1e-4).to(device)
    dd_agent.eval()
    dd_time, dd_mem = measure_mem_and_time(dd_agent.forward, separate_features[-1])
    results["diff"] = (dd_time, dd_mem)
    dd_agent = None

    rwkv_config = RWKV_config_mf()
    rwkv_config.plan_anchor_path = plan_anchor_path
    rwkv_config.cam_seq_len = 1
    rwkv_config.lidar_seq_len = 1
    rwkv_config.model_seq_len = 1
    rwkv_agent = RWKV_agent_mf(rwkv_config, lr=1e-4).to(device)
    rwkv_agent.eval()
    for feat in rwkv_features:
        rwkv_agent.forward(feat)
    rwkv_time, rwkv_mem = measure_mem_and_time(rwkv_agent.forward, rwkv_features[-1])
    results["rwkv"] = (rwkv_time, rwkv_mem)
    rwkv_agent = None

    mom_cfg = MoMConfig()
    mom_cfg.cam_seq_len = 1
    mom_agent = MoMAgent(mom_cfg, lr=1e-4).to(device)
    mom_agent.eval()
    mom_features = build_mom_features(seq_len)
    mom_time, mom_mem = measure_mem_and_time(mom_agent.forward, mom_features)
    results["mom"] = (mom_time, mom_mem)
    mom_agent = None

    for k, (t, m) in results.items():
        results[k] = (t, m / (1024 ** 2))
    return results


if __name__ == "__main__":
    # random_evaluation(seq_len=1, eva_nums=10)
    # random_evaluation(seq_len=4, eva_nums=10)
    # random_evaluation(seq_len=10, eva_nums=10)
    random_evaluation(seq_len=10, eva_nums=5)
    random_evaluation(seq_len=50, eva_nums=5)
