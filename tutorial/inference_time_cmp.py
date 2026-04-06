from typing import Any

import torch
import time
import statistics

from navsim.agents.transfuser_mf.transfuser_agent import TransfuserAgent as TF_agent_mf
from navsim.agents.transfuser_mf.transfuser_config import TransfuserConfig as tf_config_mf
from navsim.agents.rwkv7_mf.rwkv_agent import RWKVAgent as RWKV_agent
from navsim.agents.rwkv7_mf.rwkv_config import RWKVConfig as rwkv_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_kv_cache = False


def measure_mem_and_time(fn, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    starter.record()
    with torch.no_grad():
        out = fn(*args, **kwargs)
    ender.record()
    torch.cuda.synchronize(device)
    elapsed = starter.elapsed_time(ender) / 1000.0  # 秒
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return elapsed, peak_bytes


def compute_statistics(data: list[float]) -> tuple[float, float]:
    mean = statistics.mean(data)
    var = statistics.variance(data)
    return mean, var


def print_evaluation_summary(
        tf_time_mem_list,
        tf_kvcache_time_mem_list,
        rwkv_time_list):
    # 计算统计量
    avg_tf_time, var_tf_time = compute_statistics(tf_time_mem_list[0])
    avg_tf_mem, var_tf_mem = compute_statistics(tf_time_mem_list[1])

    if test_kv_cache:
        avg_kv_time, var_kv_time = compute_statistics(tf_kvcache_time_mem_list[0])
        avg_kv_mem, var_kv_mem = compute_statistics(tf_kvcache_time_mem_list[1])

    avg_rw_time, var_rw_time = compute_statistics(rwkv_time_list[0])
    avg_rw_mem, var_rw_mem = compute_statistics(rwkv_time_list[1])

    # 打印结果
    print(f"Average TF time: {avg_tf_time:.6f} seconds")
    print(f"Variance TF time: {var_tf_time:.6f} seconds")
    print(f"Average TF memory: {avg_tf_mem :.6f} MB")
    print(f"Variance TF memory: {var_tf_mem :.6f} MB")
    if test_kv_cache:
        print(f"Average TF with KVCache time: {avg_kv_time:.6f} seconds")
        print(f"Variance TF with KVCache time: {var_kv_time:.6f} seconds")
        print(f"Average TF with KVCache memory: {avg_kv_mem :.6f} MB")
        print(f"Variance TF with KVCache memory: {var_kv_mem :.6f} MB")
    print(f"Average RWKV time: {avg_rw_time:.6f} seconds")
    print(f"Variance RWKV time: {var_rw_time:.6f} seconds")
    print(f"Average RWKV memory: {avg_rw_mem :.6f} MB")
    print(f"Variance RWKV memory: {var_rw_mem :.6f} MB")
    print()  # 空行分隔不同 seq_len


def random_evaluation(seq_len, eva_nums=100):
    tf_time_list = []
    tf_kvcache_time_list = []
    rwkv_time_list = []

    print(f'Start {seq_len} frame!')
    for i in range(eva_nums):
        if i % 10 == 0:
            print(f'Processing {i} / {eva_nums} ...')

        torch.cuda.synchronize()
        results = evaluate(seq_len)
        tf_time_list.append(results['tf_full'])
        if test_kv_cache:
            tf_kvcache_time_list.append(results['tf_kv'])
        else:
            tf_kvcache_time_list.append(None)
        rwkv_time_list.append(results['rwkv'])

    print_evaluation_summary(
        list(zip(*tf_time_list)),
        list(zip(*tf_kvcache_time_list)) if test_kv_cache else None,
        list(zip(*rwkv_time_list)),
    )


def evaluate(seq_len: int) -> dict[
    Any, Any]:
    camera_data = torch.rand((seq_len, 3, 256, 1024)).to(device)
    lidar_data = torch.rand((seq_len, 1, 256, 256)).to(device)
    status_data = torch.rand((seq_len, 8)).to(device)
    valid_frame_data = torch.ones((seq_len, 1)).to(device)

    total_features = {
        "camera_feature": camera_data,
        "lidar_feature": lidar_data,
        "status_feature": status_data[-1]
    }
    total_features = {k: v.unsqueeze(0) for k, v in total_features.items()}
    results = {}

    separate_features = []
    for i in range(seq_len):
        rwkv_feature = {
            "camera_feature": camera_data[i].unsqueeze(0),
            "lidar_feature": lidar_data[i].unsqueeze(0),
            "status_feature": status_data[i],
            "valid_frame_len": valid_frame_data[i],
        }
        rwkv_feature = {k: v.unsqueeze(0) for k, v in rwkv_feature.items()}
        separate_features.append(rwkv_feature)

    # 2) 帧级特征，用于 KV-cache
    frame_features = [
        {
            "camera_feature": camera_data[i].unsqueeze(0),  # [B=1, C, H, W]
            "lidar_feature": lidar_data[i].unsqueeze(0),
            "status_feature": status_data[i],
        }
        for i in range(seq_len)
    ]

    # tranfuser

    # define tf agent
    tf_cfg = tf_config_mf()
    tf_cfg.cam_seq_len = seq_len
    tf_cfg.lidar_seq_len = seq_len

    tf_agent = TF_agent_mf(tf_cfg, lr=1e-4)
    tf_agent.to(device)
    tf_agent.eval()

    tf_time, tf_mem = measure_mem_and_time(tf_agent.forward, total_features, False)
    results['tf_full'] = (tf_time, tf_mem)
    tf_agent = None

    if test_kv_cache:
        # define tf cached agent
        tf_cfg_cache = tf_config_mf()
        tf_cfg_cache.cam_seq_len = 1
        tf_cfg_cache.lidar_seq_len = 1

        tf_cached_agent = TF_agent_mf(tf_cfg_cache, lr=1e-4)
        tf_cached_agent.to(device)
        tf_cached_agent.eval()
        # -------- Transfuser（有 KV-cache）：只计最后一帧 --------
        # 先预热
        tf_cached_agent.clear_cache()
        for idx, feat in enumerate(frame_features):
            tf_cached_agent.forward({k: v.unsqueeze(0) for k, v in feat.items()}, use_cache=True)
        # 再测
        last_feat = {k: v.unsqueeze(0) for k, v in frame_features[-1].items()}
        tf_kvc_time, tf_kvc_mem = measure_mem_and_time(tf_cached_agent.forward, last_feat, True)
        results['tf_kv'] = (tf_kvc_time, tf_kvc_mem)
        tf_cached_agent = None

    # define rwkv agent
    rwkv_cfg = rwkv_config()
    rwkv_cfg.cam_seq_len = 1
    rwkv_cfg.lidar_seq_len = 1
    rwkv_cfg.model_seq_len = 1
    rwkv_agent = RWKV_agent(rwkv_cfg, lr=1e-4)
    rwkv_agent.to(device)
    rwkv_agent.eval()

    # rwkv
    for idx, feat in enumerate(separate_features):
        rwkv_agent.forward(feat)
    rwkv_time, rwkv_mem = measure_mem_and_time(rwkv_agent.forward, separate_features[-1])
    results['rwkv'] = (rwkv_time, rwkv_mem)
    rwkv_agent = None

    for k, (t, m) in results.items():
        results[k] = (t, m / (1024 ** 2))
    return results


if __name__ == "__main__":
    # warm up
    random_evaluation(seq_len=1, eva_nums=10)
    # exit()
    random_evaluation(seq_len=1, eva_nums=50)
    # exit()
    random_evaluation(seq_len=4, eva_nums=50)

    random_evaluation(seq_len=10, eva_nums=50)

    random_evaluation(seq_len=15, eva_nums=50)

    random_evaluation(seq_len=20, eva_nums=50)

    random_evaluation(seq_len=30, eva_nums=50)

    random_evaluation(seq_len=40, eva_nums=50)

    random_evaluation(seq_len=50, eva_nums=50)

    random_evaluation(seq_len=60, eva_nums=50)
