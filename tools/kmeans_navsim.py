import os
import pickle
from typing import Dict, Any

import torch
from tqdm import tqdm
import multiprocess as mp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.rwkv7.rwkv_config import RWKVConfig
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.agents.rwkv7.rwkv_agent import RWKVAgent as RWKV_agent
from navsim.agents.rwkv7.rwkv_config import RWKVConfig as RWKV_config
from collections import defaultdict

fp = '/home/jihao/navsim_workspace/exp/training_cache'


def loadDataset(fp) -> dict[int, [np.ndarray]]:
    """
      Main entrypoint for training an agent.
      :param cfg: omegaconf dictionary
    """
    rwkv_config = RWKVConfig()
    rwkv_config.plan_anchor_path = "/mnt/nas/openscene/openscene/OpenDriveLab___OpenScene/navsim_workspace/navsim/plan_anchors.npy"
    rwkv_agent = RWKV_agent(rwkv_config, lr=1e-4)

    train_data = CacheOnlyDataset(
        cache_path=fp,
        feature_builders=rwkv_agent.get_feature_builders(),
        target_builders=rwkv_agent.get_target_builders(),
    )
    traj = {}  # command: trajectory
    for d in tqdm(train_data):
        driving_command = d[0]['status_feature'][0]
        if driving_command not in traj:
            traj[driving_command] = []
        traj[driving_command].append(d[1]['trajectory'].numpy()) #only keep xy

    # def process_item(i):
    #     d = train_data[i]
    #     driving_command = d[0]['status_feature'][0]
    #     traj_xy = d[1]['trajectory'].numpy()[:, :2]
    #     return driving_command, traj_xy
    #
    # def parallel_process_dataset(train_data, num_workers=mp.cpu_count()):
    #     with mp.Pool(num_workers) as pool:
    #         results = list(tqdm(pool.imap(process_item, range(len(train_data))), total=len(train_data)))
    #
    #     traj = defaultdict(list)
    #     for cmd, xy in results:
    #         traj[cmd].append(xy)
    #     return traj
    #
    # traj = parallel_process_dataset(train_data)

    return traj


def kmeans_trajs(trajs: dict[int,[np.ndarray]], K: int):
    all_trajs = []
    # get values of trajs
    for _,traj in trajs.items():
        all_trajs.extend(traj)
        # print("len ", len(traj))

    # for _,traj in trajs:
    #     all_trajs.extend(traj)

    traj_arr = np.stack(all_trajs)  # shape: (N, T, 3)

    traj_arr = traj_arr[:,:,:2]

    traj_flat = traj_arr.reshape(len(traj_arr), -1)  # shape: (N, T*2)

    kmeans = KMeans(n_clusters=K).fit(traj_flat)
    centers = kmeans.cluster_centers_.reshape(K, -1, 2)  # shape: (k, T, 2)
    for i in range(K):
        center = centers[i]
        plt.plot(center[:, 0], center[:, 1], label=f"Cluster {i}", linewidth=3)
    plt.savefig(f'/mnt/nas/openscene/openscene/OpenDriveLab___OpenScene/navsim_workspace/navsim/vis/kmeans/plan_{K}', bbox_inches='tight')
    plt.close()
    np.save(f'/mnt/nas/openscene/openscene/OpenDriveLab___OpenScene/navsim_workspace/navsim/vis/kmeans/kmeans_nanyi_plan_vocab_{K}_kmeans.npy', centers)


if __name__ == "__main__":
    # traj = loadDataset(fp)
    with open("/mnt/nas/openscene/openscene/OpenDriveLab___OpenScene/navsim_workspace/navsim/vis/traj_dump.pkl", "rb") as f:
    #     pickle.dump(traj, f)
        traj = pickle.load(f)
    for k in tqdm(range(20,2000,100)):
        kmeans_trajs(traj,k)
