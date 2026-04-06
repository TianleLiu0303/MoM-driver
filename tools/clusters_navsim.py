import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count

fp = '/home/jihao/navsim_workspace/exp/training_cache'

def kmeans_trajs(trajs: dict[int,[np.ndarray]], K: int):
    all_trajs = []
    # get values of trajs
    for _,traj in trajs.items():
        all_trajs.extend(traj)

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


def kmeans_trajs(trajs: dict[int,[np.ndarray]], K: int):
    cluster_dir = "/mnt/nas/openscene/openscene/OpenDriveLab___OpenScene/navsim_workspace/navsim/vis/cluster/kmeans"
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    all_trajs = []
    # get values of trajs
    for _,traj in trajs.items():
        all_trajs.extend(traj)

    traj_arr = np.stack(all_trajs)  # shape: (N, T, 3)
    traj_arr = traj_arr[:,:,:2]

    traj_flat = traj_arr.reshape(len(traj_arr), -1)  # shape: (N, T*2)

    kmeans = KMeans(n_clusters=K).fit(traj_flat)
    centers = kmeans.cluster_centers_.reshape(K, -1, 2)  # shape: (k, T, 2)
    for i in range(K):
        center = centers[i]
        plt.plot(center[:, 0], center[:, 1], label=f"Cluster {i}", linewidth=3)
    fig_path = os.path.join(cluster_dir, f'kmeans_nanyi_plan_vocab_{K}_kmeans.png')
    npy_path = os.path.join(cluster_dir, f'kmeans_nanyi_plan_vocab_{K}_kmeans.npy')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    np.save(npy_path, centers)

def run_kmeans(k):
    return kmeans_trajs(traj, k)

if __name__ == "__main__":
    # traj = loadDataset(fp)
    with open("/mnt/nas/openscene/openscene/OpenDriveLab___OpenScene/navsim_workspace/navsim/vis/traj_dump.pkl", "rb") as f:
        traj = pickle.load(f)
    # for k in tqdm(range(20,2000,20)):
    #     kmeans_trajs(traj,k)
    ks = list(range(20, 2000, 20))
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap_unordered(run_kmeans, ks), total=len(ks)))
