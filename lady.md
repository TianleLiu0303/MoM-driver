# 并行数据缓存

1. ray start --head --port=6379 # head 节点
2. ray start --address='<head_ip>:6379' # worker 节点
3. 然后分别在 head 节点和 worker 节点上执行以下命令

 ```
  ./scripts/training/run_load_cache_rwkv.sh
 ```

4. 由于以来的nuplan不支持传递environment,请使用文件的方式传递参数，比如:

   ```
   # ray_env_vars.env
   HF_ENDPOINT=https://hf-mirror.com
   NUPLAN_MAPS_ROOT=/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps
   NAVSIM_EXP_ROOT=/mnt/workspace/nanyi/navsim_workspace/exp
   NUPLAN_MAP_VERSION=nuplan-maps-v1.0
   OPENSCENE_DATA_ROOT=/mnt/pai-pdc-nas/nanyi/openscene-v1.1
   NAVSIM_DEVKIT_ROOT=/mnt/pdc-workspace/nanyi/RWKV_dataloader_cache/data_cache/RWKV-navsim
   ```

   在[run_dataset_caching.py](navsim/planning/script/run_dataset_caching.py)中
   cache_features函数，增加类似下面的内容，也就是每个ray的worker在运行的时候都加载这个文件到os.environ中，这样就可以在ray的worker中使用这些环境变量了。

```python
    env_path = "/mnt/workspace/nanyi/RWKV_dataloader_cache/data_cache/RWKV-navsim/scripts/training/ray_environ.env"
    load_env_file_to_os(env_path)
    def load_env_file_to_os(env_file: str):
    """
    Read key=value pairs from .env file and insert into os.environ.
    """
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file {env_file} not found.")

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()
    ```
    
