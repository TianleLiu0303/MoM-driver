import pickle
import gzip
with gzip.open('/home/gpus-09/nanyi/rwkv/exp/nanyi_trainning_cache_20_padding/2021.06.14.18.13.35_veh-26_00385_00471/90f1f4ebc0765656/transfuser_feature.gz', 'rb') as f:
    data_points = pickle.load(f)

print(data_points)