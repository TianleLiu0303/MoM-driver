import torch
import time

from navsim.agents.transfuser.transfuser_agent import TransfuserAgent as TF_agent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig as tf_config
from navsim.agents.transfuser_mf.transfuser_agent import TransfuserAgent as TF_agent_mf
from navsim.agents.transfuser_mf.transfuser_config import TransfuserConfig as tf_config_mf

from navsim.agents.rwkv7.rwkv_agent import RWKVAgent as RWKV_agent
from navsim.agents.rwkv7.rwkv_config import RWKVConfig as rwkv_config
from navsim.agents.rwkv7_mf.rwkv_agent import RWKVAgent as RWKV_agent_mf
from navsim.agents.rwkv7_mf.rwkv_config import RWKVConfig as rwkv_config_mf

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len = 60
camera_data = torch.rand((seq_len, 3, 256, 1024)).to(device)
lidar_data = torch.rand((seq_len, 1, 256, 256)).to(device)
status_data = torch.rand((seq_len, 8)).to(device)

# whole feature
total_features = {
    "camera_feature": camera_data,
    "lidar_feature": lidar_data,
    "status_feature": status_data[-1]
}
total_features = {k: v.unsqueeze(0) for k, v in total_features.items()}

# separate feature
separate_features = []
for i in range(seq_len):
    rwkv_feature = {
        "camera_feature": camera_data[i],
        "lidar_feature": lidar_data[i],
        "status_feature": status_data[i]
    }
    rwkv_feature = {k: v.unsqueeze(0) for k, v in rwkv_feature.items()}
    separate_features.append(rwkv_feature)

# transfuser agent
tf_agent = TF_agent(tf_config(), lr=1e-4)
tf_agent.to(device)
tf_agent.eval()

tf_agent_mf = TF_agent_mf(tf_config_mf(), lr=1e-4)
tf_agent_mf.to(device)
tf_agent_mf.eval()

# rwkv agent
rwkv_agent = RWKV_agent(rwkv_config(), lr=1e-4)
rwkv_agent.to(device)
rwkv_agent.eval()

rwkv_agent_mf = RWKV_agent_mf(rwkv_config_mf(), lr=1e-4)
rwkv_agent_mf.to(device)
rwkv_agent_mf.eval()

with torch.no_grad():
    tf_agent_mf.forward(total_features)

# cmp time
# start_time = time.perf_counter()
# with torch.no_grad():
#     for feature in separate_features:
#         tf_agent.forward(feature)
# end_time = time.perf_counter()
# print(f'Success of transfuser agent, and total time is {end_time - start_time}')

start_time = time.perf_counter()
with torch.no_grad():
    tf_agent_mf.forward(total_features)
end_time = time.perf_counter()
print(f'Success of transfuser agent_mf, and total time is {end_time - start_time}')

time_list = []
with torch.no_grad():
    for rwkv_feature in separate_features:
        start_time = time.perf_counter()
        rwkv_agent.forward(rwkv_feature)
        end_time = time.perf_counter()
        time_list.append(end_time - start_time)
print(f'Success of rwkv agent, and the total time is {time_list[-1]}')

# start_time = time.perf_counter()
# with torch.no_grad():
#     rwkv_agent_mf.forward(total_features)
# end_time = time.perf_counter()
# print(f'Success of rwkv agent_mf, and the total time is {end_time - start_time}')


