import datetime as dt
import os

ip = "192.168.1.134"

# connections with instances of KSP
conns = [
   {'name': "Game ml1", "address": ip, "rpc_port": 50000, "stream_port": 50001},
]

# A3C PARAMETERS
OUTPUT_GRAPH = True
LOG_DIR = './log'
result_file = os.path.join(LOG_DIR, "res"+str(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+".csv").replace(' ', '_'))
fieldnames = ['counter', 'altitude', 'reward']
N_WORKERS = len(conns)
MAX_EP_STEP = 200000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.90
ENTROPY_BETA = 0.01
LR_A = 0.0001
LR_C = 0.001

# environment parameters
turn_start_altitude = 250
turn_end_altitude = 45000
MAX_ALT = 45000
CONTINUOUS = True


