import numpy as np
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)

args = parser.parse_args()

apls = []
name_list = os.listdir(f'../{args.dir}/results/apls')
name_list.sort()
for file_name in name_list :
    try:
        with open(f'../{args.dir}/results/apls/{file_name}') as f:
            lines = f.readlines()
        print(file_name,lines[0].split(' ')[-1][:-2])
        apls.append(float(lines[0].split(' ')[-1][:-2]))
    except:
        break
print('APLS',np.mean(apls))
with open(f'../{args.dir}/score/apls.json','w') as jf:
    json.dump({'apls':apls,'final_APLS':np.mean(apls)},jf)