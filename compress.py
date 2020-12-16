import torch
from utils import *
from Dolores import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Checkpoint path.')
    args = parser.parse_args()

    ckpt_path = args.input
    ori_name = ckpt_path.split('.pth')[0]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    params_32bit = ckpt['state']
    ckpt32_name = ori_name + '32bit.pth'
    torch.save({'state':params_32bit}, ckpt32_name)
    params_16bit = params_32bit
    for name, value in params_32bit.items():
        params_16bit[name] = value.half()
    print('*'*50)
    for name, value in params_16bit.items():
        print(value.dtype)

    ckpt16_name = ori_name + '16bit.pth'
    torch.save({'state':params_16bit}, ckpt16_name)






