import os
import argparse
import torch
import yaml 
from evaluation.psnr import *
from evaluation.kid import *
from utils.helper import load_file
from ldm.util import get_obj_from_str

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpaint_config', type=str, default='configs/inpainting/imagenet_config.yaml') 
    parser.add_argument('--working_directory', type=str, default='results/imagenet/random_all/')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args("")

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  

    # Load configurations
    inpaint_config = load_yaml(args.inpaint_config)
    dir_vipaint = args.working_directory
    files = {"VIPaint" : "10x3x256x256-samples.npz"}

    #Load files 
    mask_type = inpaint_config["mask_opt"]["mask_type"] 
    (start, stop) = inpaint_config['data']['seq'][mask_type]

    ##Get test data 
    if os.path.exists(inpaint_config['data']['file_name']):
        dataset = np.load(inpaint_config['data']['file_name'])["images"] #.reshape(1,channels, x_size, x_size)
        dataset = np.transpose( dataset , [0, 2,3,1])
        dataset = dataset/127.5 - 1
    else: 
        dataset = get_obj_from_str(inpaint_config['data']['name'], reload=False)(0.5)

    data_all = torch.utils.data.DataLoader(dataset= dataset.astype(np.float32), batch_size=1)
    seq = load_file(inpaint_config['data']['file_seq'])[start:stop]

    get_psnr_metrics(seq, data_all, dir_vipaint, "VIPaint", files=files)
    get_kid(seq, data_all, dir_vipaint, algo = "VIPaint",files=files)


if __name__ == '__main__':
    main()