from functools import partial
import os
import argparse
import yaml
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config, get_obj_from_str
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.logger import get_logger
from utils.mask_generator import mask_generator
from utils.helper import encoder_kl, clean_directory, to_img, encoder_vq, load_file
from ldm.guided_diffusion.h_posterior import HPosterior
from PIL import Image
import numpy as np 

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpaint_config', type=str, default='configs/inpainting/imagenet_config.yaml') #lsun_config, imagenet_config
    parser.add_argument('--working_directory', type=str, default='results/imagenet/box_labels/')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    inpaint_config = load_yaml(args.inpaint_config)
    working_directory = args.working_directory
    mask_type = inpaint_config["mask_opt"]["mask_type"] 

    # Load model
    config = OmegaConf.load(inpaint_config['diffusion'])
    vae_config = OmegaConf.load(inpaint_config['autoencoder'])

    diff = instantiate_from_config(config.model)
    diff.load_state_dict(torch.load(inpaint_config['diffusion_model'],
                                     map_location='cpu')["state_dict"], strict=False)    
    diff = diff.to(device)
    diff.model.eval()
    diff.first_stage_model.eval()
    diff.eval()

    # Load pre-trained autoencoder loss config
    loss_config = vae_config['model']['params']['lossconfig']
    vae_loss = get_obj_from_str(inpaint_config['loss']['name'],
                                 reload=False)(**loss_config.get("params", dict()))
    
    # Load test data
    (start, stop) = inpaint_config['data']['seq'][mask_type]
    if os.path.exists(inpaint_config['data']['file_name']):
        dataset = np.load(inpaint_config['data']['file_name'])
    else: 
        dataset = get_obj_from_str(inpaint_config['data']['name'], reload=False)(0.5)
        seq = load_file(inpaint_config['data']['file_seq'])[start:stop]
    
    loader = torch.utils.data.DataLoader(dataset= dataset, batch_size=1)

    # Working directory
    out_path = working_directory 
    os.makedirs(out_path, exist_ok=True)

    # Generate a mask 
    if os.path.exists(inpaint_config['mask_files'][mask_type]):
        mask_gen = np.load(inpaint_config['mask_files'][mask_type])
    else:
        mask_gen = mask_generator(
            **inpaint_config['mask_opt']
            )
    # For conditional models
    if inpaint_config["conditional_model"] : 
        uc = diff.get_learned_conditioning(
            {diff.cond_stage_key: torch.tensor(inpaint_config["loss"]["batch_size"]*[1000]).cuda()}
            ).detach()
        
    # Prepare VI method
    z0_size = config['model']['params']['image_size']
    h_inpainter = HPosterior(diff, vae_loss, 
                             eta = inpaint_config["loss"]["eta"],
                             z0_size = z0_size,
                             first_stage=inpaint_config["loss"]["first_stage"],
                             t_steps_hierarchy=inpaint_config['loss']['t_steps_hierarchy']) 
    
    h_inpainter.descretize(inpaint_config['loss']['rho']) 

    x_size = inpaint_config['mask_opt']['image_size']
    channels = inpaint_config['data']['channels']
    
    # Do Inference
    for _, random_num in enumerate(seq):
        #Prepare working directory
        img_path = os.path.join(out_path, str(random_num - start)) 
        for img_dir in ['progress', 'progress_inpaintings', 'params', 'samples', 'mus', 
                        'conditional_sampling', 'unconditional_sampling']:
            sub_dir = os.path.join(img_path, img_dir)
            os.makedirs(sub_dir, exist_ok=True)

        bs = inpaint_config["loss"]["batch_size"]
        logger.info(f"Inference for image {random_num - start}")

        #Get Image/Labels
        if len(loader.dataset) ==2: 
            ref_img = loader.dataset["images"][random_num].reshape(1,channels, x_size, x_size)
            ref_img = np.transpose( ref_img , [0, 2,3,1])
            ref_img = ref_img/127.5 - 1
            label = loader.dataset["labels"][random_num]
            #label = class_label
            xc = torch.tensor(bs*[label]).cuda()
            c = diff.get_learned_conditioning({diff.cond_stage_key: xc}).detach()
        else:
            ref_img = loader.dataset[random_num].reshape(1,x_size,x_size,channels)
            c = None
            uc = None

        ref_img = torch.tensor(ref_img).to(device)

        #Get mask
        if  os.path.exists(inpaint_config['mask_files'][mask_type]):
            mask = torch.tensor(mask_gen[random_num%100].reshape(1,1,256,256)).to(device).float()
        else: 
            mask = torch.tensor(mask_gen(ref_img)).to(device)
            
        ref_img = torch.permute(ref_img, (0,3,1,2)) 
        y = torch.Tensor.repeat(mask*ref_img, [bs,1,1,1]).float()
        
        if inpaint_config["loss"]["first_stage"] == "kl" : 
            y_encoded = encoder_kl(diff, y)[0]
        else: 
            y_encoded = encoder_vq(diff, y) 

        plt.imsave(os.path.join(img_path, 'true.png'), to_img(ref_img).astype(np.uint8)[0])
        plt.imsave(os.path.join(img_path, 'observed.png'), to_img(y).astype(np.uint8)[0])

        lambda_ = h_inpainter.init(y_encoded, inpaint_config["init"]["var_scale"], 
                                inpaint_config["init"]["mean_scale"], inpaint_config["init"]["prior_scale"],
                                inpaint_config["init"]["mean_scale_top"])
        
        # Fit posterior once
        h_inpainter.fit(lambda_ = lambda_, cond=c, shape = (bs, *y_encoded.shape[1:]),
                quantize_denoised=False, mask_pixel = mask, y =y,
                log_every_t=20, iterations = inpaint_config['loss']['iterations'],
                unconditional_guidance_scale= inpaint_config["loss"]["unconditional_guidance_scale"] ,
                unconditional_conditioning=uc, kl_weight=inpaint_config["loss"]["beta"], debug=True, wdb = False,
                dir_name = img_path,
                batch_size = bs, 
                lr_init_gamma = inpaint_config["loss"]["lr_init_gamma"]
                )  
        
        # Load parameters and sample
        params_path = os.path.join(img_path, 'params', f'{inpaint_config["loss"]["iterations"]}.pt')
        [mu, logvar, gamma] = torch.load(params_path)
        
        
        h_inpainter.sample(inpaint_config["sampling"]["scale"], inpaint_config["loss"]["eta"],
                            mu.cuda(), logvar.cuda(), gamma.cuda(), mask,  y,
                            n_samples=inpaint_config["sampling"]["n_samples"], 
                            batch_size = bs, dir_name= img_path, cond=c, 
                            unconditional_conditioning=uc, 
                            unconditional_guidance_scale=inpaint_config["sampling"]["unconditional_guidance_scale"])        
            
if __name__ == '__main__':
    main()