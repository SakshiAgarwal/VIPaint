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
from torchvision.transforms.functional import pil_to_tensor

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpaint_config', type=str, default='configs/inpainting/imagenet_config.yaml') #lsun_config, imagenet_config
    parser.add_argument('--working_directory', type=str, default='results/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--k_steps', type=int, default=2)
    parser.add_argument('--case', type=str, default="random_all")
    args = parser.parse_args("")
   
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
    vae_loss = get_obj_from_str(inpaint_config['name'],
                                 reload=False)(**loss_config.get("params", dict()))
    
    # Load test data
    (start, stop) = inpaint_config['data']['seq'][mask_type]
    if os.path.exists(inpaint_config['data']['file_name']):
        dataset = np.load(inpaint_config['data']['file_name'])
        seq = np.arange(start,stop)
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
        
    #mask = torch.tensor(np.load("masks/mask_" + str(args.id) + ".npy")).to(device)
    ##Define variational posterior on latent time steps
    posterior = inpaint_config['posterior']
    if args.k_steps == 1: 
        posterior = "gauss"
        t_steps_hierarchy = [400]
    else : 
        posterior = "hierarchical"
        if args.k_steps == 2: t_steps_hierarchy = [inpaint_config[posterior]['t_steps_hierarchy'][0], 
                                                   inpaint_config[posterior]['t_steps_hierarchy'][-1]]
        elif args.k_steps == 4: t_steps_hierarchy = inpaint_config[posterior]['t_steps_hierarchy'] # [550, 500, 450, 400]
        elif args.k_steps == 6: t_steps_hierarchy = [650, 600, 550, 500, 450, 400]

    # For conditional models
    if inpaint_config["conditional_model"] : 
        uc = diff.get_learned_conditioning(
            {diff.cond_stage_key: torch.tensor(inpaint_config[posterior]["batch_size"]*[1000]).cuda()}
            ).detach()
        
    # Prepare VI method
    z0_size = config['model']['params']['image_size']
    
    h_inpainter = HPosterior(diff, vae_loss, 
                             eta = inpaint_config[posterior]["eta"],
                             z0_size = inpaint_config["data"]["latent_size"],
                             img_size = inpaint_config["data"]["image_size"],
                             latent_channels = inpaint_config["data"]["latent_channels"],
                             first_stage=inpaint_config[posterior]["first_stage"],
                             t_steps_hierarchy=t_steps_hierarchy, #inpaint_config[posterior]['t_steps_hierarchy'],
                             posterior = inpaint_config['posterior']) 
    
    h_inpainter.descretize(inpaint_config[posterior]['rho']) 

    x_size = inpaint_config['mask_opt']['image_size']
    channels = inpaint_config['data']['channels']
    
    # Do Inference
    imgs = [0, 19, 15, 13, 11]
    for i, random_num in enumerate(seq):
        #Prepare working directory
        if i<0: continue
        if i>=1: break
        img_path = os.path.join(out_path, str(i) )  # +str(args.k_steps) + "_h" #"Loss-ablation"
        for img_dir in ['progress', 'progress_inpaintings', 'params', 'samples', 'mus', 
                        'conditional_sampling', 'unconditional_sampling', "samples_K"]:
            sub_dir = os.path.join(img_path, img_dir)
            os.makedirs(sub_dir, exist_ok=True)

        bs = inpaint_config[posterior]["batch_size"]
        logger.info(f"Inference for image {random_num - start}")

        #Get Image & Labels
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
            #+50 for random_all case, imagenet +50
            mask = torch.tensor(mask_gen[(random_num)%100].reshape(1,1,256,256)).to(device).float()
        else: 
            mask = torch.tensor(mask_gen(ref_img)).to(device)
        ref_img = torch.permute(ref_img, (0,3,1,2)) 

        #Get degraded image
        y = torch.Tensor.repeat(mask*ref_img, [bs,1,1,1]).float()
        
        #Pass y through the encoder  
        if inpaint_config[posterior]["first_stage"] == "kl" : 
            y_encoded = encoder_kl(diff, y)[0]
        else: 
            y_encoded = encoder_vq(diff, y) 

        plt.imsave(os.path.join(img_path, 'true.png'), to_img(ref_img).astype(np.uint8)[0])
        plt.imsave(os.path.join(img_path, 'observed.png'), to_img(y).astype(np.uint8)[0])
        
        #Use encoded y for initiliazing variational parameters
        lambda_ = h_inpainter.init(y_encoded, inpaint_config["init"]["var_scale"], 
                                inpaint_config[posterior]["mean_scale"], inpaint_config["init"]["prior_scale"],
                                inpaint_config[posterior]["mean_scale_top"])
        # Fit posterior once
        h_inpainter.fit(lambda_ = lambda_, cond=c, shape = (bs, *y_encoded.shape[1:]),
                quantize_denoised=False, mask_pixel = mask, y =y,
                log_every_t=25, iterations = inpaint_config[posterior]['iterations'],
                unconditional_guidance_scale= inpaint_config[posterior]["unconditional_guidance_scale"] ,
                unconditional_conditioning=uc, kl_weight_1=inpaint_config[posterior]["beta_1"],
                 kl_weight_2 = inpaint_config[posterior]["beta_2"],
                debug=True, wdb = False,
                dir_name = img_path,
                batch_size = bs, 
                lr_init_gamma = inpaint_config[posterior]["lr_init_gamma"]
                )  
        
        # Load parameters and sample
        params_path = os.path.join(img_path, 'params', f'{inpaint_config[posterior]["iterations"]}.pt') #, j+1
        [mu, logvar, gamma] = torch.load(params_path)
        
        h_inpainter.sample(inpaint_config["sampling"]["scale"], inpaint_config[posterior]["eta"],
                            mu.cuda(), logvar.cuda(), gamma.cuda(), mask,  y,
                            n_samples=inpaint_config["sampling"]["n_samples"], 
                            batch_size = bs, dir_name= img_path, cond=c, 
                            unconditional_conditioning=uc, 
                            unconditional_guidance_scale=inpaint_config["sampling"]["unconditional_guidance_scale"], 
                            samples_iteration=inpaint_config[posterior]["iterations"])        
        
        #break
        

if __name__ == '__main__':
    main()