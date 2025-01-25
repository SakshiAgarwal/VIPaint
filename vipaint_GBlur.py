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
from utils.mask_generator import mask_generator, SuperResolutionOperator, get_noise, get_operator, GaussialBlurOperator
from utils.helper import encoder_kl, clean_directory, to_img, encoder_vq, load_file
from ldm.guided_diffusion.h_posterior import HPosterior
from PIL import Image
import numpy as np 
from torch.nn import functional as F
#from guided_diffusion.measurements import get_noise, get_operator

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpaint_config', type=str, default='configs/inpainting/lsun_config_GBlur.yaml') #lsun_config, imagenet_config
    parser.add_argument('--working_directory', type=str, default='results_GBlur_VIP/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--k_steps', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--beta', type=int, default=100)
    #parser.add_argument('--id_start', type=int, default=0)
    #parser.add_argument('--id_end', type=int, default=0)
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
    mask_type = inpaint_config["name_operator"] 

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
    ## Generate super resolution problem, downsample image by some configuration. 
    ## 

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
        measure_config = inpaint_config['measurement']
        operator = GaussialBlurOperator(device=device, **measure_config['operator'])
        #operator = get_operator(device=device, **inpaint_config['operator'])
        #operator = SuperResolutionOperator(device=device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])
        '''
        mask_gen = mask_generator(
            **inpaint_config['mask_opt']
            )
        '''
    logger.info(f"Operation: {mask_type} / Noise: {measure_config['noise']['name']}")

    #mask = torch.tensor(np.load("masks/mask_" + str(args.id) + ".npy")).to(device)
     
    posterior = inpaint_config['posterior']
    if args.k_steps == 1: 
        posterior = "gauss"
        t_steps_hierarchy = [400]
    else : 
        posterior = "hierarchical"
        if args.k_steps == 2: t_steps_hierarchy = [550, 400]
        elif args.k_steps == 4: t_steps_hierarchy = [550, 500, 450, 400] #[550, 450, 350, 250] # 
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
                             z0_size = z0_size,
                             latent_channels = inpaint_config["data"]["latent_channels"],
                             img_size = inpaint_config["data"]["image_size"],
                             first_stage=inpaint_config[posterior]["first_stage"],
                             t_steps_hierarchy=t_steps_hierarchy, #inpaint_config[posterior]['t_steps_hierarchy'],
                             posterior = inpaint_config['posterior']) 
    
    h_inpainter.descretize(inpaint_config[posterior]['rho']) 

    x_size = inpaint_config['mask_opt']['image_size']
    channels = inpaint_config['data']['channels']
    
    ##check higher steps, iterations, without noise
    # Do Inference
    imgs = [0, 19, 15, 13, 11]
    y_gblur_all = np.load("masks/y_gblur.npy")
    for i, random_num in enumerate(seq[0:100]):
        #Prepare working directory
        #0: 50, 3:0, 2: 1, 1: 3 #+ "_450"
        #if i <50: continue
        #if i>=100: break
        img_path = os.path.join(out_path, str(i)) #+ "_no_noise_gloss_400" # + "_" + str(8)  # +str(args.k_steps) + "_h" #"Loss-ablation"
        for img_dir in ['progress', 'progress_inpaintings', 'params', 'samples', 'mus', 
                        'conditional_sampling', 'unconditional_sampling', "samples_K"]:
            sub_dir = os.path.join(img_path, img_dir)
            os.makedirs(sub_dir, exist_ok=True)

        bs = inpaint_config[posterior]["batch_size"]
        logger.info(f"Inference for image {i}")

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
        '''
        if  os.path.exists(inpaint_config['mask_files'][mask_type]):
            #+50 for random_all case
            mask = torch.tensor(mask_gen[(random_num+50)%100].reshape(1,1,256,256)).to(device).float()
        else: 
            mask = torch.tensor(mask_gen(ref_img)).to(device)
        '''

        ref_img = torch.permute(ref_img, (0,3,1,2)) 
        #y_n = operator.forward(ref_img).float().detach()
        y_n = torch.tensor(y_gblur_all[i]).cuda() # operator.forward(ref_img.to(torch.double)).float()
        noiser_all = np.load("masks/noiser_GBlur.npy")
        if args.inpaint_config == 'configs/inpainting/lsun_config_SR.yaml':
            y_n = y_n + torch.tensor(noiser_all[i].transpose(2,0,1)).cuda() 
            y_n = y_n.float()
        #y_n = noiser(y_n).float()
        #y = torch.Tensor.repeat(mask*ref_img, [bs,1,1,1]).float()
        #up_sample = partial(F.interpolate, scale_factor=inpaint_config["measurement"]["operator"]["scale_factor"])
        #y_n_up = up_sample(y_n)

        ##For initialization use y, upsample y by  some function
        if inpaint_config[posterior]["first_stage"] == "kl" : 
            y_encoded = encoder_kl(diff, y_n)[0]
        else: 
            y_encoded = encoder_vq(diff, y_n) 

        plt.imsave(os.path.join(img_path, 'true.png'), to_img(ref_img).astype(np.uint8)[0])
        plt.imsave(os.path.join(img_path, 'observed.png'), to_img(y_n).astype(np.uint8)[0])
        
        lambda_ = h_inpainter.init(y_encoded.detach(), inpaint_config["init"]["var_scale"], 
                                inpaint_config[posterior]["mean_scale"], inpaint_config["init"]["prior_scale"],
                                inpaint_config[posterior]["mean_scale_top"])
        
        # Fit posterior once
        
        h_inpainter.fit(lambda_ = lambda_, cond=c, shape = (bs, *y_encoded.shape[1:]),
                quantize_denoised=False, mask_pixel = None, y = y_n,
                log_every_t=50, iterations = args.iterations, #inpaint_config[posterior]['iterations'],
                unconditional_guidance_scale= inpaint_config[posterior]["unconditional_guidance_scale"] ,
                unconditional_conditioning=uc, kl_weight_1= args.beta, #inpaint_config[posterior]["beta_1"], 
                 kl_weight_2=args.beta, debug=True, wdb = False, #inpaint_config[posterior]["beta_2"],
                dir_name = img_path,
                batch_size = bs, 
                lr_init_gamma = inpaint_config[posterior]["lr_init_gamma"], 
                operator = operator
                )  
        
        # Load parameters and sample
        #for j in [0,50, 100, 150, 200, 249]:
        #for scale in [20, 50, 100]:
        
        params_path = os.path.join(img_path, 'params', f'{args.iterations}.pt') #, j+1
        [mu, logvar, gamma] = torch.load(params_path)
        
        samples = h_inpainter.sample(inpaint_config["sampling"]["scale"], inpaint_config[posterior]["eta"],
                            mu.cuda(), logvar.cuda(), gamma.cuda(), None,  y_n,
                            n_samples=inpaint_config["sampling"]["n_samples"], 
                            batch_size = bs, dir_name= img_path, cond=c, 
                            unconditional_conditioning=uc, 
                            unconditional_guidance_scale=inpaint_config["sampling"]["unconditional_guidance_scale"], 
                            samples_iteration=inpaint_config[posterior]["iterations"], 
                            operator = operator)  
        #x1 = ref_img #.cuda()
        #x2 = samples
        #mse = torch.mean((x1 - x2) ** 2, dim=(1, 2, 3))
        #psnr = 10 * torch.log10(1 / (mse + 1e-10)) 
        #print(psnr)  
        #break       
        #break
        

if __name__ == '__main__':
    main()