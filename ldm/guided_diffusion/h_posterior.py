"""INFERENCE TIME OPTIMIZATION"""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.distributions as td
import gc
import wandb
import matplotlib.pyplot as plt
from utils.helper import params_train, get_optimizers,clean_directory, time_descretization, to_img, custom_to_np, save_params, save_samples, save_inpaintings, save_plot
import os
import PIL
import glob
from tqdm import trange
import time
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, extract_into_tensor, noise_like
import wandb

class HPosterior(object):
    def __init__(self, model, vae_loss, t_steps_hierarchy, eta=0.4, z0_size=32, num_hierarchy_steps=5, schedule="linear", first_stage = "kl", **kwargs):
        super().__init__()
        self.model = model                  #prior noise prediction model
        self.schedule = schedule            #noise schedule the prior was trained on
        self.vae_loss = vae_loss            #vae loss followed during training
        self.eta = eta                      #eta used to produce faster, clean samples 
        self.first_stage= first_stage       #first stage training procedure: kl or vq loss
        self.t_steps_hierarchy = np.array(t_steps_hierarchy)   #time steps for hierachical posterior
        self.z0size = z0_size               #dimension of latent space variables z

    def q_given_te(self, t, s, shape, zeta_t_star=None):
        if zeta_t_star is not None:  
            alpha_s = torch.sqrt(1 - zeta_t_star**2)
            var_s = zeta_t_star**2
        else: 
            if len(s.shape) == 0 :m = 1
            else: m = s.shape[0]
            var_s = (self.model.sqrt_one_minus_alphas_cumprod[s].reshape(m, 1 ,1 ,1))**2 
            alpha_s = torch.sqrt(1 - var_s)
            
        var_t = (self.model.sqrt_one_minus_alphas_cumprod[t])**2 
        alpha_t = torch.sqrt(1 - var_t)
        alpha_t_s = alpha_t.reshape(len(var_t), 1 ,1 ,1) /  alpha_s
        var_t_s = var_t.reshape(len(var_t), 1 ,1 ,1) - alpha_t_s**2 * var_s 
        return alpha_t_s, torch.sqrt(var_t_s)

    def qpos_given_te(self, t, s, t_star, z_t_star, z_t, zeta_T_star=None): 
        alpha_t_s, scale_t_s = self.q_given_te(t, s, z_t_star.shape)
        alpha_s_t_star, scale_s_t_star = self.q_given_te(s, t_star, z_t_star.shape, zeta_T_star)

        var = scale_t_s**2 * scale_s_t_star**2 / (scale_t_s**2 + alpha_s_t_star**2 * scale_s_t_star**2 )
        mean = (var) * ( (alpha_s_t_star/scale_s_t_star**2) * z_t_star +  (alpha_t_s/scale_t_s**2) * z_t )
        return mean, torch.sqrt(var)
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def get_error(self,x,t,c, unconditional_conditioning, unconditional_guidance_scale):

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x.float(), t, c) 
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        return e_t

    def descretize(self, rho):
        #Get time descretization for prior loss (t > T_e)
        self.timesteps_1000 = time_descretization(sigma_min=0.002, sigma_max = 0.999, rho = rho, num_t_steps = 1000)*1000
        self.timesteps_1000 = self.timesteps_1000.cuda().long()
        sigma_timesteps = self.model.sqrt_one_minus_alphas_cumprod[self.timesteps_1000] 
        self.register_buffer('sigma_timesteps', sigma_timesteps)

        #Get prior std for hierarchical time points
        sigma_hierarchy = self.model.sqrt_one_minus_alphas_cumprod[self.t_steps_hierarchy]
        self.t_steps_hierarchy = torch.tensor(self.t_steps_hierarchy.copy()).cuda()        
        alphas_h = 1  - sigma_hierarchy**2
        alphas_prev =  torch.concatenate([ alphas_h[1:], alphas_h[-1].reshape(1)]) 
        h_sigmas = torch.sqrt(self.eta * (1 - alphas_prev) / (1 - alphas_h) * (1 - alphas_h / alphas_prev) )
        h_sigmas[1:] = torch.sqrt(self.eta * (1 - alphas_prev[:-1]) / (1 - alphas_h[:-1]) * (1 - alphas_h[:-1] / alphas_prev[:-1]) )
        h_sigmas[0] = torch.sqrt(1 - alphas_h[0])

        #register tensors
        self.register_buffer('h_alphas', alphas_h)
        self.register_buffer('h_alphas_prev', alphas_prev)
        self.register_buffer('h_sigmas', h_sigmas)

    def init(self, img, std_scale, mean_scale, prior_scale, mean_scale_top = 0.1):
        num_h_steps = len(self.t_steps_hierarchy)
        img  = torch.Tensor.repeat(img,[num_h_steps,1,1,1])[:num_h_steps]
        #sigmas = self.h_sigmas[...,None, None, None].expand(img.shape) 
        sigmas = torch.zeros_like(img)
        sqrt_alphas = torch.sqrt(self.h_alphas)[...,None, None, None].expand(img.shape) 
        sqrt_one_minus_alphas = torch.sqrt(1 - self.h_alphas)[...,None, None, None].expand(img.shape) 
        ## Variances for posterior
        sigmas[0] = self.h_sigmas[0, None, None, None].expand(img[0].shape) 
        sigmas[1:] = std_scale * (1/np.sqrt(self.eta)) * self.h_sigmas[1:, None, None, None].expand(img[1:].shape)
        logvar_pos = 2*torch.log(sigmas).float()
        ## Means :  
        mean_pos = sqrt_alphas*img + mean_scale*sqrt_one_minus_alphas* torch.randn_like(img) 
        mean_pos[0] = img[0] + mean_scale_top*torch.randn_like(img[0])
        ## Gammas for posterior weighing between prior and posterior
        gamma = torch.tensor(prior_scale)[None,None,None,None].expand(img.shape).cuda() 
        return  mean_pos, logvar_pos, gamma.float()
    
    def get_kl(self,mu1, mu2, scale1, scale2, wt):
        return wt*(1/2*scale2**2)*(mu1 - mu2)**2 \
            + torch.log(scale2/scale1) + scale1**2/(2*scale2**2) - 1/2
        
    def loss_prior(self, mu_pos, logvar_pos, cond=None, 
                    unconditional_conditioning=None, 
                    unconditional_guidance_scale=1,  K=10, intermediate_mus=None):
        '''
        This function gets the kl between q(x_{T_e})||p(x_T_e) ) = E_{t>T*_e}[(x_T_e - \mu_\theta(x_t))^2]
        x_T_e = z_t_star, samples from q(x_{T_e})
        Sample z_t by adding noise scaled by sqrt(\sigma_t^2 - \zeta_t^2) so that z_t matches total noise at t
        '''
        t_e =  self.t_steps_hierarchy[0]
        ## Sample z_{T_e}
        tau_te = torch.exp(0.5*logvar_pos) 
        mu_te = torch.Tensor.repeat(mu_pos, [K,1,1,1])
        z_te = torch.sqrt(1 - tau_te**2 )* mu_te + tau_te * torch.randn_like(mu_te) 
        
        ## Sample t
        #Get allowed timesteps > T_e
        t_g = torch.where(self.sigma_timesteps > torch.max(tau_te))[0]
        t_allowed = self.timesteps_1000[t_g]
        print(len(t_g))
        def sample_uniform(t_allowed):
            t0 = torch.rand(1)
            T_max = len(t_allowed) 
            T_min = 2               #stay away from close values to T*
            t = torch.remainder(t0 + torch.arange(0., 1., step=1. / K), 1.)*(T_max-T_min) + T_min
            t = torch.floor(t).long()
            return t
        t = sample_uniform(t_allowed)
        t_cur = t_allowed[t] 
        t_prev = t_allowed[t-1]
        print((t_cur - t_prev), t_cur)

        #sample z_t from p(z_t | z_{T_e})
        alpha_t, scale_t = self.q_given_te(t_cur, t_e, z_te.shape, tau_te)
        error =  torch.randn_like(z_te)
        z_t = alpha_t*z_te + error* scale_t

        #Get prior, posterior mean variances for t_prev
        e_out = self.get_error(z_t.float(), t_cur, cond, unconditional_conditioning, unconditional_guidance_scale)
        alpha_t_, scale_t_ = self.q_given_te(t_cur,t_e, z_te.shape)
        mu_t_hat = (z_t - scale_t_*e_out)/alpha_t_
        pos_mean, pos_scale = self.qpos_given_te(t_cur, t_prev, t_e, z_te, z_t, tau_te)
        prior_mean, prior_scale = self.qpos_given_te(t_cur, t_prev, t_e, mu_t_hat, z_t, None)

        wt = (1000-t_e)/2
        kl = self.get_kl(pos_mean, prior_mean,pos_scale, prior_scale, wt=1)
        kl = torch.mean(wt*kl, dim=[1,2,3])
  
        return {"loss" : kl, "sample" : z_te, "intermediate_mus" : intermediate_mus}

    def recon_loss(self, samples_pixel, x0_pixel, mask_pixel):
        global_step = 0
        if self.first_stage == "kl":
            nll_loss, _ = self.vae_loss(x0_pixel, samples_pixel, mask_pixel, 0, global_step,
                                    last_layer=self.model.first_stage_model.get_last_layer(), split="val")
        else: 
            qloss = torch.tensor([0.]).cuda()
            nll_loss, _ = self.vae_loss(qloss, x0_pixel, samples_pixel, mask_pixel, 0, 0,
                                        last_layer=self.model.first_stage_model.get_last_layer(), split="val",
                                        predicted_indices=None)
        nll_loss = nll_loss/1000
        return { "loss" : nll_loss}

    def prior_preds(self, z_t, t_cur, cond, a_t, a_prev, sigma_t, unconditional_conditioning, unconditional_guidance_scale ):
        #Get e, pred_x0
        e_out = self.get_error(z_t, t_cur, cond, unconditional_conditioning, unconditional_guidance_scale)
        pred_x0 = (z_t - torch.sqrt(1 - a_t)  * e_out) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev -  sigma_t**2).sqrt() * e_out             
        z_next = a_prev.sqrt() * pred_x0  + dir_xt 
        return z_next, pred_x0

    def posterior_mean(self, mu_pos, mu_prior, gamma):
        wt = torch.sigmoid(gamma)
        mean_t_1 = wt*mu_prior + (1-wt)*mu_pos
        return mean_t_1   
    
    def normalize(self, img):
        img -= torch.min(img)
        return 2*img/torch.max(img) - 1
    
    def loss_posterior(self, z_t, mu_pos, logvar_pos, gamma, cond=None, 
                    unconditional_conditioning=None, 
                    unconditional_guidance_scale=1, 
                     K=10, iteration=0, to_sample = False, intermediate_mus=None):
        
        sigma_pos = torch.exp(0.5*logvar_pos)
        kl_t, t0, q_entropy = torch.zeros(z_t.shape[0]).cuda(), 100, 0
        num_steps = len(self.t_steps_hierarchy)
        intermediate_samples = np.zeros((num_steps, 1, 256, 256, 3))
        intermediate_preds = np.zeros((num_steps, 1, 256, 256, 3))
        b = z_t.shape[0]
        with torch.no_grad():
            recon = self.model.decode_first_stage(z_t)
            intermediate_samples[0] = to_img(recon)[0]
        
        alphas = self.h_alphas 
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps_hierarchy[:-1], self.t_steps_hierarchy[1:])):
            t_hat_cur = torch.ones(b).cuda() * (t_cur ) 
            a_t =  torch.full((b, 1, 1, 1), alphas[i]).cuda()
            a_prev = torch.full((b, 1, 1, 1), alphas[i+1]).cuda()
            a_t_prev = a_t/a_prev
            sigma_t = self.h_sigmas[i+1] 
            #Get prior predictions
            z_next, pred_x0 = self.prior_preds(z_t.float(), t_hat_cur, cond, a_t, a_prev, sigma_t,
                                                       unconditional_conditioning, unconditional_guidance_scale)
            std_prior = self.h_sigmas[i+1] 

            ##Posterior means and variances  
            pos_mean = self.posterior_mean(a_prev.sqrt()*mu_pos[i].unsqueeze(0), z_next, gamma[i].unsqueeze(0))
            std_pos = sigma_pos[i]

            ## Sample z_t
            z_t =  pos_mean + std_pos * torch.randn_like(pos_mean) 
            #Get kl 
            kl = self.get_kl(pos_mean, z_next, std_pos, std_prior, wt=1)
            kl_t += torch.mean(kl, dim=[1,2,3])

            with torch.no_grad():
                recon_pred = self.model.decode_first_stage(pred_x0)
                intermediate_preds[i] = to_img(recon_pred)[0] 
                intermediate_mus[i+1] = to_img(self.normalize(mu_pos[i]).unsqueeze(0)).astype(np.uint8)[0] 

        ##One-step denoising
        t_hat_cur = torch.ones(b).cuda() * (self.t_steps_hierarchy[-1])
        e_out = self.get_error(z_t.float(), t_hat_cur, cond, unconditional_conditioning, unconditional_guidance_scale)
        a_t = torch.full((b, 1, 1, 1), alphas[-1]).cuda()
        sqrt_one_minus_at = torch.sqrt(1 - a_t) 
        pred_z0 = (z_t - sqrt_one_minus_at * e_out) / a_t.sqrt()

        with torch.no_grad():
            recon = self.model.decode_first_stage(pred_z0)
            intermediate_preds[-1] =  to_img(recon)[0]  

        return {"sample" : pred_z0, "loss" : kl_t, "entropy": q_entropy,
                 "intermediates" : intermediate_samples, "interim_preds"  :intermediate_preds, 
                 "intermediate_mus" : intermediate_mus} 

    def grad_and_value(self, x_prev, x_0_hat, measurement, mask_pixel):
        nll_loss = self.recon_loss(x_0_hat, measurement, mask_pixel)["loss"]
        norm_grad = torch.autograd.grad(outputs=nll_loss, inputs=x_prev)[0] 
        return norm_grad, nll_loss

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, mask_pixel, scale,  **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, 
                                            measurement=measurement, mask_pixel=mask_pixel)
        x_t -= norm_grad*scale 
        return x_t, norm
    
    def sample(self, scale, eta, mu_pos, logvar_pos, gamma,  
               mask_pixel, y, n_samples=100,  cond=None, 
               unconditional_conditioning=None, unconditional_guidance_scale=1, 
               batch_size=10, dir_name="temp/", temp=1):
        sigma_pos = torch.exp(0.5*logvar_pos)
        t0 = 100 
        num_steps = len(self.t_steps_hierarchy)
        intermediate_samples = np.zeros((num_steps, 1, 256, 256, 3))
        intermediate_preds = np.zeros((num_steps, 1, 256, 256, 3))
        intermediate_mus = np.zeros((num_steps, 1, 256, 256, 3))
        alphas = self.h_alphas

        ##batch your sample generation
        all_images = []
        t0 = time.time()
        save_dir = os.path.join(dir_name , "samples_" + str(scale))
        os.makedirs(save_dir, exist_ok=True)
        for _ in trange(n_samples // batch_size, desc="Sampling Batches"):
            mu_10 = torch.Tensor.repeat(mu_pos[0], [batch_size,1,1,1])
            tau_t = sigma_pos[0]
            z_t = torch.sqrt(1 - tau_t**2 )* mu_10 + tau_t * torch.randn_like(mu_10) 
            ##Sample from posterior
            with torch.no_grad():
                recon = self.model.decode_first_stage(z_t)
                intermediate_samples[0] = to_img(recon)[0]
                for i, (t_cur, t_next) in enumerate(zip(self.t_steps_hierarchy[:-1], self.t_steps_hierarchy[1:])):
                    t_hat_cur = torch.ones(batch_size).cuda() * (t_cur ) 
                    a_t =  torch.full((batch_size, 1, 1, 1), alphas[i]).cuda()
                    a_prev = torch.full((batch_size, 1, 1, 1), alphas[i+1]).cuda()
                    sigma_t = self.h_sigmas[i+1] 
                    #Get prior predictions
                    z_next, pred_x0 = self.prior_preds(z_t.float(), t_hat_cur, cond, a_t, a_prev, sigma_t, 
                                                       unconditional_conditioning, unconditional_guidance_scale)
                    ##Posterior means and variances 
                    # a_prev.sqrt()*                  
                    mean_t_1 = self.posterior_mean(a_prev.sqrt()*mu_pos[i+1].unsqueeze(0), z_next, gamma[i+1].unsqueeze(0))
                    std_pos = sigma_pos[i+1]
                    #Sample z_t
                    z_t =  mean_t_1 + std_pos * torch.randn_like(mean_t_1)  
                    with torch.no_grad():
                        pred_x = self.model.decode_first_stage(pred_x0)
                        save_samples(save_dir, pred_x, k=None, num_to_save = 1, file_name =  f'sample_{i}.png')
                    
            timesteps = np.flip(np.arange(0, self.t_steps_hierarchy[-1].cpu().numpy(), 1))
            timesteps = np.concatenate((self.t_steps_hierarchy[-1].cpu().reshape(1), timesteps))
            ##Sample using DPS algorithm 
            for i, (step, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                step = int(step)
                t_hat_cur = torch.ones(batch_size).cuda() * (step)
                a_t =  torch.full((batch_size, 1, 1, 1), self.model.alphas_cumprod[step]).cuda()
                a_prev = torch.full((batch_size, 1, 1, 1), self.model.alphas_cumprod[int(t_next)]).cuda()
                sigma_t = eta *torch.sqrt( (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
                z_t = z_t.requires_grad_()
                z_next, pred_x0 = self.prior_preds(z_t.float(), t_hat_cur, cond, a_t, a_prev, sigma_t, 
                                                       unconditional_conditioning, unconditional_guidance_scale)
                pred_x = self.model.decode_first_stage(pred_x0)
                z_t, _ = self.conditioning(x_prev = z_t , x_t = z_next, 
                                              x_0_hat = pred_x, measurement = y, 
                                              mask_pixel=mask_pixel, scale=scale)
                z_t = z_t.detach_()
                if i%50 == 0:
                    with torch.no_grad():
                        recons = self.model.decode_first_stage(pred_x0)
                        save_samples(save_dir, recons, k=None, num_to_save = 1, file_name =  f'det_{step}.png')

            z_0 = pred_x0 
            with torch.no_grad():
                recon = self.model.decode_first_stage(z_0)
                intermediate_preds[-1] = to_img(recons)[0]
            
            with torch.no_grad() : recons = self.model.decode_first_stage(pred_x0)
            all_images.append(custom_to_np(recons))

        t1 = time.time()
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(save_dir, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img, t1-t0)
        
        save_inpaintings(save_dir, recons, y, mask_pixel, num_to_save = batch_size)

        return 

    def fit(self, lambda_, cond, shape, quantize_denoised=False, mask_pixel = None, 
            y = None, log_every_t=100, unconditional_guidance_scale=1., 
            unconditional_conditioning=None, dir_name = None, kl_weight=50, 
            debug=False, wdb=False, iterations=200, batch_size = 10, lr_init_gamma=0.01): 
                
        if wdb: 
            wandb.init(project='LDM', dir = '/scratch/sakshi/wandb-cache')
            wandb.config.run_type = 'hierarchical'
            wandb.run.name = "hierarchical"

        params_to_fit = params_train(lambda_) 
        mu_pos, logvar_pos, gamma = params_to_fit
        optimizers, schedulers = get_optimizers(mu_pos, logvar_pos, gamma, lr_init_gamma)
        rec_loss_all, prior_loss_all, posterior_loss_all =[], [], []
        mu_all, logvar_all, gamma_all = [], [], []
        for k in range(iterations):
            if k%100==0: print(k)
            intermediate_mus = np.zeros((len(self.t_steps_hierarchy), 64, 64, 3))
            for opt in optimizers: opt.zero_grad()
            stats_prior = self.loss_prior(mu_pos[0], logvar_pos[0], cond=cond, 
                                          unconditional_conditioning=unconditional_conditioning, 
                                          unconditional_guidance_scale=unconditional_guidance_scale, 
                                          K=batch_size, intermediate_mus=intermediate_mus)
            #stats_posterior = self.get_z0_t(stats_prior["sample"], self.t_steps_hierarchy)
            stats_posterior = self.loss_posterior(stats_prior["sample"], mu_pos[1:], logvar_pos[1:], gamma[1:],
                                                  cond=cond,
                                                    unconditional_conditioning=unconditional_conditioning, 
                                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                                    K=batch_size, iteration=k, intermediate_mus=stats_prior["intermediate_mus"])
            sample = self.model.decode_first_stage(stats_posterior["sample"])
            
            stats_recon = self.recon_loss(sample, y, mask_pixel)
            loss_total = torch.mean(kl_weight*stats_prior["loss"] \
                                    + kl_weight*stats_posterior["loss"] + stats_recon["loss"] ) #
            loss_total.backward()
            for opt in optimizers: opt.step()
            for sch in schedulers: sch.step()
            
            rec_loss_all.append(torch.mean(stats_recon["loss"].detach()).item())
            prior_loss_all.append(torch.mean(kl_weight*stats_prior["loss"].detach()).item())
            posterior_loss_all.append(torch.mean(kl_weight*stats_posterior["loss"].detach()).item())
            mu_all.append(torch.mean(mu_pos.detach()).item())
            logvar_all.append(torch.mean(logvar_pos.detach()).item())
            gamma_all.append(torch.mean(torch.sigmoid(gamma).detach()).item())                   
            sample_np = to_img(sample).astype(np.uint8)
            
            if wdb:
                wandb.log(dict(total_loss=loss_total.detach().item(),
                            kl = torch.mean(kl_weight*stats_prior["loss"]).detach().item(),
                            dec_loss= torch.mean(stats_recon["loss"]).detach().item(),
                            intermediate_samples = [wandb.Image(image) for image in stats_posterior["intermediates"]],
                            intermediate_mus = [wandb.Image(image) for image in stats_posterior["intermediate_mus"]],
                            interim_preds = [wandb.Image(image) for image in stats_posterior["interim_preds"]],
                            params = [wandb.Image(image) for image in mu_pos.detach()],
                            gamma = [wandb.Image(image) for image in gamma.detach()], 
                            std = [wandb.Image(torch.exp(0.5*image)) for image in logvar_pos.detach()],
                            klt= torch.mean(kl_weight*stats_posterior["loss"]).detach().item(),
                            lr=optimizers[0].state_dict()['param_groups'][0]['lr'],
                            std0=torch.mean(torch.exp(0.5*logvar_pos[0])).detach(), 
                            samples = [wandb.Image(image) for image in sample_np],
                            ))
                
            save_plot(dir_name, [rec_loss_all, prior_loss_all, posterior_loss_all],
                          ["Recon loss", "Prior loss (>T_e)", "Posterior loss (>T_s)"], "loss.png")
            save_plot(dir_name, [mu_all],
                          ["mean"], "mean.png") 
            save_plot(dir_name, [logvar_all],
                          ["logvar"], "logvar.png") 
            save_plot(dir_name, [gamma_all],
                          ["gamma"], "gamma.png") 
            
            if k%log_every_t == 0 or k == iterations - 1:
                save_samples(os.path.join(dir_name , "progress"), sample, k, 5)
                save_samples(os.path.join(dir_name , "mus"), stats_posterior["intermediate_mus"], k,
                              len(stats_posterior["intermediate_mus"]))
                
                save_inpaintings(os.path.join(dir_name , "progress_inpaintings"), sample, y,
                                  mask_pixel, k, num_to_save = 5)
                save_params(os.path.join(dir_name , "params"), mu_pos, logvar_pos, gamma,k)
                
            gc.collect()
        return 
    
##unconditional samplinng for debugging purposes:
'''
    def sample_T(self, x0, cond, unconditional_conditioning, unconditional_guidance_scale , eta=0.4, t_steps_hierarchy=None, dir_="out_temp2"):
        ''
        sigma_discretization_edm = time_descretization(sigma_min=0.002, sigma_max = 999, rho = 7, num_t_steps = 10)/1000
        T_max = 1000
        beta_start  = 1 # 0.0015*T_max
        beta_end = 15 # 0.0155*T_max
        def var(t):
            return 1.0 - (1.0) * torch.exp(- beta_start * t - 0.5 * (beta_end - beta_start) * t * t)
        ''

        x0 = torch.randn_like(x0) 
        t_steps_hierarchy = torch.tensor(self.t_steps_hierarchy).cuda()
        var_t =  (self.model.sqrt_one_minus_alphas_cumprod[t_steps_hierarchy[0]].reshape(1, 1 ,1 ,1))**2 # self.var(t_steps_hierarchy[0])
        x_t = x0 # torch.sqrt(1 - var_t) * x0 + torch.sqrt(var_t) * torch.randn_like(x0)

        os.makedirs(dir_, exist_ok=True)
        alphas = self.h_alphas
        b = 5
        for i, t in enumerate(t_steps_hierarchy[:-1]): 
            t_hat = torch.ones(b).cuda() * (t)
            a_t =  torch.full((b, 1, 1, 1), alphas[i]).cuda()
            a_prev = torch.full((b, 1, 1, 1), alphas[i+1]).cuda()
            sigma_t = self.h_sigmas[i+1] 
            x_t, pred_x0 = self.prior_preds(x_t.float(), t_hat, cond, a_t, a_prev, sigma_t,
                                                unconditional_conditioning, unconditional_guidance_scale)

            var_t = (self.model.sqrt_one_minus_alphas_cumprod[t].reshape(1, 1 ,1 ,1))**2
            a_t = 1 - var_t 
            x_t = x_t + sigma_t*torch.randn_like(x_t)
            recon = self.model.decode_first_stage(pred_x0)
            image_path = os.path.join(dir_, f'{i}.png')
            image_np = (recon.detach() * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        t_hat_cur = torch.ones(b).cuda() * (self.t_steps_hierarchy[-1])
        e_out = self.get_error(x_t.float(), t_hat_cur, cond, unconditional_conditioning, unconditional_guidance_scale)
        a_t = torch.full((b, 1, 1, 1), alphas[-1]).cuda()
        sqrt_one_minus_at = torch.sqrt(1 - a_t) 
        pred_x0 = (x_t - sqrt_one_minus_at * e_out) / a_t.sqrt()
        
        recon = self.model.decode_first_stage(pred_x0)
        image_path = os.path.join(dir_, f'{len(t_steps_hierarchy)}.png')
        image_np = (recon.detach() * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        return

'''