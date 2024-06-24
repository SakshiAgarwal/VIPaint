import torch
import numpy as np
import os
import pickle
from ldm.util import default
import glob
import PIL
import matplotlib.pyplot as plt

def load_file(filename):
    with open(filename , 'rb') as file:
        x = pickle.load(file)
    return x

def save_file(filename, x, mode="wb"):
        with open(filename, mode) as file:
            pickle.dump(x, file)

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))

def to_img(sample):
    return (sample.detach().cpu().numpy().transpose(0,2,3,1) * 127.5 + 128).clip(0, 255)

def save_plot(dir_name, tensors, labels, file_name="loss.png"):
    t = np.linspace(0, len(tensors[0]), len(tensors[0]))
    colours = ["r", "b", "g"]
    plt.figure()
    for j in range(len(tensors)):
        plt.plot(t, tensors[j],color =  colours[j], label = labels[j])
    plt.legend()
    plt.savefig(os.path.join(dir_name, file_name))
    #plt.show()
    
def save_samples(dir_name, sample, k=None, num_to_save = 5, file_name = None):
    if type(sample) is not np.ndarray: sample_np = to_img(sample).astype(np.uint8)
    else: sample_np = sample.astype(np.uint8)
    
    for j in range(num_to_save): 
        if file_name is None: 
            if k is not None: file_name_img =  f'sample_{k+1}'f'{j}.png'
            else:  file_name_img = f'{j}.png'
        else: file_name_img = file_name
        image_path = os.path.join(dir_name,file_name_img)
        image_np = sample_np[j]
        PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        file_name_img = None

def save_inpaintings(dir_name, sample, y, mask_pixel, k=None, num_to_save = 5, file_name = None):
    recon_in = y*(mask_pixel) + ( 1-mask_pixel)*sample
    recon_in = to_img(recon_in)
    for j in range(num_to_save): 
        if file_name is None: 
            if k is not None: file_name_img =  f'sample_{k+1}'f'{j}.png'
            else:  file_name_img = f'{j}.png'
        else: file_name_img = file_name
        image_path = os.path.join(dir_name, file_name_img)
        image_np = recon_in.astype(np.uint8)[j]
        PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        file_name_img = None

def save_params(dir_name, mu_pos, logvar_pos, gamma,k):
    params_to_fit = params_untrain([mu_pos.detach().cpu(), logvar_pos.detach().cpu(), gamma.detach().cpu()])
    params_path = os.path.join(dir_name, f'{k+1}.pt')
    torch.save(params_to_fit, params_path)

def custom_to_np(img): 
    sample = img.detach().cpu()
    #sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    #sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def encoder_kl(diff, img):
    _, params = diff.encode_first_stage(img, return_all = True)
    params = diff.scale_factor * params
    mean, logvar = torch.chunk(params, 2, dim=1)
    noise = default(None, lambda: torch.randn_like(mean))
    mean = mean + diff.scale_factor*noise 
    return mean, logvar
      
def encoder_vq(diff, img):
    quant = diff.encode_first_stage(img) #, diff, (_,_,ind)
    quant = diff.scale_factor * quant
    #mean, logvar = torch.chunk(params, 2, dim=1)
    noise = default(None, lambda: torch.randn_like(quant))
    mean = quant + diff.scale_factor*noise #
    return mean

def clean_directory(dir_name):
    files = glob.glob(dir_name)
    for f in files:
        os.remove(f)

def params_train( params ): 
    for item in params: 
        item.requires_grad = True
    return params

def params_untrain(params):
    for item in params: 
        item.requires_grad = False
    return params

def time_descretization(sigma_min=0.002, sigma_max = 80, rho = 7, num_t_steps = 18):
    step_indices = torch.arange(num_t_steps, dtype=torch.float64).cuda()
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_t_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    inv_idx = torch.arange(num_t_steps -1, -1, -1).long()
    t_steps_fwd = t_steps[inv_idx]
    #t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps_fwd

def get_optimizers(means, variances, gamma_param, lr_init_gamma=0.01) :
    [lr, step_size, gamma] = [0.1, 10, 0.99]  #was 0.999  for right-half: [0.01, 10, 0.99] 
    optimizer = torch.optim.Adam([means], lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    optimizer_2 = torch.optim.Adam([variances], lr=0.001, betas=(0.9, 0.99)) #0.001 for lsun
    optimizer_3 = torch.optim.Adam([gamma_param], lr=lr_init_gamma, betas=(0.9, 0.99)) #0.01

    scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=step_size, gamma=gamma) ##added this
    scheduler_3 = torch.optim.lr_scheduler.StepLR(optimizer_3, step_size=step_size, gamma=gamma)

    return [optimizer, optimizer_2, optimizer_3 ], [scheduler, scheduler_2,  scheduler_3]

def check_directory(filename_list):
    for filename in filename_list:
        if not os.path.exists(filename):
            os.mkdir(filename)

def s_file(filename, x, mode="wb"):
    with open(filename, mode) as file:
        pickle.dump(x, file)

def r_file(filename, mode="rb"): 
    with open(filename, mode) as file:
        x = pickle.load(file)   
    return x 

def sample_from_gaussian(mu, alpha, sigma):
    noise = torch.randn_like(mu)
    return alpha*mu + sigma * noise

'''
def make_batch(image, mask=None, device=None):
    image = torch.permute(image, (0,3,1,2)) 
    batch_size = image.shape[0]
    if mask is None : 
        mask = torch.zeros_like(image)
        mask[0, :, :256, :128] = 1
    else : 
        mask = torch.tensor(mask)
    masked_image = (mask)*image #+ mask*noise*0.2
    mask = mask[:,0,:,:].reshape(batch_size,1,image.shape[2], image.shape[3])
    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device)
    return batch

def get_sigma_t_steps(net, n_steps=3, kwargs=None):
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    ##Get the time-steps based on iddpm discretization
    num_steps = n_steps #11 # kwargs["num_steps"]
    C_2 = kwargs["C_2"]
    C_1 = kwargs["C_1"]
    M = kwargs["M"]
    step_indices = torch.arange(num_steps, dtype=torch.float64).cuda()
    u = torch.zeros(M + 1, dtype=torch.float64).cuda()
    alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
    for j in torch.arange(M, 0, -1, device=step_indices.device): # M, ..., 1
        u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
    u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
    sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    #print(sigma_steps)

    ##get noise schedule
    sigma = lambda t: t
    sigma_deriv = lambda t: 1
    sigma_inv = lambda sigma: sigma

    ##scaling schedule 
    s = lambda t: 1
    s_deriv = lambda t: 0

    ##compute some final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))

    return t_steps, sigma_inv, sigma, s, sigma_deriv

def data_replicate(data, K):
    if len(data.shape)==2: data_batch = torch.Tensor.repeat(data,[K,1]) 
    else: data_batch = torch.Tensor.repeat(data,[K,1,1,1]) 
    return data_batch

'''


def sample_T(self, x0, eta=0.4, t_steps_hierarchy=None):
    '''
    sigma_discretization_edm = time_descretization(sigma_min=0.002, sigma_max = 999, rho = 7, num_t_steps = 10)/1000
    T_max = 1000
    beta_start  = 1 # 0.0015*T_max
    beta_end = 15 # 0.0155*T_max
    def var(t):
        return 1.0 - (1.0) * torch.exp(- beta_start * t - 0.5 * (beta_end - beta_start) * t * t)
    '''
    t_steps_hierarchy = torch.tensor(t_steps_hierarchy).cuda()
    var_t =  (self.model.sqrt_one_minus_alphas_cumprod[t_steps_hierarchy[0]].reshape(1, 1 ,1 ,1))**2 # self.var(t_steps_hierarchy[0])
    x_t = torch.sqrt(1 - var_t) * x0 + torch.sqrt(var_t) * torch.randn_like(x0)

    os.makedirs("out_temp2/", exist_ok=True)
    for i, t in enumerate(t_steps_hierarchy): 
        t_hat = torch.ones(10).cuda() * (t)
        e_out = self.model.model(x_t, t_hat)
        var_t = (self.model.sqrt_one_minus_alphas_cumprod[t].reshape(1, 1 ,1 ,1))**2
        #score_out = - e_out / torch.sqrt()
        a_t = 1 - var_t 
        #beta_t = 1 - a_t/a_prev
        #std_pos = ((1 - a_prev)/(1 - a_t)).sqrt()*torch.sqrt(beta_t)
        pred_x0 = (x_t - torch.sqrt(1 - a_t) * e_out) / a_t.sqrt()

        if i != len(t_steps_hierarchy) - 1: 
            var_t1 = (self.model.sqrt_one_minus_alphas_cumprod[t_steps_hierarchy[i+1]].reshape(1, 1 ,1 ,1))**2
            a_prev = 1 - var_t1 # var(t_steps_hierarchy[i+1]/1000) # torch.full((10, 1, 1, 1), alphas[t_steps_hierarchy[i+1]]).cuda()
            sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_out
            x_t = a_prev.sqrt() * pred_x0 + dir_xt + torch.randn_like(x_t) * sigma_t + sigma_t*torch.randn_like(x_t)
            
        #x_t= (x_t - torch.sqrt( 1 - a_t/a_prev) * e_out ) / (a_t/a_prev).sqrt() + std_pos*torch.randn_like(x_t)

        '''
        def pred_mean(pred_x0, z_t):
            posterior_mean_coef1 = beta_t * torch.sqrt(a_prev) / (1. - a_t)
            posterior_mean_coef2 = (1. - a_prev) * torch.sqrt(a_t/a_prev) / (1. - a_t)
            return posterior_mean_coef1*pred_x0 + posterior_mean_coef2*z_t
        
        x_t = torch.sqrt(a_prev) * pred_x0  # pred_mean(pred_x0, x_t) #+ 0.4*torch.sqrt(beta_t) *torch.randn_like(x_t)
        '''
        recon = self.model.decode_first_stage(pred_x0)
        image_path = os.path.join("out_temp2/", f'{i}.png')
        image_np = (recon.detach() * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    return

    