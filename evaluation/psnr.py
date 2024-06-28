import torch
import os
from torchmetrics.functional import structural_similarity_index_measure    
from evaluation.lpips import LPIPS
import numpy as np

def get_psnr_metrics(seq, image_loader, dir_algo, algo = "vipaint",
                     files={}, common_dir="evaluation"):
    psnrs = []
    ssims = []
    lpips = []
    lpips_all = []
    lpips_min = []
    lpips_max = []

    perceptual_loss = LPIPS().cuda().eval()
    n_samples = 10
    
    for num, num_random in enumerate(seq): 
        dir_name = os.path.join(dir_algo, str(int(num)))
        filename = files[algo]
        dir_samples = os.path.join(dir_name, "samples_2", filename)   
        dataset2 = np.load(dir_samples)
        x2 = dataset2['arr_0']
        x2 = x2[:n_samples]
        x2 = ((torch.tensor(x2) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = (torch.tensor(x2).float()/127.5 - 1).cuda().permute(0,3,1,2)

        t_mins = dataset2['arr_1']/60

        image = image_loader.dataset[num_random]
        x1 = torch.tensor(image).reshape(1,256,256,3)
        x1 = torch.permute(x1, (0,3,1,2)) 
        x1 = torch.Tensor.repeat(x1, [x2.shape[0],1,1,1])
        
        x1 = x1.cuda()
        x2 = x2.cuda()
        mse = torch.mean((x1 - x2) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(1 / (mse + 1e-10))
        ssim = structural_similarity_index_measure(x2, x1, reduction=None)
        
        with torch.no_grad():
            lpip = perceptual_loss(x2, x1).reshape(n_samples)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpips.append(lpip.mean().item())
        lpips_all.append(lpip.cpu().numpy())
        lpips_min.append(lpip.min().item())
        lpips_max.append(lpip.max().item())

    psnrs = torch.cat(psnrs, dim=0)
    ssims = torch.cat(ssims, dim=0)
    lpips = torch.tensor(lpips).cuda()
    lpips_all = np.concatenate(lpips_all)
    lpips_min = torch.tensor(lpips_min).cuda()
    lpips_max = torch.tensor(lpips_max).cuda()

    os.makedirs(common_dir, exist_ok=True)
    shape_str = "x".join([str(x) for x in lpips_all.shape])
    nppath = os.path.join(common_dir, f"{shape_str}-lpips.npz")
    np.savez(nppath, lpips_all)

    results_file = os.path.join(common_dir , "/psnr.txt" )
    with open(results_file, 'a') as f:
        f.write(f'\n ALgorithm: {algo}\n')
        f.write(f'PSNR: {psnrs.mean().item()}\n')
        f.write(f'SSIM: {ssims.mean().item()}\n')
        f.write(f"LPIPS: {lpips.mean().item()}")
        f.write(f"LPIPS_MIN: {lpips_min.mean().item()}")
        f.write(f"LPIPS_MAX: {lpips_max.mean().item()}")
        f.write(f"Time: {t_mins}")


