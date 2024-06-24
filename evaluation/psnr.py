#import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
#from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure     #StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from evaluation.lpips import LPIPS
import torchvision
import numpy as np
import torchvision.transforms as transforms

def get_psnr_metrics(case, seq, image_loader, dir_algo, algo = "blended",
                     files={},samples_qavi="final_samples2/", common_dir="evaluation", start=0, stop=0):
    psnrs = []
    ssims = []
    lpips = []
    lpips_all = []
    lpips_min = []
    lpips_max = []

    perceptual_loss = LPIPS().cuda().eval()
    n_samples = 10
    if start==0 and stop==0:
        if case == "random_all":
            start = 0
            stop = 100
        else:
            start = 100
            stop = 200
    
    for num, num_random in enumerate(seq): #[start:stop]
    #for b1, b2 in tqdm(loader):
        #Read file for num
        if num>=100: break

        dir_name = dir_algo + str(int(num+50)%100) + "/"
        if case == "random_half" : filename = files[case]
        else: filename = files[case][algo]

        if algo == "qavi" or algo == "qavi_10" : 
            dir_samples = dir_name + samples_qavi + filename
        else: 
            dir_samples = dir_name + filename
        
        dataset2 = np.load(dir_samples)
        x2 = dataset2['arr_0']
        x2 = x2[:n_samples]
        #if algo == "qavi" or algo == "dps": 
        x2 = ((torch.tensor(x2) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = (torch.tensor(x2).float()/127.5 - 1).cuda().permute(0,3,1,2)

        t_mins = dataset2['arr_1']/60

        image = image_loader.dataset[num_random]
        x1 = torch.tensor(image).reshape(1,256,256,3)
        x1 = torch.permute(x1, (0,3,1,2)) 
        x1 = torch.Tensor.repeat(x1, [x2.shape[0],1,1,1])
        
        #x1, x2 = b1[0], b2[0]
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
        #lpips_max.append(torch.max(lpips_single).item())

    psnrs = torch.cat(psnrs, dim=0)
    ssims = torch.cat(ssims, dim=0)
    lpips = torch.tensor(lpips).cuda()
    lpips_all = np.concatenate(lpips_all)
    lpips_min = torch.tensor(lpips_min).cuda()
    lpips_max = torch.tensor(lpips_max).cuda()

    results_dir = "evaluation" #+ str(algo) 
    os.makedirs(results_dir, exist_ok=True)
    shape_str = "x".join([str(x) for x in lpips_all.shape])
    nppath = os.path.join(results_dir, f"{shape_str}-lpips.npz")
    np.savez(nppath, lpips_all)
    #psnrs_list = [torch.zeros_like(psnrs) for i in range(dist.get_world_size())]
    #ssims_list = [torch.zeros_like(ssims) for i in range(dist.get_world_size())]
    #lpips_list = [torch.zeros_like(lpips) for i in range(dist.get_world_size())]
    #dist.gather(psnrs, psnrs_list, dst=0)
    #dist.gather(ssims, ssims_list, dst=0)
    #dist.gather(lpips, lpips_list, dst=0)

    #if dist.get_rank() == 0:
    results_file = common_dir + "/psnr" + ".txt" # get_results_file(cfg, logger)
    #psnrs = torch.cat(psnrs, dim=0)
    #ssims = torch.cat(ssims, dim=0)
    #lpips = torch.cat(lpips, dim=0)
    #logger.info(f"PSNR: {psnrs.mean().item()} +/- {psnrs.std().item()}")
    #logger.info(f"SSIM: {ssims.mean().item()}")
    #logger.info(f"LPIPS: {lpips.mean().item()}")

    with open(results_file, 'a') as f:
        f.write(f'\n ALgorithm: {algo}\n')
        f.write(f'PSNR: {psnrs.mean().item()}\n')
        f.write(f'SSIM: {ssims.mean().item()}\n')
        f.write(f"LPIPS: {lpips.mean().item()}")
        f.write(f"LPIPS_MIN: {lpips_min.mean().item()}")
        f.write(f"LPIPS_MAX: {lpips_max.mean().item()}")
        f.write(f"Time: {t_mins}")

import os

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_ca(seq, dataset1, dir_algo, algo = "blended", files={} ):
    torch.hub.set_dir(os.path.join("evaluation/", 'hub'))
    #torch.cuda.set_device(dist.get_rank())
    exp_root = os.path.join("evaluation/", "ca_stats")
    os.makedirs(exp_root, exist_ok=True)
    model = torch.hub.load("pytorch/vision:v0.13.1", "resnet50", weights='IMAGENET1K_V1').cuda()    #, force_reload=True
    model.eval()
    top1, top5 = 0, 0
    count = 0

    mean = torch.tensor(np.array([0.485, 0.456, 0.406])).reshape(1,3,1,1)
    std = torch.tensor(np.array([0.229, 0.224, 0.225])).reshape(1,3,1,1)
    normalize = transforms.Normalize(mean,std )
    for num, num_random in enumerate(seq):
        #Read file for num
        if num>=10: break
        dir_name = dir_algo + str(num)  + "/"
        
        #normalize data
        if algo == "qavi" : 
            dir_samples = dir_name + "final_samples_5/701x1-samples.npz"
        elif algo == "repaint" and num==99:
            dir_samples = dir_name + "100x3x64x64-samples.npz"
        else: dir_samples = dir_name + files[algo] #"/100x64x64x3-samples.npz"
        dataset2 = np.load(dir_samples)
        x2 = dataset2['arr_0']
        if algo == "qavi" or algo ==  "dps": 
            x2 = ((torch.tensor(x2) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            x2 = x2.permute(0, 2, 3, 1)
        
        if algo == "repaint" and num == 99: 
            x2 = ((torch.tensor(x2) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            x2 = x2.permute(0, 2, 3, 1)
        x2 = torch.tensor(x2).cuda().permute(0,3,1,2)/255
        x2 = ((x2 - mean.cuda())/std.cuda()).float()
        ## data in the range of 0-1
        #x2 = (torch.tensor(x2).float()/(127.5) - 1 )
        n = x2.shape[0]
        #Get true image
        t_mins = dataset2['arr_1']/60
        data = dataset1[num_random]
        x1, label = data[:2]
        y = torch.tensor(label).cuda().reshape(1)
        #y = torch.nn.functional.one_hot(y, 1000).reshape(1,1000)
        y = torch.Tensor.repeat(y, [x2.shape[0]])
        with torch.no_grad():
            y_ = model(x2)
            t1, t5 = accuracy(y_, y, topk=(1, 5))
            top1 += t1 * n
            top5 += t5 * n
            count += n
    
    features = torch.tensor([top1, top5, count]).cpu()
    #features_list = [torch.zeros_like(features) for i in range(dist.get_world_size())]
    #dist.gather(features, features_list, dst=0)

    #if dist.get_rank() == 0:
    #features = torch.stack(features_list, dim=1)
    top1_tot = torch.sum(features[0], dim=0).item()
    top5_tot = torch.sum(features[1], dim=0).item()
    count_tot = torch.sum(features[2], dim=0).item()

    #logger.info(f"Top1: {top1_tot / count_tot}, Top5: {top5_tot / count_tot}.")
    results_file = "evaluation/ca_" + str(algo) + ".txt"  

    with open(results_file, 'a') as f:
        f.write(f"Total: {count_tot}\nTop1: {top1_tot / count_tot}\nTop5: {top5_tot / count_tot}\n")

    #dist.barrier()

