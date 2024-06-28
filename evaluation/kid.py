import torch
import numpy as np
import torchvision.transforms as transforms
import os
from cleanfid.inception_pytorch import InceptionV3
from cleanfid.fid import frechet_distance, kernel_distance
import torch.nn.functional as F

def get_kid( seq, image_loader, dir_algo, algo = "blended", files = {}, 
            common_dir = "evaluation"):

    torch.hub.set_dir(os.path.join("evaluation/", 'hub'))
    model = InceptionV3(output_blocks=[3], resize_input=False).cuda()
    model.eval()
    def model_fn(x): return model(x/255)[0].squeeze(-1).squeeze(-1)

    feats1, feats2 = [], []
    feats1_small, feats2_small = [], []

    moment1_s = None
    moment2_s = None
    size =0 
    moment1_t = None
    moment2_t = None
    n_samples = 10

    for num, num_random in enumerate(seq):
        dir_name = os.path.join(dir_algo, str(int(num)))
        filename = files[algo]
        dir_samples = os.path.joinn(dir_name, "sample_2", filename)
        dataset2 = np.load(dir_samples)
        x2 = dataset2['arr_0']
        x2 = x2[:n_samples]

        x2 = ((torch.tensor(x2) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = torch.tensor(x2).cuda().permute(0,3,1,2).float()

        image = image_loader.dataset[num_random]
        x1 = torch.tensor(image).reshape(1,256,256,3)
        x1 = (x1+1)*127.5
        x1 = torch.permute(x1, (0,3,1,2)) 
        x1 = torch.Tensor.repeat(x1, [x2.shape[0],1,1,1]).cuda()

        x1 = F.interpolate(x1, size=(299, 299), mode="bilinear", align_corners=False).clip(0, 255) #.permute((0, 2, 3, 1))
        x2 = F.interpolate(x2, size=(299, 299), mode="bilinear", align_corners=False).clip(0, 255) #.permute((0, 2, 3, 1))

        features_samples = model_fn(x2).cpu().numpy().astype(np.float64)
        features_true = model_fn(x1).cpu().numpy().astype(np.float64)

        def get_moments(moment1, moment2, size, features):
            if moment1 is None:
                moment1 = np.mean(features, axis=0)
                moment2 = np.cov(features, rowvar=False, ddof=0) + (moment1.reshape([-1, 1]) @ moment1.reshape([1, -1]))
            else:
                m1 = np.mean(features, axis=0)
                m2 = np.cov(features, rowvar=False, ddof=0) + (m1.reshape([-1, 1]) @ m1.reshape([1, -1]))
                moment1 = (moment1 * size + m1 * features.shape[0]) / (size + features.shape[0])
                moment2 = (moment2 * size + m2 * features.shape[0]) / (size + features.shape[0])
            return moment1, moment2
        
        moment1_s, moment2_s = get_moments(moment1_s, moment2_s, size, features_samples)
        moment1_t, moment2_t = get_moments(moment1_t, moment2_t, size, features_true)
        
        size = size + features_samples.shape[0]

        feats1.append(features_samples)
        feats2.append(features_true)

        feats1_small.append(features_samples[0].reshape(1,2048))
        feats2_small.append(features_true[0].reshape(1,2048))

    def get_mu_sigma(moment1, moment2):
        mu = moment1
        sigma = moment2 - moment1.reshape([-1, 1]) @ moment1.reshape([1, -1])
        sigma = sigma * size / (size - 1)
        return mu, sigma
    
    #mu_t, sigma_t =  get_mu_sigma(moment1_t, moment2_t)
    #mu_s, sigma_s =  get_mu_sigma(moment1_s, moment2_s)

    np_feats1 = np.concatenate(feats1)
    np_feats2 = np.concatenate(feats2)

    np_feats1_small = np.concatenate(feats1_small)
    np_feats2_small = np.concatenate(feats2_small)

    mu_s = np.mean(np_feats1, axis=0)
    sigma_s = np.cov(np_feats1, rowvar=False)

    mu_t = np.mean(np_feats2, axis=0)
    sigma_t = np.cov(np_feats2, rowvar=False)

    kid = kernel_distance(np_feats1, np_feats2)
    #fid = frechet_distance(mu_t, sigma_t, mu_s, sigma_s)
    kid_small = kernel_distance(np_feats1_small, np_feats2_small)
    results_file = common_dir + "/kid"  + ".txt"  #+ str(algo)

    with open(results_file, 'a') as f:
        f.write(f"\n Algorithm: {algo}\n ")
        f.write(f"KID: {kid}\n ")
        f.write(f"KID small: {kid_small}\n ")

