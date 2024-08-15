# VIPaint
Image Inpainting using (latent) Diffusion Models as Priors using Variational Inference

![Plot1](https://github.com/SakshiAgarwal/VIPaint/blob/main/results/working_of_VIPaint.png)


# Requirements
1) Clone the repository

`git clone https://github.com/SakshiAgarwal/VIPaint.git`

`cd VIPaint`

2) A suitable conda environment named ldm can be created and activated with:

`conda env create -f environment.yaml`
`conda activate ldm`

You can also update an existing latent diffusion environment by running

`pip install transformers==4.19.2 pytorch-lightning==1.6.5 numpy==1.23.5 pillow==9.5.0 torchmetrics==0.6.0 diffusers invisible-watermark kornia==0.6.4`

`pip install -e`

Since torch._six is deprecated with the versions we use, replace `from torch._six import string_classes` with `string_classes = str` in `src/taming-transformers/taming/data/utils.py`

# Get the models

Running the following script downloads und extracts all available pretrained autoencoding models.

`bash scripts/download_first_stages.sh`

The first stage models can then be found in 

`models/first_stage_models/<model_spec>`

For instance, the model_spec of pre-trained autoencoder used by the LSUN Churches dataset is kl-f8. 

The LDMs can jointly be downloaded and extracted via

`bash scripts/download_models.sh`

The models can then be found in `models/ldm/<model_spec>`. For LSUN churches, this is models/ldm/lsun_churches256. For Imagenet, please refer to [Class-Conditional Synthesis with Latent Diffusion Models](https://colab.research.google.com/github/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb#scrollTo=NHgUAp48qwoG) ipynb file. 

Please follow [Latent Diffusion Model](https://github.com/CompVis/latent-diffusion) for more information. 

# Inference : Image Inpainting

By using a variational inference, the diffusion model can be used for different tasks such as image inpainting. We provide a script to perform image inpainting with Stable Diffusion.

The following describes how to run VIPaint for ImageNet (for LSUN use configs/inpainting/lsun_config.yaml) where a masked image is filled with consistent, multiple inpaintings. Give it a moment to fit the posterior!

`python vipaint.py --inpaint_config="configs/inpainting/imagenet_config.yaml" --working_directory='results/imagenet/random_all/'`

# Evaluation 

To evaluate the inpaintings, we calculate the KID, LPIPs score and you can run the following:

`python evaluation.py --inpaint_config="configs/inpainting/imagenet_config.yaml" --working_directory='results/imagenet/random_all/'`

# Consistency

![Plot1](https://github.com/SakshiAgarwal/VIPaint/blob/main/results/consistency_imagenet_256.png)

![Plot1](https://github.com/SakshiAgarwal/VIPaint/blob/main/results/consistency_lsun.png)

# Variety
When you input different label to the masked image, VIPaint can produce realistic and consistent images. 

![Plot1](https://github.com/SakshiAgarwal/VIPaint/blob/main/results/variety_labels.png)



