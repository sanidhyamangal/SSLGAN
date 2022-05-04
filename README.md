# SSL GANs
This repo contains all the code for running Self Supervised learning inducing gans and running instructions which will help TA and instructor to execute the code

### Helper modules
* _utils.py_: Contains all the helper functions
* _models.py_: Code for all the generator and discriinator models.
* _get\_split.py_: Script to split the dataset for the training from the big chunk of images.
* _dataloader.py_: Houses the dataloader pipeline to load anime dataset and preprocess it for the rotation net and contrastive learning
* _loss.py_: Implementation of contrastive learning loss.
* _train\_vanilla\_gan.py_: A script to train VanillaGAN
* _train\_rotnet\_gan.py_: A script to train RotnetGAN
* _train\_contrastive\_gan.py_: A script to train constrastive GAN
* _viz.py_: Script to generate the samples from trained models


## How to execute the code
### Splitting Dataset
First you need to modify the script by entering the path to dataset and the path where you wish to copy the images and then you can run the script using following command
```bash
python3 get_split.py
```
### VanillaGAN
To execute the training script you can simply run _train\_vanilla\_gan.py_. It will train GAN.
```bash
python3 train_vanilla_gan.py
```
### RotnetGAN
To execute the training script you can simply run _train\_rotnet\_gan.py_. It will train RotnetGAN.
```bash
python3 train_rotnet_gan.py
```
### ContrastiveGAN
To execute the training script you can simply run _train\_contrastive\_gan.py_. It will train ContrastiveGAN.
```bash
python3 train_contrastive_gan.py
```

### Sample Generation
To generate samples from the trained model you can run the script _viz.py_. It will generate 64 samples of the images. Ussage for the same is shown below
```bash
python3 viz.py --help
usage: Script to generate viz samples from the trained generator [-h] [--model_path MODEL_PATH] [--image_path IMAGE_PATH] [--show_image SHOW_IMAGE]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the trained model
  --image_path IMAGE_PATH
                        Path to save generated image
  --show_image SHOW_IMAGE
                        Flag to show generated image
```

## Visualization Loss Plots
All the visualizations during the training are produced in ```plots``` folder which contains all training samples across the epochs, loss curve, gaussian encoding and 3d regenration.

## References
Some part of the training were inspired from official tutorial for the DCGAN available at the pytorch documentation [page](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## Dataset
You can download the anime gan images from the [G-Drive](https://drive.google.com/file/d/0B4wZXrs0DHMHMEl1ODVpMjRTWEk/view?usp=sharing&resourcekey=0-cLy-WrY7ZuUWrhPIquiJkg).
