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

## Visualization Loss Plots
All the visualizations during the training are produced in ```plots``` folder which contains all training samples across the epochs, loss curve, gaussian encoding and 3d regenration.

## References
Some part of the training were inspired from official tutorial for the DCGAN available at the pytorch documentation [page](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
