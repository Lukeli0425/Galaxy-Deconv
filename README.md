# Galaxy Image Deconvolution for Weak Gravitational Lensing with Physics-informed Deep Learning

This repository holds code for Galaxy Image Deconvolution for Weak Gravitational Lensing with Physics-informed Deep Learning [pdf link tp come].

To clone this project, run:
```zsh
git clone https://github.com/Lukeli0425/Galaxy-Deconvolution.git
```

## Environment

To create a virtual environment to run this project, run:
```zsh
pip install -r requirements.txt
```

## Download [COSMOS Real Galaxy Dataset](https://zenodo.org/record/3242143#.Ysmezi-KFQJ)

Create a `data` folder under the root directory:
```zsh
mkdir data
```

Go under `data` directory and download COSMOS data:
```zsh
cd data
wget https://zenodo.org/record/3242143/files/COSMOS_23.5_training_sample.tar.gz
wget https://zenodo.org/record/3242143/files/COSMOS_25.2_training_sample.tar.gz
```

Unzip the downloaded files:
```zsh
tar zxvf COSMOS_23.5_training_sample.tar.gz
tar zxvf COSMOS_25.2_training_sample.tar.gz
```

## Using the model on your own data

We provide a tutorial for using the suggested model on your data, see ['tutorial/deconv.ipynb'](tutorial/deconv.ipynb) for details. 

## Simulating your own dataset

We simulated our dataset with the modular galaxy image simulation toolkit [Galsim](https://github.com/GalSim-developers/GalSim) and the [COSMOS Real Galaxy Dataset](https://zenodo.org/record/3242143#.Ytjzki-KFAY). To create your own dataset, one need to firstd download the COSMOS data.

Create a `data` folder under the root directory:
```zsh
mkdir data
```

Go under `data` directory and download COSMOS data:
```zsh
cd data
wget https://zenodo.org/record/3242143/files/COSMOS_23.5_training_sample.tar.gz
wget https://zenodo.org/record/3242143/files/COSMOS_25.2_training_sample.tar.gz
```

Unzip the downloaded files:
```zsh
tar zxvf COSMOS_23.5_training_sample.tar.gz
tar zxvf COSMOS_25.2_training_sample.tar.gz
```

## Retraining on simulated data

If you want to train the model with your own dataset, you can either train it from scratch or use our saved model ([`saved_models`](saved_models)) and train with a transfer learning manner.

## Recreating our figures

The [`figures`](figures) folder holds the figures in the paper and the files that created them (see ['figures/README.md'](figures/README.md)).



-- all the figure code is in a directory, in that directory is a figure readme that names the notebook for (fig1: grid.ipynb, etc)
-- how to rerun on your own data/simualtions


