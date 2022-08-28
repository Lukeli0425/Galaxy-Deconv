# Algorithm Unrolling for PSF Deconvolution of Galaxy Surveys

To clone this project, run:
```zsh
git clone https://github.com/Lukeli0425/Galaxy-Deconvolution.git
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

## Environment

To create a virtual environment to run this project, run:
```zsh
pip install -r requirements.txt
```


