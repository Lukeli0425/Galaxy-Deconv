# SCORE

The Shape COnstraint REstoration algorithm ([SCORE](#SCORE "recursive call")) is a proximal algorithm based on sparsity and shape constraints to restore images. Its main purpose is to restore images while preserving their shape information. The chosen shape information here, is the ellipticity which is used to study galaxies for cosmological purposes. In practice, SCORE give an estimation <img src="https://render.githubusercontent.com/render/math?math=\hat{X}" width="15"> of the ground truth image, <img src="https://render.githubusercontent.com/render/math?math=X_T" width="25">, of the inverse problem :

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=Y = X_T\ast H %2B N\quad," height="18"></p>

with

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\hat{X} = \underset{X}{\text{argmin}} \left[\frac{1}{2\sigma^2}\|X\ast H - Y\|^2%2B\frac{\gamma}{2\sigma^2}M(X)%2B\iota_{%2B}(X)%2B \|\Lambda \odot \Phi X\|_1\right]\quad," height="37"></p>

where <img src="https://render.githubusercontent.com/render/math?math=Y, H \text{ and }N" height="15"> are respectively the observation, the convolution kernel and a white additive Gaussian of standard deviation <img src="https://render.githubusercontent.com/render/math?math=\sigma" height="8">, <img src="https://render.githubusercontent.com/render/math?math=M(\cdot)" height="20"> is the shape constraint operator, <img src="https://render.githubusercontent.com/render/math?math=\gamma" height="12"> is trade-off between the datafidelity and the shape constraint, <img src="https://render.githubusercontent.com/render/math?math=\iota_{%2B}(\cdot)" height="20"> is the positivity constraint operator (under the assumption that all the entries of <img src="https://render.githubusercontent.com/render/math?math=X_T" width="25"> are non-negative values) and finally, <img src="https://render.githubusercontent.com/render/math?math=\|\Lambda \odot \Phi \cdot\|_1" height="18"> is the sparsity constraint (assuming that <img src="https://render.githubusercontent.com/render/math?math=\Phi X_T" height="18"> is sparse).

- [Getting Started](#Getting-Started)
  * [Prerequisites](###Prerequisites)
  * [Installing](###Installing)
- [Running the examples](##Running-the-examples)
  * [Example 1](###Example-1)
  * [Example 2](###Example-2)
- [Parameters](##Parameters)
- [Reproducible Research](##Reproducible-Research)
- [Authors](##Authors)
- [License](##License)
- [Acknowledgments](##Acknowledgments)

## Getting Started


### Prerequisites


These instructions will get you a copy of the project up and running on your local machine. One easy way to install the prerequisites is using Anaconda. To install Anaconda see : https://docs.conda.io/projects/conda/en/latest/user-guide/install/

* Numpy

```sh
conda install -c anaconda numpy
```
* Scipy

```sh
conda install -c anaconda scipy
```

* Skimage

```sh
conda install -c conda-forge scikit-image
```

* α-shearlet Transform

&nbsp;&nbsp;&nbsp;&nbsp;Clone the library (https://github.com/dedale-fet/alpha-transform) and run the commands below to get the approriate version  : 

```sh
git clone https://github.com/dedale-fet/alpha-transform.git
cd alpha-transform/
git checkout adcf993
```


&nbsp;&nbsp;&nbsp;&nbsp;Then add the path of the α-shearlet Transform library to the PYTHONPATH variable in the bash profile

```sh
export PYTHONPATH="$HOME/path/to/alpha-transform-master:$PYTHONPATH"
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Replace `path/to` by the corresponding path_

* GalSim [optional] (for research reproducibility)

```sh
conda install -c conda-forge galsim 
```

* Matplotlib [optional]

```sh
conda install -c conda-forge matplotlib
```

### Installing

After install the prerequisites, clone or download `score` repository. And to be able to access from any working directory, use the following command to add the path to `score` to PYTHONPATH variable in the bash profile :

```sh
export PYTHONPATH="$HOME/path/to/score:$PYTHONPATH"
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Replace `path/to` by the corresponding path_

## Running the examples

This repository contains two examples. They respectively illustrate a denoising and a deconvolution case.

### Example 1

In this simple denoising case, we restore a galaxy image corrupted by noise. The core of the code is:

```python
#instantiate score and, for example, set the value of gamma the other parameters will take their default values
denoiser = score(gamma=0.5)
#denoise
denoiser.denoise(obs=gal_obs) #the result will be in denoiser.solution
```

It is also possible to change [other parameters](##Parameters) in `score`.


### Example 2

In this deconvolution case, we compare the score algorithm with a value of γ = 1 (which is close to its optimal computed value) and the Sparse Restoration Algorithm (γ = 0 and no Removal of Isolated Pixels). We loop on a stack of galaxy images and perfom both deconvolution operation on each image:

```python
#loop
for obs, psf, gt in zip(gals_obs,psfs,gals):
    #deconvolve
    g1.deconvolve(obs=obs,ground_truth=gt,psf=psf)
    g0.deconvolve(obs=obs,ground_truth=gt,psf=psf)
    #update ellipticity error lists
    g1_error_list += [g1.relative_ell_error]
    g0_error_list += [g0.relative_ell_error]
```
## Parameters

The following is an exhaustive list of parameters of `score` :


| Parameter     | Type                                 | Information                                                        |
| ------------- |--------------------------------------| -------------------------------------------------------------------|
| `obs`         | 2D Numpy Array                       | observation (required)                                             |
| `psf`         | 2D Numpy Array                       | convolution kernel (required for deconvolution)                    |
| `ground_truth`| 2D Numpy Array (none by default)     | ground_truth image (optional)                                      |
| `sigma`       | positive scalar                      | noise standard deviation (optional)                                |
| `beta_factor` | positive scalar < 1 (0.95 by default)| multiplicative factor to ensure that beta is not too big (optional)|
| `epsilon`     | positive scalar                      | error upperbound for the Lipschitz constant estimation (optional)  |
| `n_maps`      | positive integer                     | threshold estimation parameter for hard thresholding (optional)    |
| `n_shearlet`  | positive integer (3 by default)      | number of scales for the shearlet transform (optional)             |
| `n_starlet`   | positive integer (4 by default)      | number of scales for the starlet transform (optional)              |
| `starlet_gen` | positive integer (either 1 or 2)     | starlet generation (optional)                                      |
| `beta`        | positive scalar                      | gradient step size (optional)                                      |
| `k`           | positive integer (3 by default)      | threshold parameter for hard thresholding (optional)               |
| `rip`         | boolean (true by default)            | activate Removal of Isolated Pixels after restoration (optional)   |
| `gamma`       | non-negative scalar (1 by default)   | trade-off between data-fidelity and shape constraint (optional)    |
| `n_itr`       | positive integer                     | maximum number of iterations (optional)                            |
| `tolerance`   | positive scalar < 1 (1e-6 by default)| threshold of the convergence test for deconvolution (optional)     |
| `verbose`     | boolean (true by default)            | to activate verbose (optional)                                     |

## Reproducible Research

The code `generate_dataset.py` allows to recreate the exactly the same dataset used for the numerical experiments of the [original paper of `score`](https://arxiv.org/abs/2101.10021). To be able to run this code, the catalog `COSMOS_25.2_training_sample` is required. It is available as a compressed file, `COSMOS_25.2_training_sample.tar.gz`, on [this link](https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data).

## Authors 

* [**Fadi Nammour**](http://www.cosmostat.org/people/fadi-nammour)
* [**Morgan Schmitz**](http://www.cosmostat.org/people/mschmitz)
* [**Fred Maurice Ngolè Mboula**](https://www.cosmostat.org/people/fred-ngole-mboula)
* [**Jean-Luc Starck**](https://www.cosmostat.org/people/jeanluc-starck)
* [**Julien Girard**](https://www.cosmostat.org/people/julien-girard)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments 

We would like to thank the following researchers :

* [**Samuel Farrens**](http://www.cosmostat.org/people/sfarrens) for giving precious tips in programming.
* [**Axel Guinot**](http://www.cosmostat.org/people/axel-guinot) for helping generating the dataset.
* [**Ming Jiang**](http://www.cosmostat.org/people/ming-jiang) for providing the starlet transform code.
* [**Jérôme Bobin**](http://www.cosmostat.org/people/jerome-bobin) for giving us tips in optimisation.
