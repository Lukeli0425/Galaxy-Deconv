To reproduce the numerical experiments in the paper, use the following python
codes:

1. `generate_data.py`
  * set the corresponding values for `path_catalog` and `data_path`
  * run the python code

After successfully running `generate_data.py`, in the folder corresponding to
`data_path`, the following items should be present:

* `galaxies.npy`, (300,96,96) Numpy Array, 300 ground truth galaxy images
* `ellipticities.npy`, (300,2) Numpy Array, ground truth galaxy ellipticities
* `PSFs.npy`, (300,96,96) Numpy Array, 300 PSF images
* `convolved_galaxies.npy`, (300,96,96) Numpy Array, 300 convolved galaxy images
* `convolved_ellipticities.npy`, (300,2) Numpy Array, convolved galaxy
ellipticities
* 4 `SNR{X}` folders with {X} = 40,75,150 and 380. In them:

  -  `observed_galaxies_SNR{X}.npy`, (300,96,96) Numpy Array, 300 observed
  galaxy images with a SNR level of {X}.

2. `denoising.py`, `deconvolution_gamma_star.py` and
`deconvolution_gamma_zero.py`
  * set the corresponding values for `root_path`, `data_path` and `results_path`
  for each of the scripts
  * run the python code

After successfully running the three scripts, in the folder corresponding to
`results_path`, there should be two folders `150_conv_k4` and `40,itr_k4`, each
containing 4 folders ,`SNR{X}` with {X} = 40,75,150 and 380, within each one of
them the following items :

* `{mode}_galaxies_gamma_{value}.npy` (with {mode} = `denoising` or
`deconvolution` and {value} = `zero` and`star`), (300,96,96) Numpy Array, 300
restored galaxy images.
* `ellipticities_gamma_{value}.npy`(with {value} = `zero` and`star`), (300,2)
 Numpy Array, 300 restored galaxy ellipticities.
