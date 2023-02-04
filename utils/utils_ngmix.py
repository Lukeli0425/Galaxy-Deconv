import numpy as np
import ngmix

def make_data(obs_im, psf_im, pixel_scale=0.2):
    
    cen = (np.array(obs_im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0], col=cen[1], scale=pixel_scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=pixel_scale,
    )
        
    psf_obs = ngmix.Observation(
        image=psf_im,
        jacobian=psf_jacobian,
    )
    obs = ngmix.Observation(
        image=obs_im,
        jacobian=jacobian,
        psf=psf_obs,
    )

    return obs

def get_prior(*, rng, scale, T_range=None, F_range=None, nband=None):
    """
    get a prior for use with the maximum likelihood fitter

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    scale: float
        Pixel scale
    T_range: (float, float), optional
        The range for the prior on T
    F_range: (float, float), optional
        Fhe range for the prior on flux
    nband: int, optional
        number of bands
    """
    if T_range is None:
        T_range = [-1.0, 1.e3]
    if F_range is None:
        F_range = [-100.0, 1.e9]

    g_prior = ngmix.priors.GPriorBA(sigma=0.1, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(minval=T_range[0], maxval=T_range[1], rng=rng)
    fracdev_prior = ngmix.priors.Normal(mean=0.5, sigma=0.1, rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=F_range[0], maxval=F_range[1], rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    prior = ngmix.joint_prior.PriorBDFSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        fracdev_prior=fracdev_prior,
        F_prior=F_prior,
    )

    return prior


def get_ngmix_Bootstrapper(psf_ngauss=1, pixel_scale=0.2, ntry=2, seed=9131):
    """Get a ngmix Bootstrapper that fits PSF and galaxy.

    Args:
        psf_ngauss (int, optional): Number of Gaussians in PSF. Defaults to 1.
        pixel_scale (float, optional): Pixel scale. Defaults to 0.2.
        ntry (int, optional): Number of tries. Defaults to 2.
        seed (int, optional): Seed for the random number generator. Defaults to 9131.

    Returns:
        ngmix.bootstrap.Bootstrapper: Wrapper that bootstraps fits to psf and object.
    """
    rng = np.random.RandomState(seed)
    
    # fit the object to an exponential disk
    prior = get_prior(rng=rng, scale=pixel_scale)
    # fit bulge+disk with fixed size ratio, using the levenberg marquards algorithm
    fitter = ngmix.fitting.Fitter(model='bdf', prior=prior)
    # make parameter guesses based on a psf flux and a rough T
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )
    # psf fitting with coelliptical Gaussians
    psf_fitter = ngmix.em.EMFitter()
    # guesses full gmix objects
    psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)

    # this runs the fitter. We set ntry>1 to retry the fit if it fails.
    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter, guesser=psf_guesser,
        ntry=ntry,
    )
    runner = ngmix.runners.Runner(
        fitter=fitter, guesser=guesser,
        ntry=ntry,
    )

    # this bootstraps the process, first fitting psfs then the object
    boot = ngmix.bootstrap.Bootstrapper(
        runner=runner,
        psf_runner=psf_runner,
    )
    
    return boot