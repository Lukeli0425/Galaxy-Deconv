#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:16:22 2019

@author: fnammour
"""

#libraries
import numpy as np
import cadmos_lib as cl
import starlets

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#error message
np2d_error_message = "'{}' must be a 2D Numpy Array."
posnum_error_message = "'{}' must be a non negative number."
posint_error_message = "'{}' must be a non negative integer."

class score:
    """This class restores blurred images using the Forward Backward Splitting
    algorithm along with positivity, sparse and shape constraints. The user can
    perfom a full restoration or only denoising of the observation."""
    # class variable
    object_type = 'score'
    
    # class initialiser
    def __init__(self,**kwargs):
        
        #to check if the value of beta is given by the user
        self._bool_beta = False
        
        #flag for the first initialisation (init_const)
        self._bool_init = False
        
        #flag for the denoising initialisation (init_case)
        self._bool_init_den = False
        
        #flag for the deconvolution initialisation (init_case)
        self._bool_init_dec = False
        
        #flag for the additional halting criterion in deconvolution
        self._bool_halt = False
        
        #Point Spread Function
        self.psf = None
        
        #noise map for deconvolution
        self.std_map_dec = None
        
        #noise map for denoising
        self.std_map_den = None
        
        #deconvolution solution
        self.solution = None
        
        #loss
        self.loss = None
        
        #number of rows
        self.n_row = None
        
        #number of columns
        self.n_col = None
        
        #alpha transform of the solution
        self.alpha = None
        
        #residual
        self.residual = None
        
        #centroid estimate of the observation
        self.obs_centroid = None
        
        #thresholds for the hard thresholding
        self.thresholds = None
        
        #current iteration number
        self.itr = None
        
        #flag to indicate if the current operation is deconvolution or denosing
        self._bool_dec = None
        
        #gamma, trade-off parameter between data fidelity and shape constraint
        self.gamma = None
        
        #Beta factor, multiplicative factor to guarantee that beta did not
        #exceed its upperbound
        self.beta_factor = None
        
        #Error bound epsilon for the Lipschitz constant
        self.epsilon_lip = None
        
        #number of noise maps
        self.n_maps = None
        
        #starlet_gen to select the starlet generation to be used
        self.starlet_gen = None
        
        #gradient step, beta
        self.beta = None
        
        #k for k-sigma_{MAD}
        self.k = None
        
        #boolean to activate the Removal of Isolated Pixels (rip)
        self.rip = None
        
        #trade-off gamma between the data-fidelity and the shape constraint
        self.gamma = None
        
        #number of iterations
        self.n_itr = None
        
        #Tolerance for convergence criteria in deconvolution
        self.tolerance = None
        
        #boolean to activate verbose
        self.verbose = None
        
        #user given values
        self.set_param(**kwargs)
        
        #relative Mean Square Error of the solution
        self.relative_mse = None
        
        #grounf truth ellipticity estimation
        self.ell_ground_truth = None
        
        #solution ellipticity estimation
        self.ell_solution = None
        
        #relative ellipticity error
        self.relative_ell_error = None
        
        #number of shearlet scales
        self.n_shearlet = None
        
        #number of starlet scales
        self.n_starlet = None
    
    def set_param(self,**kwargs):
        """This methods allows the user to set the values of the optional
        parameters.
        INPUT: None
        OUTPUT: None
        -----------------------------------------------------------------------
        PARAMETERS: 
            beta_factor, positive number
            epsilon, positive number
            n_maps, positive integer
            n_shearlet, positive integer
            n_starlet, positive integer
            starlet_gen, 1 or 2
            beta, postive number
            k, positive integer
            rip, boolean
            gamma, non-negative number
            n_itr, positive integer
            tolerance, positive number
            verbose, boolean"""
        #beta factor, multiplicative factor to guarantee that beta did not
        #exceed its upperbound
        if 'beta_factor' in kwargs:
            if is_number(kwargs['beta_factor']):
                if 1>=kwargs['beta_factor']>=0:
                    self.beta_factor = kwargs['beta_factor']
                else:
                    raise ValueError("'beta_factor' must be between 0 and 1.")
            else:
                raise TypeError("'beta_factor' must be between 0 and 1.")
        
        #error bound epsilon for the Lipschitz constant
        if 'epsilon' in kwargs:
            if is_number(kwargs['epsilon']):
                if kwargs['epsilon']>=0:
                    self.epsilon_lip = kwargs['epsilon']
                else:
                    raise ValueError(posnum_error_message.format('epsilon'))
            else:
                raise TypeError(posnum_error_message.format('epsilon'))
        
        #number of noise maps
        if 'n_maps' in kwargs:
            if is_number(kwargs['n_maps']):
                if abs(int(kwargs['n_maps']))==kwargs['n_maps']:
                    self.n_maps = int(kwargs['n_maps'])
                else:
                    raise ValueError(posint_error_message.format('n_maps'))
            else:
                raise TypeError(posint_error_message.format('n_maps'))
        
        #number of shearlet scales
        if 'n_shearlet' in kwargs:
            if is_number(kwargs['n_shearlet']):
                if abs(int(kwargs['n_shearlet']))==kwargs['n_shearlet']:
                    self.n_shearlet = int(kwargs['n_shearlet'])
                else:
                    raise ValueError(posint_error_message.format('n_shearlet'))
            else:
                raise TypeError(posint_error_message.format('n_shearlet'))
            
        #number of starlet scales
        if 'n_starlet' in kwargs:
            if is_number(kwargs['n_starlet']):
                if abs(int(kwargs['n_starlet']))==kwargs['n_starlet']:
                    self.n_starlet = int(kwargs['n_starlet'])
                else:
                    raise ValueError(posint_error_message.format('n_starlet'))
            else:
                raise TypeError(posint_error_message.format('n_starlet'))
        
        #starlet_gen to select the starlet generation to be used
        if 'starlet_gen' in kwargs:
            if is_number(kwargs['starlet_gen']):
                if kwargs['starlet_gen'] in [1,2]:
                    self.starlet_gen = int(kwargs['starlet_gen'])
                else:
                    raise ValueError("'starlet_gen' should be either 1 or 2")
            else:
                raise TypeError("'starlet_gen' should be either 1 or 2")
        
        #gradient step, beta
        if 'beta' in kwargs:
            if is_number(kwargs['beta']):
                if kwargs['beta']>=0:
                    self.beta = kwargs['beta']
                    self._bool_beta = True
                else:
                    raise ValueError(posnum_error_message.format('beta'))
            else:
                raise TypeError(posnum_error_message.format('beta'))
        
        #k for k-sigma_{MAD}
        if 'k' in kwargs:
            if is_number(kwargs['k']):
                if abs(int(kwargs['k']))==kwargs['k']:
                    self.k = int(kwargs['k'])
                else:
                    raise ValueError(posint_error_message.format('k'))
            else:
                raise TypeError(posint_error_message.format('k'))
        
        #boolean to activate the Removal of Isolated Pixels (rip)
        if 'rip' in kwargs:
            if isinstance(kwargs['rip'],bool):
                self.rip = kwargs['rip']
            else:
                raise TypeError("'rip' must be a boolean.")
        
        #trade-off gamma between the data-fidelity and the shape constraint
        if 'gamma' in kwargs:
            if is_number(kwargs['gamma']):
                if kwargs['gamma']>=0:
                    if self.gamma != kwargs['gamma']:
                        self._bool_init_den = False
                        self._bool_init_dec = False
                    self.gamma = kwargs['gamma']
                else:
                    raise ValueError(posnum_error_message.format('gamma'))
            else:
                raise TypeError(posnum_error_message.format('gamma'))
        
        #number of iterations
        if 'n_itr' in kwargs:
            if is_number(kwargs['n_itr']):
                if abs(int(kwargs['n_itr']))==kwargs['n_itr']:
                    self.n_itr = int(kwargs['n_itr'])
                else:
                    raise ValueError(posint_error_message.format('n_itr'))
            else:
                raise TypeError(posint_error_message.format('n_itr'))
        
        #Tolerance for convergence criteria in deconvolution
        if 'tolerance' in kwargs:
            if is_number(kwargs['tolerance']):
                if kwargs['tolerance']>=0:
                    self.tolerance = kwargs['tolerance']
                else:
                    raise ValueError(posnum_error_message.format('tolerance'))
            else:
                raise TypeError(posnum_error_message.format('tolerance'))
            
        #boolean to activate verbose
        if 'verbose' in kwargs:
            if isinstance(kwargs['verbose'],bool):
                self.verbose = kwargs['verbose']
            else:
                raise TypeError("'verbose' must be a boolean.")
                
        #thresholds for hard thresholding
        if 'thresholds' in kwargs:
            self.thresholds = kwargs['thresholds']       
            
    def set_defaults(self,**kwargs):
        """This methods set the parameters which value has not been given by
        the user, to their default value. For more details see 'set_param'."""
        
        if self.beta_factor == None:
            self.beta_factor = 0.95
        
        if self.epsilon_lip == None:
            self.epsilon_lip = 1e-3
            
        if self.n_maps == None:
            self.n_maps = 100
        
        if self.n_shearlet == None:
            self.n_shearlet = 3
        
        if self.n_starlet == None:
            self.n_starlet = 4
        
        if self.starlet_gen == None:
            self.starlet_gen = 2
        
        if self.k == None:
            self.k = 4
        
        if self.rip == None:
            self.rip = False
        
        if self.gamma == None:
            self.gamma = 1.0
        
        if self.n_itr == None:
            if self._bool_dec:
                self.n_itr = 150
            else:
                self.n_itr = 40
        
        if self.tolerance == None:
            self.tolerance = 1e-6
        
        if self.verbose == None:
            self.verbose = True
        
        self.relative_mse = None
        
        self.ell_solution = None
        
        self.relative_ell_error = None
    
    def estimate_sigma(self):
        """This method estimates the standard noise deviation in the 
        observation. It is performed using a sigma on a masked image of the
        observation. The mask is used to remove the galaxy. The former is a
        binary mask centered on the galaxy and assuming that it is contained
        in a surface to the $16^\text{th}$ of the total surface of the image.
        INPUT: None
        OUTPUT: None"""
        #generate a binary to mask the galaxy
        mask = cl.compute_background_mask(self.obs,center=self.obs_centroid)
        noise = self.obs[mask]
        self.sigma = cl.sigma_mad(noise)
        
    def grad_op(self):
        """This method performs a forward step of the SCORE algorithm.
        INPUT: None
        OUTPUT: None"""
        temp = self.residual
        if self._bool_dec:
            temp = cl.convolve(self.residual,self.psf,'same')
        temp = np.array(cl.comp_grad(temp,self.psu,self.mu,self.gamma))
        if self._bool_dec:
            temp =  cl.convolve(temp,self.psf_rot,'same')
        self.solution -= self.beta*temp
        self.update_alpha()
    
    def prox_op(self):
        """This method performs a backward step on the solution.
        INPUT: None
        OUTPUT: None"""
        
        #sparsity constraint
        #multiscale threshold except coarse scale
        self.alpha[:-1] = cl.MS_hard_thresh(self.alpha[:-1],\
                          self.beta*np.array(self.thresholds[:-1]))
        
        self.update_solution()
        
        #positivity constraint
        self.solution = self.solution*(self.solution>0)
        self.update_residual()
        self.update_alpha()
    
    def update_alpha(self):
        """This method updates alpha, the starlet transformation of the 
        solution.
        INPUT: None
        OUTPUT: None"""
        self.alpha = self.starlet_op(self.solution)
    
    def update_solution(self):
        """This method updates the solution using its starlet transform.
        INPUT: None
        OUTPUT: None"""
        self.solution = self.inv_starlet_op(self.alpha)
    
    def update_residual(self):
        """This method updates the residual using its corresponding solution.
        INPUT: None
        OUTPUT: None"""
        if self._bool_dec:
            self.residual = cl.convolve(self.solution,self.psf,'same')-self.obs
        else:
            self.residual = self.solution-self.obs
    
    def update_loss(self):
        """This method updates the loss of SCORE.
        INPUT: None
        OUTPUT: None"""
        data_fid = np.linalg.norm(self.residual)**2/2.
        sparsity = cl.norm1(self.alpha)
        shape_constraint = self.gamma*(np.array(\
                [[mu_ij*((self.residual*psu_ij).sum())**2*psu_ij\
                  for mu_ij,psu_ij in zip(mu_j, psu_j)]\
                for mu_j,psu_j in zip(self.mu,self.psu)])/2.).sum()
        self.loss += [data_fid+sparsity+shape_constraint]
    
    def update_halt(self):
        """This method updates the halt criterion of SCORE in the deconvolution
        case.
        INPUT: None
        OUTPUT: None"""
        if self._bool_dec:
            #test halting criterion
            if self.itr >=3:
                t1 = self.loss[-4]+self.loss[-3]
                t2 = self.loss[-2]+self.loss[-1]
                loss_diff = np.abs(t1-t2)/t1
                
                if loss_diff <= self.tolerance:
                    self._bool_halt = True
    
    def estimate_std_map(self):
        """This method estimates the standard deviation map of propagated 
        normalised noise in the starlet space.
        INPUT: None
        OUTPUT: std_map, 3D Numpy Array"""
        def noise_op(res):
            """This function backprojects the noise to the image space.
            INPUT: res, 2D Numpy Array
            OUTPUT: bp_res, 2D Numpy Array"""
            bp_res = np.array(cl.comp_grad(res,self.psu,self.mu,self.gamma))
            if self._bool_dec:
                bp_res = cl.convolve(bp_res,self.psf_rot,'same')
            return bp_res
        noise = np.random.randn(self.n_maps,self.n_row,self.n_col)
        #noise backprojection
        bp_noise = np.array([noise_op(n) for n in noise])
        #Starlet transforms of noise
        starlet_noise = np.array([self.starlet_op(bn) for bn in bp_noise])
        #estimate the noise standard deviation condering every noise
        #realisation for every pixel in every scale
        std_map = np.array([[[np.std(y) for y in pos] for pos in scale] \
                                 for scale in np.moveaxis(starlet_noise,0,-1)])        
        return std_map
    
    def estimate_centroid(self):
        """This method is a method that estimates the centroid of the galaxy.
        INPUT: None
        OUTPUT: None"""
        flux = self.obs.sum()
        i_c = int(np.round(cl.scal(self.obs, self.U[0])/flux))
        j_c = int(np.round(cl.scal(self.obs, self.U[1])/flux))
        self.obs_centroid = np.array([i_c,j_c])
        
    def set_thresholds(self):
        """This method sets the thresholds for the hard thresholding part in 
        the backward step of the SCORE algorithm.
        INPUT: None
        OUTPUT: None"""
        if self._bool_dec:
            std_map = self.std_map_dec
        else:
            std_map = self.std_map_den
        sigma_map = self.sigma*std_map
        self.thresholds = np.vstack(([(self.k+1)*s for s in sigma_map[:1]],\
                                     [self.k*s for s in sigma_map[1:]]))
    
    def init_const(self):
        """This method initialises the constants and operators that are 
        common for both denoising and deconvolution cases.
        INPUT: None
        OUTPUT: None"""
        self.U = cl.makeUi(self.n_row,self.n_col)
        self.shearlets,self.shearlets_adj = cl.get_shearlets(self.n_row
                                                             ,self.n_col
                                                             ,self.n_shearlet)
        #Adjoint shealret transform of U, i.e Psi^{Star}(U)
        self.psu = np.array([cl.convolve_stack(ui,self.shearlets_adj) for ui in
                             self.U])
        self.mu = cl.comp_mu(self.psu)
        
        #boolean for starlet generation (if true, second generation;else first)
        bool_gen = bool(self.starlet_gen-1)
        
        def starlet_op(signal):
            n_scale = self.n_starlet
            return starlets.star2d(signal,scale=n_scale,gen2=bool_gen)
        
        self.starlet_op = starlet_op
        
        def inv_starlet_op(signal):
            return starlets.istar2d(signal,gen2=bool_gen)
        
        self.inv_starlet_op = inv_starlet_op

    def init_case(self):
        """This method initialises the constants and the operators that are
        specific to the deconvolution case if self._bool_dec is true, the 
        denoising case otherwise.
        INPUT: None
        OUTPUT: None"""
        if self._bool_dec:
            self.psf_rot = cl.rotate180(self.psf)
        else:
            self._bool_init_den = True
        
        if not(self._bool_beta):
            def grad_op_lip(residual):
                """This function computes the gradient of the differentiable 
                part of the loss function of SCORE.
                INPUT: residual, 2D Numpy Array
                OUTPUT: temp, 2D Numpy Array"""
                temp = residual
                if self._bool_dec:
                    temp = cl.convolve(residual,self.psf,'same')
                temp = np.array(cl.comp_grad(temp,self.psu,self.mu,self.gamma))
                if self._bool_dec:
                    temp =  cl.convolve(temp,self.psf_rot,'same')
                return temp
                
            lip_cst,_ = cl.power_iteration(grad_op_lip, (self.n_row,self.n_col)\
                                           ,self.epsilon_lip)
            self.beta = self.beta_factor/lip_cst
            
            if self._bool_dec:
                self.beta_dec = np.copy(self.beta)
            else:
                self.beta_den = np.copy(self.beta)
                
        std_map = self.estimate_std_map()
        if self._bool_dec:
            self.std_map_dec = np.copy(std_map)
        else:
            self.std_map_den = np.copy(std_map)
            
    def init_input(self, **kwargs):
        """This method restores the observed image. If self._bool_dec is true,
        a deconvolution, it performs a deconvolution. Else it performs a 
        denosing.
        INPUT: obs, 2D Numpy Array
               psf (if self._bool_dec==True), 2D Numpy Array
               ground_truth (optional), 2D Numpy Array
               first_guess (optional), 2D Numpy Array
        OUTPUT: restored, 2D Numpy Array
        -----------------------------------------------------------------------
        PARAMETERS:
           For more details on the input parameters see 'set_param'. """
        #fetch required inputs
        
        #start by assuming that the value of beta is not given by the user
        self._bool_beta = False
        
        #observed image
        if 'obs' in kwargs:
            if type(kwargs['obs']) is np.ndarray:
                if np.ndim(kwargs['obs'])==2:
                    self.obs = kwargs['obs']
                else:
                    raise ValueError(np2d_error_message.format('obs'))
            else:
                raise TypeError(np2d_error_message.format('obs'))
        else:
            raise KeyError('SCORE cannot do restoration without obs')
        
        #ground Truth image
        if 'ground_truth' in kwargs:
            if type(kwargs['ground_truth']) is np.ndarray:
                if np.ndim(kwargs['ground_truth'])==2:
                    self.ground_truth = kwargs['ground_truth']
                else:
                    raise ValueError(np2d_error_message.format('ground_truth'))
            else:
                raise TypeError(np2d_error_message.format('ground_truth'))
        else:
            self.ground_truth = None
        
        #first guess image
        if 'first_guess' in kwargs:
            if type(kwargs['first_guess']) is np.ndarray:
                if np.ndim(kwargs['first_guess'])==2:
                    self.first_guess = kwargs['first_guess']
                else:
                    raise ValueError(np2d_error_message.format('first_guess'))
            else:
                raise TypeError(np2d_error_message.format('first_guess'))
        else:
            self.first_guess = np.ones(self.obs.shape)/self.obs.size
        
        #Error bound epsilon for the Lipschitz constant
        if 'sigma' in kwargs:
            if is_number(kwargs['sigma']):
                if kwargs['sigma']>=0:
                    self.sigma = kwargs['sigma']
                else:
                    raise ValueError(posnum_error_message.format('sigma'))
            else:
                raise TypeError(posnum_error_message.format('sigma'))
        else:
            self.sigma = None
        
        if self._bool_dec:
            #Point Spread Fuction
            if 'psf' in kwargs:
                if type(kwargs['psf']) is np.ndarray:
                    if np.ndim(kwargs['psf'])==2:
                        if (self.psf != kwargs['psf']).any():
                            self._bool_init_dec = False
                        self.psf = kwargs['psf']
                    else:
                        raise ValueError(np2d_error_message.format('psf'))
                else:
                    raise TypeError(np2d_error_message.format('psf'))
            else:
                raise KeyError('SCORE cannot do deconvolution without psf')
        
        #Check if the current observed image and the previous one have the same
        #dimensions
        if not(np.all((self.n_row, self.n_col) == self.obs.shape)):
            self._bool_init = False
            if self._bool_dec:
                self._bool_init_dec = False
            else:
                self._bool_init_den = False
                
        self.n_row, self.n_col = self.obs.shape
        
        if not self._bool_init:
            self.init_const()
            self._bool_init = True
        
        if self._bool_dec:
            if not self._bool_init_dec:
                self.init_case()
                self._bool_init_dec = True
            if not(self._bool_beta):
                self.beta = np.copy(self.beta_dec)
        else:
            if not self._bool_init_den:
                self.init_case()
                self._bool_init_den = True
            if not(self._bool_beta):
                self.beta = np.copy(self.beta_den)
        
        if self.sigma == None:
            self.estimate_centroid()
            self.estimate_sigma()
            
        if np.all(self.thresholds == None):
            self.set_thresholds()
    
    def forward_backward(self):
        """This method applies the forward backward algorithm.
        INPUT: None
        OUTPUT: None"""
        #initialise solution
        self.solution = np.copy(self.first_guess)
        self.update_residual()
        self.update_alpha()
        
        self.itr = 0
        self.loss = list()
        self.update_loss()
        self._bool_halt = False
    
        while (self.itr<self.n_itr) and not(self._bool_halt):
            #forward step
            self.grad_op()
            #backward step
            self.prox_op()
            self.update_loss()
            self.update_halt()
            self.itr += 1
        self.loss = np.array(self.loss)
    
    def _restore(self,**kwargs):
        """This method performs restoration. 
        -----------------------------------------------------------------------
        PARAMETERS:
           For more details on the input parameters see 'set_param'."""
        #user given values of parameters
        self.set_param(**kwargs)
        if self.verbose:
            print("RESTORATION PROCESS INITIATED")
            print("Initializing variables...")
        #the remaining parameters are given defaults values
        self.set_defaults(**kwargs)
        self.init_input(**kwargs)
        if self.verbose:
            print("Running restoration...")
        self.forward_backward()
        if self.rip:
            if self.verbose:
                print("Removing Isolated Pixels...")
            mask = cl.bordering_blobs_mask(self.solution)
            self.solution *= mask
        
        self.ell_solution = self.estimate_ell()
        #evaluate performance if ground_truth is given by user
        if np.array(self.ground_truth != None).all():
            self.ell_ground_truth = self.estimate_ell(self.ground_truth)
            self.evaluate_error()      

        if self.verbose:
            print("RESTORATION PROCESS DONE")
        if self.verbose:
            print('Running diagnostic...')
            self.diagnostic()
    
    def deconvolve(self, **kwargs):
        """This method performs deconvolution.
        -----------------------------------------------------------------------
        PARAMETERS:
           For more details on the input parameters see 'set_param'."""
        self._bool_dec = True
        self._restore(**kwargs)

            
    def denoise(self, **kwargs):
        """This method performs denoising.
        -----------------------------------------------------------------------
        PARAMETERS:
           For more details on the input parameters see 'set_param'."""
        self._bool_dec = False
        self._restore(**kwargs)
    
    def estimate_ell(self, img = None):
        """This method estimates the ellipticity, e, of the image, img.
        INPUT: img, 2D Numpy Array
        OUTPUT: e, 1D Numpy Array (tuple)"""
        if np.array(img == None).any():
            img = self.solution
        GX = np.array([cl.scal(img,U_i) for U_i in self.U])
        mu20 = 0.5*(GX[3]+GX[4])-GX[0]**2/GX[2]
        mu02 = 0.5*(GX[3]-GX[4])-GX[1]**2/GX[2]
        mu11 = GX[5]-GX[0]*GX[1]/GX[2]
        e1 = (mu20-mu02)/(mu20+mu02)
        e2 = 2*(mu11)/(mu20+mu02)
        e = np.array([e1,e2])
        return e
    
    def evaluate_error(self):
        """This method evaluate the relative ellipticity and Mean Square errors
        of the solution.
        INPUT: None
        OUTPUT: None"""
        self.relative_mse = ((self.solution-self.ground_truth)**2).mean()/ \
                             (self.ground_truth**2).mean()
        self.relative_ell_error = \
        ((self.ell_solution-self.ell_ground_truth)**2).mean()/ \
                                           (self.ell_ground_truth**2).mean()
        
    
    def diagnostic(self,**kwargs):
        """This method prints the diagnostic of score.
        INPUT: ground_truth (optional), 2D Numpy Array
        OUTPUT: None"""
        
        #variables name and value lists
        name_list = list()
        value_list = list()
        
        #starlets generation
        name_list += ['starlets generation']
        value_list += [self.starlet_gen]
        
        #beta, gradient descent step-size
        name_list += ['beta']
        value_list += [self.beta]
        
        #k, k-MAD
        name_list += ['k']
        value_list += [self.k]
        
        #gamma, trade-off between data-fidelity and shape constraint
        name_list += ['gamma']
        value_list += [self.gamma]
        
        #Remove Isolated Pixel option
        name_list += ['RIP']
        value_list += [self.rip]
        
        if self._bool_dec:
            #tolerance for halt criterion
            name_list += ['tolerance']
            value_list += [self.tolerance]
            
            #halt criterion value
            name_list += ['halt criterion']
            value_list += [self._bool_halt]
        
        #fetch ground truth if given by user
        if 'ground_truth' in kwargs: 
            if type(kwargs['ground_truth']) is np.ndarray:
                if np.ndim(kwargs['ground_truth'])==2:
                    self.ground_truth = kwargs['ground_truth']
                    self.ell_ground_truth = self.estimate_ell(self.ground_truth)
                else:
                    raise ValueError(np2d_error_message.format('ground_truth'))
            else:
                raise TypeError(np2d_error_message.format('ground_truth'))
            
        if np.array(self.ground_truth != None).all():
            
            if (self.relative_ell_error == None)or(self.relative_mse == None):
                self.evaluate_error()
            
            #relative pixel Mean Square Error of the solution
            name_list += ['relative pixel MSE']
            value_list += [self.relative_mse]
            
            #relative ellipticity error of the solution
            name_list += ['relative ellipticity error']
            value_list += [self.relative_ell_error]
            
        #relative ellipticity error of the solution
        name_list += ['# of iterations']
        value_list += [self.itr]
        
        #relative ellipticity error of the solution
        name_list += ['total # of iterations']
        value_list += [self.itr]
        
        max_len = (max(len(name) for name in name_list))+2
        row_format ="{:<"+str(max_len)+"}{:.5f}"
        for variable, row in zip(name_list, value_list):
            print(row_format.format(variable, row))