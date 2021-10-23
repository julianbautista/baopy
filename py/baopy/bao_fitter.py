from math import e
import numpy as np
import pylab as plt
import hankl
import iminuit
import scipy.interpolate 
import scipy.linalg

from scipy.optimize import curve_fit

plt.ion()

def legendre(ell, mu):

    if ell == 0:
        return mu*0+1
    elif ell == 2:
        return 0.5 * (3*mu**2-1)
    elif ell == 4:
        return 1/8 * (35*mu**4 - 30*mu**2 +3)
    elif ell == 6:
        return 1/16 * (231*mu**6 - 315*mu**4 + 105*mu**2 - 5)
    elif ell == 8:
        return 1/128* (6435*mu**8 - 12012*mu**6 + 6930*mu**4 - 1260*mu**2 + 35)
    else:
        return -1

def multipoles(x, ell_max=8):
    ''' Get multipoles of any function of ell
        It assumes symmetry around mu=0 
        Input
        -----
        x: np.array with shape (nmu, nx) from which the multipoles will be computed    
        mu: np.array with shape (nmu) where nmu is the number of mu bins
        
        Returns
        ----
        f_mult: np.array with shape (nell, nx) 
    '''
    
    n_mu = x.shape[0]
    n_x = x.shape[1]
    n_ell = ell_max//2+1
    x_mult = np.zeros((n_ell, n_x))
    ell = np.arange(0, ell_max+2, 2)
    mu = np.linspace(0, 1, n_mu)
    for i in range(n_ell):
        leg = legendre(ell[i], mu)
        x_mult[i] = np.trapz(x*leg[:, None], x=mu, axis=0)
        x_mult[i] *= (2*ell[i]+1)
    return x_mult





class Model:

    def __init__(self, pk_file=None): 

        k, pk = np.loadtxt(pk_file, unpack=True)
        #-- Fourier transform the power spectrum to correlation function
        r, xi = hankl.P2xi(k, pk, l=0)
        xi = xi.real

        #-- Get xi without BAO peak 
        xi_nopeak = self.get_sideband_xi(r, xi)

        #-- Get pk without BAO peak with an inverse Fourier transform
        _, pk_nopeak = hankl.xi2P(r, xi_nopeak, l=0) 
        pk_nopeak = pk_nopeak.real

        self.r = r
        self.xi = xi
        self.xi_nopeak = xi_nopeak
        self.k = k
        self.pk = pk
        self.pk_nopeak = pk_nopeak
        self.mu = None
        self.mu_2d = None 
        self.k_2d = None

    def get_sideband_xi(self, r, xi, 
        r_range=[[50., 80.], [160., 190.]]):
        ''' Gets sideband correlation function, i.e., the correlation function
            without the BAO peak.

            r_range is defines the boundaries of data used to fit the sideband

            Need to check a better algorithm from Bernal et al. 2020
            https://arxiv.org/abs/2004.07263 
        '''

        peak_range = [r_range[0][1], r_range[1][0]]
        w = (((r > r_range[0][0]) & 
              (r < r_range[0][1])) | 
             ((r > r_range[1][0]) &
              (r < r_range[1][1])))
        x_fit = r[w]
        y_fit = xi[w]

        def broadband(x, *pars):
            xx = x/100
            return pars[0]*xx + pars[1] + pars[2]/xx + pars[3]/xx**2 + \
                   pars[4]*xx**2 + pars[5]*xx**3 + pars[6]*xx**4  

        popt, _ = curve_fit(broadband, x_fit, y_fit,
                                p0=[0, 0, 0, 0, 0, 0, 0])
       
        xi_sideband = xi*1.
        w_peak = (r>peak_range[0])&(r<peak_range[1])
        xi_sideband[w_peak] = broadband(r[w_peak], *popt)
        
        self.xi_model = broadband(r, *popt)
        self.xi_sideband = xi_sideband
        self.peak_range = peak_range
        self.r_range = r_range

        return xi_sideband


    def plots_pk_cf_sidebands(self):
        k = self.k
        pk = self.pk
        pks = self.pk_nopeak
        r = self.r
        xi = self.xi
        xis = self.xi_nopeak

        plt.figure(figsize=(6,4))
        plt.plot(k, pk*k, 'k', lw=2)
        plt.plot(k, pks*k, 'r--', lw=2)
        plt.xscale('log')
        plt.xlabel(r'$k \ [h \ \rm{Mpc}^{-1}]$')
        plt.ylabel(r'$kP_{\rm lin}(k) \ [h^{-2}\mathrm{Mpc}^2]$')
        plt.xlim(1e-3, 10)
        plt.tight_layout()

        plt.figure(figsize=(6,4))
        plt.plot(r, xi*r**2, 'k', lw=2)
        plt.plot(r, xis*r**2, 'r--', lw=2)
        plt.xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        plt.ylabel(r'$r^2 \xi_{\rm lin} \ [h^{-2} \mathrm{Mpc}^{2}]$')
        plt.xlim(0, 200)
        plt.tight_layout()
        plt.show()

    def get_pk_2d(self, pars, ell_max=4, no_peak=False, decouple_peak=True):
        ''' Compute P(k, mu) for a set of parameters and
            pk, pk_sideband
        Input
        -----
        pars (dict): available parameters are:
                     alpha_perp - alpha perpendicular or transverse
                     alpha_para - alpha parallel to line-of-sight or radial
                     alpha_iso - alpha isotropic
                     epsilon - anisotropic parameter
                     bias - linear large scale bias parameter
                     beta - redshift space distortion parameter
                     growth_rate - growth rate of structures
                     sigma_perp - damping of BAO in the perpendicular direction
                     sigma_para - damping of BAO in the parallel direction
                     sigma_iso - damping of BAO isotropic
                     sigma_fog - Finger's of God damping
                     sigma_rec - Reconstruction damping
                     bias2 - linear bias for second tracer
                     beta2 - RSD parameter for second tracer
                     beam - transverse damping (for 21cm data)
        '''

        if 'alpha_iso' in pars:
            alpha_perp = pars['alpha_iso']/(1+pars['epsilon'])
            alpha_para = pars['alpha_iso']*(1+pars['epsilon'])**2
        elif 'alpha_para' in pars:
            alpha_perp = pars['alpha_perp']
            alpha_para = pars['alpha_para']
        else:
            alpha_perp = 1.
            alpha_para = 1.
       
        if 'sigma_iso' in pars:
            sigma_para = pars['sigma_iso']
            sigma_perp = pars['sigma_iso']
        elif 'sigma_para' in pars or 'sigma_perp' in pars:
            sigma_para = pars['sigma_para']
            sigma_perp = pars['sigma_perp']
        else:
            sigma_para = 0
            sigma_perp = 0

        #-- Read bias and growth rate / RSD parameter
        bias = pars['bias']
        beta = pars['beta'] if 'beta' in pars else pars['growth_rate']/bias

        #-- Read parameters for cross-correlation
        #-- or making them equal if fitting auto-correlation
        bias2 = pars['bias2'] if 'bias2' in pars else bias*1
        beta2 = pars['beta2'] if 'beta2' in pars else beta*bias/bias2
 
        k = self.k
        pk = self.pk
        pk_nopeak = self.pk_nopeak

        #-- Defining 2D arrays for mu and k 
        #-- These have shape = (nmu, nk) 
        if self.mu is None:
            mu = np.linspace(0, 1, 101)
            mu_2d = np.tile(mu, (k.size, 1)).T
            k_2d = np.tile(k, (mu.size, 1))
        else:
            mu = self.mu 
            mu_2d = self.mu_2d 
            k_2d = self.k_2d 

        #-- Scale k and mu by alphas 
        #-- This is the correct formula (Eq. 58 and 59 from Beutler et al. 2014)
        factor_ap = alpha_para/alpha_perp
        ak_2d = k_2d/alpha_perp  * np.sqrt( 1 + mu_2d**2 * (1/factor_ap**2 - 1) )
        amu_2d = mu_2d/factor_ap / np.sqrt( 1 + mu_2d**2 * (1/factor_ap**2 - 1) )

        #-- Sideband model (no BAO peak)
        #-- If decouple_peak ==  True, do not scale sideband by alpha
        if decouple_peak:
            pk_nopeak_2d = np.tile(pk_nopeak, (mu.size, 1))
        else:
            pk_nopeak_2d = np.interp(ak_2d, k, pk_nopeak)

        #-- Anisotropic damping applied to BAO peak only in Fourier space
        #-- exp^{- 0.5 k^2 [ (1-\mu^2)\Sigma_\perp^2 + \mu^2 \Sigma_\parallel^2 ] }
        if no_peak:
            pk_2d = pk_nopeak_2d
        else:
            sigma_nl_k2 = ak_2d**2 * ( (1-amu_2d**2)*sigma_perp**2 + amu_2d**2*sigma_para**2 )
            #sigma_nl_k2 =   k_2d**2 * ( (1- mu_2d**2)*sigma_perp**2 +  mu_2d**2*sigma_para**2 )
            #-- Scale BAO peak part by alphas
            pk_peak_2d = np.interp(ak_2d, k, pk-pk_nopeak)
            pk_2d  = pk_peak_2d * np.exp(-0.5*sigma_nl_k2)
            pk_2d += pk_nopeak_2d

        #-- Reconstruction damping
        #-- exp^{ -0.5 k^2 \mu^2 \Sigma_r^2 }
        if 'sigma_rec' in pars and pars['sigma_rec'] != 0:
            recon_damp = 1 - np.exp(-0.5*(ak_2d*pars['sigma_rec'])**2) 
            #recon_damp =  1 - np.exp(-0.5*( k_2d*pars['sigma_rec'])**2) 
        else:
            recon_damp = 1.

        #-- Fingers of God
        if 'sigma_fog' in pars and pars['sigma_fog'] != 0:
            fog = 1./( 1 + 0.5*(amu_2d*ak_2d*pars['sigma_fog'])**2)
            #fog =  1./( 1 + 0.5*( mu_2d* k_2d*pars['sigma_fog'])**2)
        else:
            fog = 1.

        #-- This parameters is for intensity mapping only
        if 'beam' in pars:
            beam = np.exp( - 0.5*pars['beam']**2*np.outer(1-mu**2, k**2)) 
        else:
            beam = 1.
        
        #-- Kaiser redshift-space distortions
        kaiser = (bias  * (1 + beta *amu_2d**2*recon_damp) * 
                  bias2 * (1 + beta2*amu_2d**2*recon_damp))
        #kaiser = (bias  * (1 + beta *mu_2d**2*recon_damp) * 
        #          bias2 * (1 + beta2*mu_2d**2*recon_damp))
        
        pk_2d *= kaiser
        pk_2d *= fog**2 
        pk_2d *= beam
        pk_2d /= (alpha_perp**2*alpha_para)

        #-- Multipoles of pk
        pk_mult = multipoles(pk_2d, ell_max=ell_max)
        
        #-- Multipoles of xi
        ell = np.arange(0, ell_max+2, 2)
        xi_mult = np.zeros((ell.size, k.size))
        for i in range(ell.size):
            r, xi = hankl.P2xi(k, pk_mult[i], l=ell[i])
            xi_mult[i] = xi.real

        self.ell = ell
        self.mu = mu
        self.mu_2d = mu_2d 
        self.k_2d = k_2d
        self.r_mult = r
        self.ell = ell
        self.pk_mult = pk_mult
        self.xi_mult = xi_mult
        self.pk_2d = pk_2d

    def get_multipoles(self, data_space, data_ell, data_x, pars):
        ''' General function that computes pk or xi multipoles 
            from a set of parameters

            Input
            -----
            data_space (np.array) :  Array containing either 0 or 1 ( 0 = pk,  1 = xi)
            data_ell (np.array) : Array containing the order of multipoles
            data_x (np.array) : Array containing the x-axis value (k for pk, r for xi)
            pars (dict): Dictionary containing the parameters of the model, see get_pk_2d()

            Returns
            -------
            y (np.array): Array containing the interpolated value of the model 

        '''
        self.get_pk_2d(pars)

        n_data = data_x.size
        model = np.zeros(n_data)
        
        #-- Loop over xi and pk 
        for space, x_model, y_model in zip([0,            1], 
                                           [self.k,       self.r], 
                                           [self.pk_mult, self.xi_mult]):
            if space not in data_space: 
                continue
            w_s = (data_space == space)
            #-- Loop over multipole order
            for l in np.unique(data_ell[w_s]):
                w = w_s & (data_ell == l)
                i = int(l)//2 
                model[w] = np.interp(data_x[w], x_model, y_model[i])

        return model

    def plot_multipoles(self, f=None, axs=None, ell_max=None, 
        k_range=(0., 0.5), r_range=(0, 200), convolved=False):

        ell = self.ell
        n_ell = ell.size
        k = self.k
        r = self.r
        if convolved:
            pk_mult = self.pk_mult_convol
            xi_mult = self.xi_mult_convol
        else: 
            pk_mult = self.pk_mult
            xi_mult = self.xi_mult

        if ell_max is None:
            n_ell = ell.size
        else:
            n_ell = ell_max//2 + 1

        if f is None:
            f, axs = plt.subplots(ncols=2, nrows=n_ell, figsize=(8, 7), sharex='col')

        for i in range(n_ell):
            axs[i, 0].plot(k, pk_mult[i]*k**2, color=f'C{i}', label=r'$\ell=$'+f'{ell[i]}')
            axs[i, 0].set_xlim(k_range)
            axs[i, 0].legend()
        
        for i in range(n_ell):
            axs[i, 1].plot(r, xi_mult[i]*r**2, color=f'C{i}', label=r'$\ell=$'+f'{ell[i]}')
            axs[i, 1].set_xlim(r_range)
            axs[i, 1].legend()

        axs[-1, 0].set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
        axs[-1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
        axs[0, 0].set_title(r'$k^2 P_\ell(k)$')
        axs[0, 1].set_title(r'$r^2 \xi_\ell(k)$')

        return f, axs

    def test_with_fake_data(self):

        #-- Some example parameters
        pars = {'alpha_para': 1., 
                'alpha_perp': 1., 
                'bias': 1., 
                'beta': 0.3, 
                'sigma_para': 10., 
                'sigma_perp': 6., 
                'sigma_fog': 4. }
        
        #-- Some fake data vector
        data_space = np.array([0, 1])
        data_ell = np.array([0, 2, 4])
        data_k = np.linspace(0.01, 0.3, 11)
        data_r = np.linspace(50, 150, 21)

        data = []
        for s in data_space:
            for ell in data_ell:
                if s == 0:
                    for x in data_k:
                        data.append([s, ell, x])
                if s == 1:
                    for x in data_r:
                        data.append([s, ell, x])
        data = np.array(data)
        data = {'space': data[:, 0], 'ell': data[:, 1], 'x': data[:, 2]}

        #-- Calculate model at the requested coordinates
        data_y = self.get_multipoles(data['space'], data['ell'], data['x'], pars)
        
        #-- Make the plots of model
        f, axs = self.plot_multipoles()       
        
        #-- Add 'data' points 
        for s in np.unique(data['space']):
            for l in np.unique(data['ell']):
                w = (data['space'] == s) & (data['ell'] == l)
                if np.sum(w)>0:
                    x = data['x'][w]
                    y = data_y[w] 
                    i_row = int(l)//2
                    i_col = int(s)
                    axs[i_row, i_col].plot(x, y*x**2, 'o')

    def read_window_function(self, window_file):
        ''' Reads windown function multipoles from file
            This file format was created by R. Neveux
        '''
        data = np.loadtxt(window_file)
        r_window = data[0]
        window = data[1:]

        n_ell = window.shape[0]
        n_r = self.r.size
        window_mult = np.zeros((n_ell, n_r))
        for i in range(n_ell):
            win = window[i]
            window_spline = scipy.interpolate.InterpolatedUnivariateSpline(r_window, win)
            window_mult[i] = window_spline(self.r)
        self.window_mult = window_mult

    def get_multipoles_window(self):
        ''' Get xi and pk multipoles convolved by a window function

            Compute convolved multipoles of correlation function 
            given Eq. 19, 20 and 21 of Beutler et al. 2017

            NOT YET TESTED
        '''

        xi = self.xi_mult
        win = self.window_mult
        ell = self.ell 

        #-- Computes convolved xi 
        xi_mono = (  xi[0]*win[0] 
                   + xi[1]*(1/5*win[1]) 
                   + xi[2]*(1/9*win[2]) )
        xi_quad = (  xi[0]*win[1] 
                   + xi[1]*(win[0] + 2/7*win[1] + 2/7*win[2]) 
                   + xi[2]*(2/7*win[1] + 100/693*win[2] + 25/143*win[3]) )
        xi_hexa = (  xi[0]*win[2] 
                   + xi[1]*(18/35*win[1] + 20/77*win[2] + 45/143*win[3]) 
                   + xi[2]*(win[0] + 20/77*win[1] + 162/1001*win[2] + 20/143*win[3] + 490/2431*win[4]) )
        xi_convol = np.array([xi_mono, xi_quad, xi_hexa])

        #-- Computes convoled pk with Hankel transforms
        pk_convol = np.zeros_like(xi_convol)
        for i in range(ell.size):
            k, pk = hankl.xi2P(self.r, xi_convol[i], l=ell[i])
            pk_convol[i] = pk.real

        self.xi_mult_convol = xi_convol 
        self.pk_mult_convol = pk_convol 

class Data: 

    def __init__(self, data_file=None, cova_file=None):

        space, ell, scale, y_value = np.loadtxt(data_file, unpack=1)
        s1, l1, x1, s2, l2, x2, cova_12 = np.loadtxt(cova_file, unpack=1)
        
        coords = {'space': space.astype(int), 'ell': ell.astype(int), 'scale': scale}
        cova = np.reshape(cova_12, (y_value.size, y_value.size))
        self.coords = coords
        self.y_value = y_value
        self.cova = cova
       
    def apply_cuts(self, cuts):

        coords = self.coords
        self.coords = {k: coords[k][cuts] for k in coords}
        self.y_value = self.y_value[cuts]
        cova = self.cova[:, cuts]
        self.cova = cova[cuts]
    
    def inverse_cova(self, nmocks=0):
        inv_cova = np.linalg.inv(self.cova)
        if nmocks > 0:
            correction = (1 - (self.y_value.size + 1.)/(nmocks-1))
            inv_cova *= correction
        self.inv_cova = inv_cova
    
    def plot(self, y_model=None, f=None, axs=None, scale_r=2, label=None, figsize=(7, 8)):

        coords = self.coords
        space = np.unique(coords['space'])
        n_space = space.size
        ell = np.unique(coords['ell'])
        n_ell = ell.size

        y_value = self.y_value
        y_error = np.sqrt(np.diag(self.cova))

        if f is None:
            f, axs = plt.subplots(ncols=n_space, nrows=n_ell, figsize=(8, 7), sharex='col', squeeze=False)

        xlabels = [r'$k$ [$h$ Mpc$^{-1}$]', r'$r$ [$h^{-1}$ Mpc]']
        titles = [r'$k^2 P_\ell(k)$', r'$r^2 \xi_\ell(k)$']

        for col in range(n_space):
            w_space = coords['space'] == space[col]
            for row in range(n_ell):
                w_ell = coords['ell'] == ell[row]
                w = w_space & w_ell
                x = coords['scale'][w]
                
                axs[row, col].errorbar(x, y_value[w]*x**scale_r, y_error[w]*x**scale_r, fmt='o',
                    color=f'C{row}')
                if not y_model is None:
                    axs[row, col].plot(x, y_model[w]*x**scale_r, color=f'C{row}')
            axs[row, col].set_xlabel(xlabels[space[col]])
            axs[0, col].set_title(titles[space[col]])

        return f, axs
    
class Chi2: 

    def __init__(self, data=None, model=None, parameters=None, options=None):
        self.data = data
        self.model = model
        self.parameters = parameters
        self.options = options
        self.ndata = data.y_value.size 
        
        self.setup_iminuit()
        if 'fit_broadband' in options and options['fit_broadband'] == True:
            bb_min = options['bb_min'] if 'bb_min' in options else -2
            bb_max = options['bb_max'] if 'bb_max' in options else 0
            self.setup_broadband(bb_min=bb_min, bb_max=bb_max)

    def setup_iminuit(self):

        parameters = self.parameters 
        
        #-- Obtain list of names and values to initialise Minuit class
        pars_names = []
        pars_values = []
        for par in parameters:
            pars_names.append(par)
            pars_values.append(parameters[par]['value'])

        mig = iminuit.Minuit(self, tuple(pars_values), name=tuple(pars_names))
        mig.errordef = iminuit.Minuit.LEAST_SQUARES

        #-- Setup fixed parameters, limits and step sizes 
        for par in parameters:
            par_dict = parameters[par]
            mig.errors[par] = par_dict['error'] if 'error' in par_dict else par_dict['value']
            mig.fixed[par] = par_dict['fixed'] if 'fixed' in par_dict else False
            mig.errors[par] = 0 if 'fixed' in par_dict and par_dict['fixed'] else mig.errors[par]
            limit_low = par_dict['limit_low'] if 'limit_low' in par_dict else None
            limit_upp = par_dict['limit_upp'] if 'limit_upp' in par_dict else None
            mig.limits[par] = (limit_low, limit_upp)
            
        self.mig = mig 

    def get_model(self, pars):
        coords = self.data.coords
        y_model = self.model.get_multipoles(coords['space'], coords['ell'], coords['scale'], pars)
        return y_model
    
    def setup_broadband(self, bb_min=-2, bb_max=0):
        ''' Setup analytical solution for best-fit polynomial nuisance terms
	        http://pdg.lbl.gov/2016/reviews/rpp2016-rev-statistics.pdf eq. 39.22 and surrounding
        '''
        coords = self.data.coords
        space = coords['space']
        ell = coords['ell']
        scale = coords['scale']

        upairs = np.unique([space, ell], axis=1)
        n_upairs = upairs.shape[1]
        power = np.arange(bb_min, bb_max+1) 
        n_bb_pars = n_upairs * power.size
        h_matrix = np.zeros((scale.size, n_bb_pars))
        for row in range(scale.size):
            col_space_ell = np.where((upairs[0]== space[row]) & (upairs[1]==ell[row]))[0][0]
            for j in range(power.size):
                col = col_space_ell*power.size + j
                p = power[j]
                h_matrix[row, col] = scale[row]**p

        norm_matrix = np.linalg.inv(h_matrix.T @ self.data.inv_cova @ h_matrix)
        self.h_matrix = h_matrix
        self.norm_matrix = norm_matrix

    def fit_broadband(self, residual):
       
        norm_matrix = self.norm_matrix
        h_matrix = self.h_matrix 
        inv_cova = self.data.inv_cova 

        #-- Eq. 39.22 of the reference
        bb_pars = norm_matrix.dot(h_matrix.T.dot(inv_cova.dot(residual)))

        return bb_pars

    def get_broadband(self, bb_pars):

        return self.h_matrix.dot(bb_pars)

    def __call__(self, p):
        ''' Compute chi2 for a set of free parameters (and only the free parameters!)
        '''

        #-- Create dictionary from array, to give it to Model
        pars = {}
        i = 0
        for i, par in enumerate(self.parameters):
            pars[par] = p[i]
            i+=1

        #-- Compute model, residuals
        data = self.data 
        model = self.model 
        coords = data.coords
        y_model = model.get_multipoles(coords['space'], coords['ell'], coords['scale'], pars)
        y_residual = data.y_value - y_model
        inv_cova = data.inv_cova

        #-- Add broadband function
        if not self.options is None and self.options['fit_broadband']:
            bb_pars = self.fit_broadband(y_residual)
            broadband = self.get_broadband(bb_pars)
            y_residual -= broadband

        chi2 = np.dot(y_residual, np.dot(inv_cova, y_residual))

        #-- Add Gaussian priors if present
        for par in pars:
            if 'prior_mean' in self.parameters[par]:
                mean = self.parameters[par]['prior_mean']
                sigma = self.parameters[par]['prior_sigma']
                chi2 += ((pars[par]-mean)/sigma)**2

        return chi2

    def log_prob(self, p):
        #print(p)
        return -0.5*self.__call__(p)    

    def fit(self):

        #-- Run the minimizer 
        mig = self.mig
        mig.migrad()

        #-- Recover best-fit parameters and model 
        best_pars = {k: mig.params[k].value for k in mig.parameters}
        best_model = self.get_model(best_pars)

        #-- Add broadband 
        if not self.options is None and self.options['fit_broadband']:
            best_bb_pars = self.fit_broadband(self.data.y_value - best_model)
            best_broadband = self.get_broadband(best_bb_pars)
            n_bb_pars = best_bb_pars.size
        else:
            best_broadband = best_model*0
            best_bb_pars = None 
            n_bb_pars = 0

        self.best_pars = best_pars
        self.best_model = best_model + best_broadband
        self.best_bb_pars = best_bb_pars 
        self.best_broadband = best_broadband

        self.mig = mig
        self.chi2min = mig.fval
        self.npar = mig.nfit + n_bb_pars
        self.ndof = self.ndata - self.npar
        self.rchi2min = self.chi2min/(self.ndof)
        print(f'chi2/(ndata-npars) = {self.chi2min:.2f}/({self.ndata}-{self.npar}) = {self.rchi2min:.2f}') 
        
    def plot(self, f=None, axs=None, scale_r=2, label=None, figsize=(10, 4)):

        f, axs = self.data.plot(y_model=self.best_model)
        return f, axs

    def mcmc(self, sampler_name='emcee', nsteps=1000, nwalkers=10, use_pool=False):
        ''' NOT YET WORKING 
        
        '''
        if sampler_name == 'zeus':
            import zeus
            sampling_function = zeus.sampler
        elif sampler_name == 'emcee':
            import emcee
            sampling_function = emcee.EnsembleSampler
        else:
            print("ERROR: Need a valid sampler: 'zeus' or 'emcee'")
            return 

        def log_prob(p):
            return self.log_prob(p)

        pars_free = [par for par in parameters if not parameters[par]['fixed']]
        npars = len(pars_free)

        #-- Use limits to set the start point for random walkers
        print('\nSetting up starting point of walkers:')
        start = np.zeros((nwalkers, npars))
        for j, par in enumerate(pars_free):
            if par in chi2.best_pars and chi2.best_pars[par]['error']>0.:
                limit_low = parameters[par]['limit_low']
                limit_upp = parameters[par]['limit_upp']
                value = chi2.best_pars[par]['value']
                error = chi2.best_pars[par]['error']
                limit_low = np.max([limit_low, value-10*error])
                limit_upp = np.min([limit_upp, value+10*error])
                print('Randomly sampling for', par, 'between', limit_low, limit_upp )
                start[:, j] = np.random.rand(nwalkers)*(limit_upp-limit_low)+limit_low

        if use_pool == 'True':
            from multiprocessing import Pool, cpu_count
            print(f'Using multiprocessing with {cpu_count()} cores')
            with Pool() as pool:
                sampler = sampling_function(nwalkers, npars, log_prob, pool=pool)
                sampler.run_mcmc(start, nsteps, progress=True)
        else:
            print('Using a single core')
            sampler = sampling_function(nwalkers, npars, log_prob)
            sampler.run_mcmc(start, nsteps, progress=True)

        if sampler_name == 'zeus':
            sampler.summary

        chain = sampler.get_chain(flat=True)
