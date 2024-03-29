''' Module containing several clustering models

'''

import numpy as np
import matplotlib.pyplot as plt
import hankl

import scipy.interpolate 
import scipy.linalg
import scipy.optimize
import scipy.interpolate
from scipy.special import legendre

def multipoles(x, ell_max=8, symmetric=True):
    ''' Get multipoles of any function of ell
        Uses ``numpy.trapz()`` to integrate over mu.  
        
        Parameters
        ----------
        x : np.array with shape (nmu, nx) 
            Function from which the multipoles will be computed
        ell_max : int 
            Maximum multipole order to compute. Currently only supports even multipoles 
        symmetric : bool 
            True if x is defined for 0 <= mu <= 1, when x is a pair function
            
        Returns
        -------
        x_mult : np.array with shape (nell, nx)
            Multipoles, where ``nell`` is the number of even multipoles between ``0`` and ``ell_max``. 

    '''
    
    n_mu = x.shape[0]
    n_x = x.shape[1]
    n_ell = ell_max//2+1
    x_mult = np.zeros((n_ell, n_x))
    ell = np.arange(0, ell_max+2, 2)
    mu_min = 0 if symmetric else -1
    mu = np.linspace(mu_min, 1, n_mu)

    for i in range(n_ell):
        leg = legendre(ell[i])(mu)
        x_mult[i] = np.trapz(x*leg[:, None], x=mu, axis=0)
        x_mult[i] *= (2*ell[i]+1)
        if not symmetric: 
            x_mult[i] /= 2
    return x_mult



class Model:
    ''' General class that holds both BAO or RSD models 

    '''
    
    def __init__(self):
        self.r = None
        self.xi = None
        self.k = None
        self.pk = None
        self.mu = None
        self.mu_2d = None 
        self.k_2d = None
        self.ell = None 
        self.pk_mult = None 
        self.xi_mult = None
        self.window_mult = None

    def get_multipoles(self, data_space, data_ell, data_x, pars):
        ''' General function that computes pk or xi multipoles 
            from a set of parameters

            Parameters
            ----------
            data_space :  np.array
                Array containing either 0 or 1 ( 0 = pk,  1 = xi)
            data_ell : np.array
                Array containing the order of multipoles
            data_x : np.array
                Array containing the x-axis value (k for pk, r for xi)
            pars : dict
                Dictionary containing the parameters of the model, see get_pk_2d()

            Returns
            -------
            y : np.array
                Array containing the interpolated value of the model 

        '''
        self.get_pk_2d(pars)

        n_data = data_x.size
        model = np.zeros(n_data)

        xi_mult = self.xi_mult        
        if not self.window_mult is None:
            pk_mult = self.pk_mult_convol
        else:
            pk_mult = self.pk_mult

        #-- Loop over xi and pk 
        for space, x_model, y_model in zip([0,       1], 
                                           [self.k,  self.r], 
                                           [pk_mult, xi_mult]):
            if space not in data_space: 
                continue
            w_s = (data_space == space)
            #-- Loop over multipole order
            for l in np.unique(data_ell[w_s]):
                w = w_s & (data_ell == l)
                i = int(l)//2 
                model[w] = np.interp(data_x[w], x_model, y_model[i])

        return model

    def plot_multipoles(self, 
        f=None, axs=None, ell_max=None, 
        k_range=(0., 0.5), r_range=(0, 200), 
        power_k=1, power_r=2, convolved=True, ls=None, figsize=None):
        ''' Plots the multipoles computed for a set of parameters 
        
        Parameters
        ----------
        f : plt.figure.Figure, optional
            If not provided, creates a new figure 
        axs : plt.axes.Axes, optional
            If not provided, creates a new set of axes 
        ell_max : int, optional
            Maximum order of multipoles to be plotted 
        k_range : tuple
            Range of wavenumbers in h/Mpc to be plotted 
        r_range : tuple
            Range of separations in Mpc/h to be plotted 
        power_k : int
            Scales the plotted power spectra by ``k**power_k``
        power_r : int 
            Scales the plotted correlation function by ``r**power_r``
        convolved : bool
            If true, plots the convolved power spectrum (the correlation function remains the same)
        
        Returns
        -------
        f : plt.figure.Figure 
            Figure object 
        axs : 
            Set of axes 

        '''
        ell = self.ell
        n_ell = ell.size
        k = self.k
        r = self.r
        if not r_range is None:
            wr = (r>=r_range[0]) & (r<=r_range[1])
        else: 
            wr = r==r 
        if not k_range is None:
            wk = (k>=k_range[0]) & (k<=k_range[1])
        else:
            wk = k==k

        if convolved and not self.window_mult is None:
            pk_mult = self.pk_mult_convol
        else: 
            pk_mult = self.pk_mult
        xi_mult = self.xi_mult

        if ell_max is None:
            n_ell = ell.size
        else:
            n_ell = ell_max//2 + 1

        if f is None:
            f, axs = plt.subplots(ncols=2, nrows=n_ell, figsize=figsize, sharex='col')

        for i in range(n_ell):
            x = k[wk] 
            y = pk_mult[i, wk]*x**power_k
            axs[i, 0].plot(x, y, color=f'C{i}', ls=ls,) #label=r'$\ell=$'+f'{ell[i]}')
            axs[i, 0].set_xlim(k_range)
            #axs[i, 0].legend()
        
        for i in range(n_ell):
            x = r[wr]
            y = xi_mult[i, wr] * x**power_r
            axs[i, 1].plot(x, y, color=f'C{i}', ls=ls, )#label=r'$\ell=$'+f'{ell[i]}')
            axs[i, 1].set_xlim(r_range)
            #axs[i, 1].legend()

        axs[-1, 0].set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]')
        axs[-1, 1].set_xlabel(r'$r$ [$h^{-1}$ Mpc]')
        axs[0, 0].set_title(r'$k^{{{power_k}}} P_\ell(k)$'.format(power_k=power_k))
        axs[0, 1].set_title(r'$r^{{{power_r}}} \xi_\ell(k)$'.format(power_r=power_r))

        return f, axs

    def test_with_fake_data(self):

        #-- Some example parameters
        pars = {'alpha_para': 1.1, 
                'alpha_perp': 0.95, 
                'bias': 2., 
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
        f, axs = self.plot_multipoles(power_k=2, power_r=2)       
        
        #-- Add 'data' points 
        for s in np.unique(data['space']):
            for l in np.unique(data['ell']):
                w = (data['space'] == s) & (data['ell'] == l)
                if np.sum(w)>0:
                    x = data['x'][w]
                    y = data_y[w] 
                    i_row = int(l)//2
                    i_col = int(s)
                    axs[i_row, i_col].plot(x, y*x**2, 'o', alpha=0.3)
        plt.show()

    def read_window_function(self, window_file):
        ''' Reads windown function multipoles from text file

            This file format was created by R. Neveux

            Parameters
            ----------
            window_file : str
                Name of text file containing the window function multipoles

        '''
        data = np.loadtxt(window_file, unpack=1)
        r_window = data[0]
        window = data[1:]

        n_ell = window.shape[0]
        n_r = self.r.size
        window_mult = np.zeros((n_ell, n_r))
        for i in range(n_ell):
            win = window[i]
            window_spline = scipy.interpolate.InterpolatedUnivariateSpline(r_window, win, ext=1)
            window_mult[i] = window_spline(self.r)
        self.window_mult = window_mult

    def get_multipoles_window(self):
        ''' Get power spectrum multipoles convolved by a window function
            following Eq. 19, 20 and 21 of Beutler et al. 2017

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
            k, pk = hankl.xi2P(self.r, xi_convol[i], l=ell[i], lowring=True)
            pk_convol[i] = pk.real

        self.pk_mult_convol = pk_convol 


class BAO(Model):
    ''' Implements the BAO model from 
        Bautista et al. 2021 
        https://ui.adsabs.harvard.edu/abs/2021MNRAS.500..736B/abstract
        and 
        Gil-Marin et al. 2020
        https://ui.adsabs.harvard.edu/abs/2020MNRAS.498.2492G/abstract
    
    '''

    def __init__(self, k=None, pk=None, pk_file=None, nopeak_method='spline'): 
        super().__init__()

        if not pk_file is None:
            print(f'Reading linear power spectrum from: {pk_file}')
            k, pk = np.loadtxt(pk_file, unpack=True)
        
        #-- Fourier transform the power spectrum to correlation function
        r, xi = hankl.P2xi(k, pk, l=0, lowring=True)
        xi = xi.real

        self.r = r
        self.xi = xi
        self.k = k
        self.pk = pk

        #-- Get xi without BAO peak 
        if nopeak_method == 'spline':
            xi_nopeak = self.get_xi_nopeak_spline()
        elif nopeak_method == 'poly':
            xi_nopeak = self.get_xi_nopeak_poly()
        else:
            print("ERROR : 'sideband_method' argument must be 'spline' or 'poly'.")

        #-- Get pk without BAO peak with an inverse Fourier transform
        _, pk_nopeak = hankl.xi2P(r, xi_nopeak, l=0, lowring=True) 
        pk_nopeak = pk_nopeak.real

        self.xi_nopeak = xi_nopeak
        self.pk_nopeak = pk_nopeak

    def get_xi_nopeak_spline(self, fix_min_max=False):
        """
            Gets the correlation function without the BAO peak
            using splines. 

            Algorithm inspired from Bernal et al. 2020
            https://arxiv.org/abs/2004.07263 
        
            Parameters
            ----------
            r : array 
                Separations
            fix_min_max : bool
                If true, it will not attempt to fit the values of rmin and rmax
            
            Returns
            -------
            xi_sm : array 
                Isotropic correlation function without BAO peak 
        """
        from iminuit import Minuit
        from iminuit.cost import LeastSquares
        
        def broadband_spl(r1, r2, s, r_min, r_max):
            """ Implements the fit for the no-peak correlation function using splines

            Parameters
            ----------
            r1, r2 : float
                Lower/upper bounds of the cut arround the peak
            s : float 
                Positive smoothing factor used to choose the number of knots : sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
            r_min, r_max : float 
                Scales used for the fit

            Returns
            -------
            xi_sm : array_like 
                An interpolated array (using cubic Splines) of the smoothed correlation function xi

            """

            w = ((r < r1)|(r > r2))
            #-- Note : for Splines interpolation the [r_min,r_max] boundaries are not needed,
            #   but without it the computation time is significantly longer and the results are not really better.
            w = (((r < r1)&(r>r_min)) | ((r > r2)&(r<r_max)))
            x_interp = r[w]
            y_interp = xi[w]*x_interp**2
            #- Rq when s get to high, oscillations appeares in pk_sm for small k 
            spl = scipy.interpolate.UnivariateSpline(x_interp, y_interp, w=None, k=3, s=s) 
            
            xi_sm_r2 = xi*r**2
            w_peak = (r>r_min)&(r<r_max)

            xi_sm_r2[w_peak] = spl(r[w_peak])
            xi_sm = xi_sm_r2*r**-2

            return xi_sm

        def get_pk_sm_spl(k, r1, r2, s, r_min, r_max):
            """
            k : array : wavevector bins
            r1,r2 : floats : lower/upper bounds of the cut
            s : float : Positive smoothing factor used to choose the number of knots : sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

            return the smoothed pk evaluated in any k. This function is meant to be fitted to get the best parameters r1,r2 (s?)
            """

            xi_sm = broadband_spl(r1, r2, s, r_min, r_max)
            _, pk_sm = hankl.xi2P(r, xi_sm, l=0, lowring=True)
            pk_sm = pk_sm.real

            return np.interp(k, self.k, pk_sm)

        r = self.r 
        xi = self.xi 

        #- Define the condition at large scales pk_sm = pk
        k_limit = 10**-3 #-- Note : taking k_limit between [5e-5,1e-2] does not affect the result very much

        w = (self.k < k_limit)
        data_x = self.k[w]
        data_y = self.pk[w]
        data_yerr = 0.1 #default err

        least_squares = LeastSquares(data_x, data_y, data_yerr, get_pk_sm_spl)
        m = Minuit(least_squares, r1=70, r2=250, s=1e-1, r_min=50, r_max=500)

        m.limits = [(60, 70), (200, 300), (0., 1), (10, 50), (350, 1e3)] 

        #allow to fix the parameters r_min and r_max if they are not relevant           
        if fix_min_max == True:
            m.fixed["r_min"] = True
            m.fixed["r_max"] = True

        m.scan(ncall=50).migrad()

        xi_sm = broadband_spl(  m.values["r1"],
                                m.values["r2"],
                                m.values["s"],
                                m.values["r_min"],
                                m.values["r_max"])
        
        return xi_sm

    def get_xi_nopeak_poly(self,  r_range=[[50., 80.], [160., 190.]]):
        ''' Gets the correlation function
            without the BAO peak using polynomial fit. 

            Based on method by Kirkby et al 2013
            https://ui.adsabs.harvard.edu/abs/2013JCAP...03..024K/abstract 

            Parameters
            ----------
            r_range : tuple
                Defines the boundaries of data used to fit the sideband

            Returns
            -------
            xi_sideband : array 
                Isotropic correlation function without BAO peak 
        '''
        r = self.r*1 
        xi = self.xi*1

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

        popt, _ = scipy.optimize.curve_fit(broadband, x_fit, y_fit, p0=[0, 0, 0, 0, 0, 0, 0])
       
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

        f, axs = plt.subplots(figsize=(12,7), nrows=2, ncols=2, sharex='col')

        ax = axs[0][0]
        ax.plot(k, pk*k, 'k', lw=2)
        ax.plot(k, pks*k, 'r--', lw=2)
        ax.set_xlabel(r'$k \ [h \ \mathrm{Mpc}^{-1}]$')
        ax.set_ylabel(r'$kP_\mathrm{lin}(k) \ [h^{-2}\mathrm{Mpc}^2]$')
        #ax.xlim(k.min(), k.max())
        
        ax = axs[1][0]
        ax.plot(k, pk/pks -1, 'C0', lw=2)
        ax.set_xscale('log')
        ax.set_xlabel(r'$k \ [h \ \rm{Mpc}^{-1}]$')
        ax.set_ylabel(r'$O_{\rm lin}(k) = P_\mathrm{lin}(k)/P_\mathrm{no peak}(k)-1$')
        
        ax = axs[0][1] 
        ax.plot(r, xi*r**2, 'k', lw=2)
        ax.plot(r, xis*r**2, 'r--', lw=2)
        ax.set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        ax.set_ylabel(r'$r^2 \xi_{\rm lin} \ [h^{-2} \mathrm{Mpc}^{2}]$')
        ax.set_xlim(0, 500)

        ax = axs[1][1] 
        ax.plot(r, (xi-xis)*r**2, 'C0', lw=2)
        ax.set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        ax.set_ylabel(r'$r^2 (\xi_\mathrm{lin}-\xi_\mathrm{no peak}) \ [h^{-2} \mathrm{Mpc}^{2}]$')
        ax.set_xlim(0, 500)

        plt.tight_layout()
        plt.show()

    def get_pk_2d(self, pars, ell_max=4, no_peak=False, decouple_peak=True, n_mu=101, mu_symmetric=True):
        ''' Compute P(k, mu) for a set of parameters and its multipoles in Fourier and Configuration Space
        
        Parameters
        ----------
        pars : dict 
            Dictionary containing the model parameters
            Available parameters: 
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
            mu_min = 0 if mu_symmetric is True else -1
            mu = np.linspace(mu_min, 1, n_mu)
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
            #recon_damp = 1 - np.exp(-0.5*(ak_2d*pars['sigma_rec'])**2) 
            recon_damp =  1 - np.exp(-0.5*( k_2d*pars['sigma_rec'])**2) 
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

        #- Lyman-alpha damping function from Arinyo-y-Prats et al. 2015
        if 'q_1' in pars:
            cutoff = 10.
            damp = np.ones_like(k)
            w = k>cutoff 
            damp[w] *= np.exp( - 0.5*(k[w]-cutoff)**2/cutoff**2)

            delta = k**3*pk_nopeak/(2*np.pi**2)
            jeans = np.exp(-k**2/pars['k_p']**2)
            term_a = (pars['q_1']*delta**2 + pars['q_2']*delta**4) * damp
            #term_b = np.outer(mu**pars['b_v'], (1- (k/pars['k_v'])**pars['a_v']))
            term_b = 1 - np.outer(mu**pars['b_v'], k**pars['a_v']/pars['k_v'])
            damp = np.exp( term_a*term_b )*jeans
        else:
            damp=1.
        
        #-- Kaiser redshift-space distortions
        kaiser = (bias  * (1 + beta *amu_2d**2*recon_damp) * 
                  bias2 * (1 + beta2*amu_2d**2*recon_damp))
        #kaiser = (bias  * (1 + beta *mu_2d**2*recon_damp) * 
        #          bias2 * (1 + beta2*mu_2d**2*recon_damp))
        
        pk_2d *= kaiser
        pk_2d *= fog**2 
        pk_2d *= beam
        pk_2d *= damp 

        #-- Multipoles of pk
        pk_mult = multipoles(pk_2d, ell_max=ell_max, symmetric=mu_symmetric)
        
        #-- Multipoles of xi
        ell = np.arange(0, ell_max+2, 2)
        xi_mult = np.zeros((ell.size, k.size))
        for i in range(ell.size):
            r, xi = hankl.P2xi(k, pk_mult[i], l=ell[i], lowring=True)
            xi_mult[i] = xi.real

        pk_2d /= (alpha_perp**2*alpha_para)
        pk_mult /= (alpha_perp**2*alpha_para)
            
        self.ell = ell
        self.mu = mu
        self.mu_2d = mu_2d 
        self.k_2d = k_2d
        self.r_mult = r
        self.ell = ell
        self.pk_mult = pk_mult
        self.xi_mult = xi_mult
        self.pk_2d = pk_2d
        
        if not self.window_mult is None:
            self.get_multipoles_window()
            self.pk_mult_convol *= 1/(alpha_perp**2*alpha_para)

        
class RSD_TNS(Model):
    
    ''' 
    Implements the TNS model (Taruya et al. 2012)
    Written by Vincenzo Aronica and Julian Bautista
    
    2-loop Power spectra, 1-loop bias terms and RSD correction terms 
    can be computed with pyRegPT : https://github.com/adematti/pyregpt

    '''
    def __init__(self, 
        pk_regpt_file=None, 
        bias_file = None, 
        a2loop_file=None, 
        b2loop_file=None,
        kmax_cutoff=0.5,
        ): 
        super().__init__()
    
        #-- Read pyregpt power spectra parameters
        pk_regpt = np.loadtxt(pk_regpt_file, unpack=True)
        k = pk_regpt[0]
        pk_regpt = pk_regpt[1:]

        #-- Fourier transform the power spectrum to correlation function
        xi_regpt = np.zeros_like(pk_regpt)
        for i in range(3):
            r, xi = hankl.P2xi(k, pk_regpt[i], l=0, lowring=True)
            xi_regpt[i] = xi.real

        #-- Read bias expansion terms
        pk_bias = np.loadtxt(bias_file, unpack=True)[1:]
        w = k>kmax_cutoff
        
        pk_bias[:, w] *= np.exp( - (k[w]/kmax_cutoff-1)**2)

        a_terms = np.loadtxt(a2loop_file, unpack=True)[1:]

        b_terms = np.loadtxt(b2loop_file, unpack=True)[1:]

        self.r = r
        self.k = k

        self.pk_regpt = pk_regpt
        self.pk_bias = pk_bias
        self.a_terms = a_terms
        self.b_terms = b_terms
      
        self.xi_regpt = xi_regpt
        self.mu = None
        self.mu_2d = None 
        self.k_2d = None
        self.window_mult = None

    def plot(self):
        ''' This just plots all terms
        '''
        plt.figure()
        k = self.k 
        plt.plot(k, self.pk_regpt.T)
        plt.legend([r'$P_{\delta \delta}$', r'$P_{\delta \theta}$', r'$P_{\theta \theta}$'])
        plt.xscale('log')
        plt.title('RegPT 2-loop')

        plt.figure()
        plt.plot(k, self.pk_bias.T)
        plt.xscale('log')
        plt.title('Bias terms')
        plt.legend([r'$P_{b2, \delta}$',
            r'$P_{bs2, \delta}$',
            r'$P_{b22}$',
            r'$P_{b2s2}$',
            r'$P_{bs22}$',
            r'$P_{b2, \theta}$',
            r'$P_{bs2, \theta}$',
            r'$\sigma_3^2 P_{\rm lin}}$',
            ])

        plt.figure()
        plt.plot(k, self.a_terms.T)
        plt.xscale('log')
        leg = [f'A{i}' for i in range(0, 5)]
        plt.legend(leg)
        plt.title('RSD A terms')

        plt.figure()
        plt.plot(k, self.b_terms.T)
        plt.xscale('log')
        leg = [f'B{i}' for i in range(0, 8)]
        plt.legend(leg)
        plt.title('RSD B terms')

    def get_pk_2d(self, pars, ell_max=4, n_mu=201, mu_symmetric=True):
        ''' Compute P(k, mu)  eq. (2.198) of De Mattia Thesis 
        Input
        -----
        pars (dict): available parameters are:
            alpha_para = scaling of radial separations
            alpha_perp = scaling of transverse separations
            b1 = linear bias
            b2 = second order bias
            bs2 = other bias paramater
            b3nl = other bias paramater
            beta = redshift space distortion parameter
            sigma_fog = Finger's of God damping
            Ng = value for shot noise power
        '''

        b1 = pars['b1']
        b2 = pars['b2']

        #-- If not free, we fix bs2 and b3nl (eq. (15)/(16) DeMattia et al. 2020) 
        bs2 = pars['bs2'] if 'bs2' in pars else -4/7*(b1-1)
        b3nl = pars['b3nl'] if 'b3nl' in pars else 32/315 * (b1-1)

        f = pars['f']
        beta = f/b1
        sn = pars['shot_noise']
        
        aper = pars['aper']
        apar = pars['apar']


        k = self.k

        #-- Defining 2D arrays for mu and k 
        #-- These have shape = (nmu, nk) 
        if self.mu is None:
            mu_min = 0 if mu_symmetric is True else -1
            mu = np.linspace(mu_min, 1, n_mu)
            mu_2d = np.tile(mu, (k.size, 1)).T
            k_2d = np.tile(k, (mu.size, 1))
        else:
            mu = self.mu 
            mu_2d = self.mu_2d 
            k_2d = self.k_2d 

        #-- Scale k and mu by alphas 
        #-- This is the correct formula (Eq. 58 and 59 from Beutler et al. 2014)
        factor_ap = apar/aper
        ak_2d = k_2d/aper  * np.sqrt( 1 + mu_2d**2 * (1/factor_ap**2 - 1) )
        amu_2d = mu_2d/factor_ap / np.sqrt( 1 + mu_2d**2 * (1/factor_ap**2 - 1) )
        amu = mu /factor_ap / np.sqrt( 1 + mu**2 * (1/factor_ap**2 - 1) )

        #-- Defining pk 2D 
        pk_regpt_2d = np.array([np.interp(ak_2d, k, pk) for pk in self.pk_regpt])
        pk_bias_2d = np.array([np.interp(ak_2d, k, pk) for pk in self.pk_bias])
        a_terms_2d = np.array([np.interp(ak_2d, k, pk) for pk in self.a_terms])
        b_terms_2d = np.array([np.interp(ak_2d, k, pk) for pk in self.b_terms])

        #-- Fingers of God
        if 'sigma_fog' in pars and pars['sigma_fog'] != 0:
            fog = 1./( 1 + 0.5*(amu_2d*ak_2d*pars['sigma_fog'])**2)
            #fog =  1./( 1 + 0.5*( mu_2d* k_2d*pars['sigma_fog'])**2)
        else:
            fog = 1.

        #-- RSD correction terms
        amu = amu[:, None]
        A = (beta    *  amu**2 * a_terms_2d[0] + 
             beta**2 *  amu**2 * a_terms_2d[1] +
             beta**2 *  amu**4 * a_terms_2d[2] +  
             beta**3 *  amu**4 * a_terms_2d[3] + 
             beta**3 *  amu**6 * a_terms_2d[4] ) 
             
        B = (
            beta**2 *  amu**2 * b_terms_2d[0] + 
            beta**3 *  amu**2 * b_terms_2d[1] + 
            beta**4 *  amu**2 * b_terms_2d[2] + 
            beta**2 *  amu**4 * b_terms_2d[3] + 
            beta**3 *  amu**4 * b_terms_2d[4] + 
            beta**4 *  amu**4 * b_terms_2d[5] + 
            beta**3 *  amu**6 * b_terms_2d[6] + 
            beta**4 *  amu**6 * b_terms_2d[7] +
            beta**4 *  amu**8 * b_terms_2d[8]   
            )

        pk_g_dd = (
            b1**2 * pk_regpt_2d[0] + 
            2*b1*b2 * pk_bias_2d[0] + 
            2*b1*bs2 * pk_bias_2d[1] + 
            2*b1*b3nl * pk_bias_2d[7] + 
            b2**2 * pk_bias_2d[2] + 
            2*b2*bs2 * pk_bias_2d[3] + 
            bs2**2 * pk_bias_2d[4] )


        pk_g_dt = (b1 * pk_regpt_2d[1] + 
                 b2 * pk_bias_2d[5] + 
                 bs2 * pk_bias_2d[6] + 
                 b3nl * pk_bias_2d[7])


        pk_g_tt = pk_regpt_2d[2]

        #-- Fingers of God
        pk_rsd = 1 
        pk_rsd *= fog**2
                                                             
        #-- Final model 
        pk_rsd *= (pk_g_dd + 
                   pk_g_dt * 2 * amu**2 * beta * b1  + 
                   pk_g_tt * (amu**2 * beta * b1)**2 + 
                   b1**3*A + 
                   b1**4*B)
                                                             
        

        
        #-- Multipoles of pk
        pk_mult = multipoles(pk_rsd, ell_max=ell_max, symmetric=mu_symmetric)

        #-- Multipoles of xi
        ell = np.arange(0, ell_max+2, 2)
        xi_mult = np.zeros((ell.size, k.size))
                                                             
        for i in range(ell.size):
            r, xi = hankl.P2xi(k, pk_mult[i], l=ell[i], lowring=True)
            xi_mult[i] = xi.real
            
        #-- Adding shot noise now to monopole only so it is not included in xi
        pk_mult[0] += sn
        
         #-- AP Jacobian for pk only 
        pk_mult *= 1/apar/aper*2

        self.ell = ell
        self.mu = mu
        self.amu = amu 
        
        #-- 2D quantities
        self.amu_2d = amu_2d
        self.ak_2d = ak_2d
        self.mu_2d = mu_2d 
        self.k_2d = k_2d
        self.pk_2d = pk_rsd

        #-- Multipole quantities
        self.r_mult = r
        self.ell = ell
        self.pk_mult = pk_mult
        self.xi_mult = xi_mult

        if not self.window_mult is None:
            self.get_multipoles_window()
            self.pk_mult_convol[0] += sn
            self.pk_mult_convol *= 1/apar/aper**2

            
class RSD_ME_red(Model):
    ''' 
    Implements of Moment Extansion model (Chen, Vlah & White (2020), https://arxiv.org/abs/2005.00523)
    using velocileptors : https://github.com/sfschen/velocileptors
    '''
    def __init__(self, k1=None, pk_lin=None, 
        pk_lin_file=None, theory = 'LPT', beyond_gauss = False, 
        one_loop= True,kmin = 5e-4, kmax = 0.4, 
        nk = 500, cutoff=2, extrap_min = -4, extrap_max = 3, 
        N = 2000, threads=1, jn=5, shear=True): 
        super().__init__()
        
        if not pk_lin_file is not None:
            k1, pk_lin = np.loadtxt(pk_lin_file, unpack=True)

        if theory == 'LPT':
            from velocileptors.LPT.moment_expansion_fftw import MomentExpansion
        if theory == 'EPT':
            from velocileptors.EPT.moment_expansion_fftw import MomentExpansion

        moments = MomentExpansion(
            k1, pk_lin, 
            beyond_gauss = beyond_gauss, 
            one_loop=one_loop,kmin = kmin, kmax = kmax, 
            nk = nk, cutoff=cutoff, extrap_min = extrap_min, extrap_max = extrap_max, 
            N = N, threads= threads, jn=jn, shear=shear
            )
        
        self.k1 = k1
        self.pk_lin = pk_lin
        self.window_mult = None
        self.moments=moments
        
    def get_pk_2d(self, pars, ell_max=4):
        ''' 
        Input
        -----
        pars (dict): available parameters are:

        '''
        k1 = self.k1
        pk_lin = self.pk_lin
        moments=self.moments
        
        b1 = pars['b1'] if 'b1' in pars else 0
        b2 = pars['b2'] if 'b2' in pars else 0
        bs = pars['bs'] if 'bs' in pars else 0
        b3 = pars['b3'] if 'b3' in pars else 0
        alpha0 = pars['alpha0'] if 'alpha0' in pars else 0
        alpha2 = pars['alpha2'] if 'alpha2' in pars else 0
        alpha4 = pars['alpha4'] if 'alpha4' in pars else 0
        sn = pars['sn'] if 'sn' in pars else 0
        s0 = pars['s0']if 's0' in pars else 0
        f = pars['f']  if 'f' in pars else 0
        aper = pars['aper'] if 'aper' in pars else 1
        apar = pars['apar'] if 'apar' in pars else 1

        param = [b1, b2, bs, b3, alpha0, alpha2, alpha4,sn,s0]
        
        k = moments.kv
        nus, ws = np.polynomial.legendre.leggauss(2*10)
        nus_calc = nus[0:10]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        nk=500
        pknu = np.zeros((len(nus),nk))

        for j, nu in enumerate(nus_calc):

            if 'sigma_fog' in pars and pars['sigma_fog'] != 0:
                fog = 1./( 1 + 0.5*(nu*k*pars['sigma_fog'])**2)**2
            else : 
                fog=1
                
            pknu[j,:] = fog*moments.compute_redshift_space_power_at_mu(param,f,nu,reduced=True,counterterm_c3=0,apar=apar,aperp=aper)[1]
            
        pknu[10:,:] = np.flip(pknu[0:10],axis=0)
        
        p0k = 0.5 * np.sum((ws*L0)[:,None]*pknu,axis=0)
        p2k = 2.5 * np.sum((ws*L2)[:,None]*pknu,axis=0)
        p4k = 4.5 * np.sum((ws*L4)[:,None]*pknu,axis=0)
        
        kint = np.logspace(-5,3,1024)
        damping = np.exp(-(kint/10)**2)

        p0int = loginterp(k, p0k)(kint) * damping
        p2int = loginterp(k, p2k)(kint) * damping
        p4int = loginterp(k, p4k)(kint) * damping

        r,xi0 = hankl.P2xi(kint, p0int, l=0, lowring=True)
        r,xi2 = hankl.P2xi(kint, p2int, l=2, lowring=True)
        r,xi4 = hankl.P2xi(kint, p4int, l=4, lowring=True)   
        
        pk_mult = [p0k,p2k,p4k]
        xi_mult = [xi0.real,xi2.real,xi4.real]
        ell = np.arange(0, ell_max+2, 2)
        
        
        #-- Multipole quantities
        self.r = r
        self.k = k
        self.ell = ell
        self.pk_mult = pk_mult
        self.xi_mult = xi_mult


                 
class RSD_ME_ext(Model):
    
    ''' 
    Implements of Moment Extansion model (Chen, Vlah & White (2020), https://arxiv.org/abs/2005.00523)
    using velocileptors : https://github.com/sfschen/velocileptors
    '''
                                                             
    def __init__(self, 
        pk_lin_file=None, theory = None): 
        super().__init__()
        k1,pk_lin=np.loadtxt(pk_lin_file).T
        
        if theory == 'LPT':
            from velocileptors.LPT.moment_expansion_fftw import MomentExpansion
        if theory == 'EPT':
            from velocileptors.EPT.moment_expansion_fftw import MomentExpansion
            

        mome = MomentExpansion(k1,pk_lin)
        
        moments = MomentExpansion(
                                k1,  pk_lin, beyond_gauss = False, 
                                one_loop= True,kmin = 5e-4, kmax = 0.4, 
                                nk = 500, cutoff=2, extrap_min = -4, extrap_max = 3, 
                                N = 2000, threads=1, jn=5, shear=True
                                )
        
        self.k1 = k1
        self.pk_lin = pk_lin
        self.window_mult = None
        self.moments=moments

    def get_pk_2d(self, pars, ell_max=4):
        ''' 
        Input
        -----
        pars (dict): available parameters are:

        '''
        k1 = self.k1
        pk_lin = self.pk_lin
        moments=self.moments
        
        b1 = pars['b1'] if 'b1' in pars else 0
        b2 = pars['b2'] if 'b2' in pars else 0
        bs = pars['bs'] if 'bs' in pars else 0 
        b3 = pars['b3'] if 'b3' in pars else 0
        alpha = pars['alpha'] if 'alpha' in pars else 0
        alpha_v = pars['alpha_v'] if 'alpha_v' in pars else 0
        alpha_s0 = pars['alpha_s0'] if 'alpha_s0' in pars else 0
        alpha_s2 = pars['alpha_s2'] if 'alpha_s2' in pars else 0
        sn = pars['sn'] if 'sn' in pars else 0
        sv = pars['sv'] if 'sv' in pars else 0
        s0 = pars['s0'] if 's0' in pars else 0
        c3 = pars['c3'] if 'c3' in pars else 0
        f = pars['f']
        beta = f/b1     
        aper = pars['aper']
        apar = pars['apar']

        param = [b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, sn, sv, s0]
        
        k = moments.kv
        nus, ws = np.polynomial.legendre.leggauss(2*10)
        nus_calc = nus[0:10]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        nk = 500
        pknu = np.zeros((len(nus),nk))

        for j, nu in enumerate(nus_calc):

            if 'sigma_fog' in pars and pars['sigma_fog'] != 0:
                fog = 1./( 1 + 0.5*(nu*k*pars['sigma_fog'])**2)**2
            else : 
                fog=1
                
            pknu[j,:] = fog*moments.compute_redshift_space_power_at_mu(param,f,nu,reduced=False,counterterm_c3=c3,apar=apar,aperp=aper)[1]
            
        pknu[10:,:] = np.flip(pknu[0:10],axis=0)
        
        
        
        
        
        p0k = 0.5 * np.sum((ws*L0)[:,None]*pknu,axis=0)
        p2k = 2.5 * np.sum((ws*L2)[:,None]*pknu,axis=0)
        p4k = 4.5 * np.sum((ws*L4)[:,None]*pknu,axis=0)
        
            
        
        kint = np.logspace(-5,3,1024)
        damping = np.exp(-(kint/10)**2)

        p0int = loginterp(k, p0k)(kint) * damping
        p2int = loginterp(k, p2k)(kint) * damping
        p4int = loginterp(k, p4k)(kint) * damping

        r,xi0 = hankl.P2xi(kint, p0int, l=0, lowring=True)
        r,xi2 = hankl.P2xi(kint, p2int, l=2, lowring=True)
        r,xi4 = hankl.P2xi(kint, p4int, l=4, lowring=True)   
        
        pk_mult = [p0k,p2k,p4k]
        xi_mult = [xi0.real,xi2.real,xi4.real]
        
        ell = np.arange(0, ell_max+2, 2)        

        #-- Multipole quantities
        self.r = r
        self.k = k
        self.ell = ell
        self.pk_mult = pk_mult
        self.xi_mult = xi_mult
