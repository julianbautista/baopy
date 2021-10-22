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

def get_multipoles(x, ell_max=8):
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
        k0, pk_nopeak = hankl.xi2P(r, xi_nopeak, l=0) 
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

    def get_pk_2d(self, pars, ell_max=4, no_peak=False, decouple_peak=False):
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
            pk_nopeak_2d = np.tile(pk_nopeak, (k.size, 1))
        else:
            pk_nopeak_2d = np.interp(ak_2d, k, pk_nopeak)

        #-- Anisotropic damping applied to BAO peak only in Fourier space
        #-- exp^{- 0.5 k^2 [ (1-\mu^2)\Sigma_\perp^2 + \mu^2 \Sigma_\parallel^2 ] }
        if no_peak:
            pk_2d = pk_nopeak_2d
        else:
            sigma_nl_k2 = ak_2d**2 * ( (1-amu_2d**2)*sigma_perp**2 + amu_2d**2*sigma_para**2 )
            #-- Scale BAO peak part by alphas
            pk_peak_2d = np.interp(ak_2d, k, pk-pk_nopeak)
            pk_2d  = pk_peak_2d * np.exp(-0.5*sigma_nl_k2)
            pk_2d += pk_nopeak_2d

        #-- Reconstruction damping
        #-- exp^{ -0.5 k^2 \mu^2 \Sigma_r^2 }
        if 'sigma_rec' in pars and pars['sigma_rec'] != 0:
            recon_damp = 1 - np.exp(-0.5*ak_2d**2*pars['sigma_rec']**2) 
        else:
            recon_damp = 1.

        #-- Fingers of God
        if 'sigma_fog' in pars and pars['sigma_fog'] != 0:
            fog = 1./( 1 + 0.5*(amu_2d*ak_2d*pars['sigma_fog'])**2)
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
        
        
        pk_2d *= kaiser
        pk_2d *= fog**2 
        pk_2d *= beam
        pk_2d /= (alpha_perp**2*alpha_para)

        #-- Multipoles of pk
        pk_mult = get_multipoles(pk_2d, ell_max=ell_max)
        
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
        y = np.zeros(n_data)
        
        #-- Loop over xi and pk 
        for s, x_model, y_model in zip([0,            1], 
                                       [self.k,       self.r], 
                                       [self.pk_mult, self.xi_mult]):
            if s not in data_space: 
                continue
            w_s = (data_space == s)
            #-- Loop over multipole order
            for l in np.unique(data_ell[w_s]):
                w = w_s & (data_ell == l)
                i = int(l)//2 
                y[w] = np.interp(data_x[w], x_model, y_model[i])

        return y

    def plot_multipoles(self, k_range=(0., 0.5), r_range=(0, 200)):

        ell = self.ell
        n_ell = ell.size
        k = self.k
        pk_mult = self.pk_mult
        r = self.r 
        xi_mult = self.xi_mult

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



    #-----------------------
    #-- OLD STUFF below this 
    #-----------------------


    def get_pk_multipoles(self, kout, pars, options):
        ''' Compute P_\ell(k) from a set of parameters
        Input
        -----
        kout (np.array): contains the wavevector values in h/Mpc
        pars (dict): contains the parameters required for P(k, \mu) 
        options (dict): contains options for the calculation including
                        ell_max, no_peak, decouple_peak, apply_window
        Output
        -----
        pk_mult_out (np.array): array with shape (n_ell, kout.size) with the P_\ell(k)
        '''
        pk2d = self.get_2d_power_spectrum(pars, 
                                          no_peak =options['no_peak'], 
                                          decouple_peak = options['decouple_peak'])
        pk_mult = self.get_multipoles(self.mu, pk2d, ell_max=options['ell_max'])

        if options['apply_window']:
            _, xi_mult = self.get_xi_multipoles_from_pk(self.k, pk_mult, output_r=self.r) 
            xi_convol = self.get_convoled_xi(xi_mult, self.window_mult)
            _, pk_mult_out = self.get_pk_multipoles_from_xi(self.r, xi_convol, output_k=kout)
        else:
            pk_mult_out = []
            for pk in pk_mult:
                pk_mult_out.append(np.interp(kout, self.k, pk))
        pk_mult_out = np.array(pk_mult_out)

        return pk_mult_out

    def read_window_function(self, window_file):
        data = np.loadtxt(window_file)
        r_window = data[0]
        window = data[1:]

        window_mult = []
        for win in window:
           window_spline = scipy.interpolate.InterpolatedUnivariateSpline(r_window, win)
           window_mult.append(window_spline(self.r))
        window_mult = np.array(window_mult)
        self.window_mult = window_mult

    def get_convolved_xi(self, xi_mult, window_mult):
        ''' Compute convolved multipoles of correlation function 
            given Eq. 19, 20 and 21 of Beutler et al. 2017 
        ''' 
        xi = xi_mult
        win = window_mult

        #-- Mono
        xi_mono = xi[0]*win[0] + xi[1]*(1/5 * win[1]) + xi[2]*(1/9*win[2])
        #-- Quad 
        xi_quad = xi[0]*win[1] + xi[1]*(      win[0] + 2/7    *win[1] + 2/7     *win[2]) \
                               + xi[2]*(2/7 * win[1] + 100/693*win[2] + 25/143  *win[3])
        #-- Hexa
        xi_hexa = xi[0]*win[2] + xi[1]*(18/35*win[1] + 20/77  *win[2] + 45/143  *win[3]) \
                + xi[2]*(win[0] + 20/77 *win[1] + 162/1001*win[2] + 20/143*win[3] + 490/2431*win[4])
    
        xi_conv = np.array([xi_mono, xi_quad, xi_hexa])
        return xi_conv
    
class Data: 

    def __init__(self, r, mono, coss, quad=None, hexa=None, 
                 rmin=40., rmax=180., nmocks=None):

        cf = mono
        rr = r
        if not quad is None:
            rr = np.append(rr, r)
            cf = np.append(cf, quad)
        if not hexa is None:
            rr = np.append(rr, r)
            cf = np.append(cf, hexa)
   
        
        ncf = cf.size
        ncov = coss.shape[0]
        
        print(' Size r:', r.size) 
        print(' Size cf:', ncf)
        print(' Size cov:', ncov)
       
        if ncf > ncov or ncov % mono.size > 0:
            print('Problem: covariance shape is not compatible '+
                  f'with correlation function. CF size: {ncf}  COV shape: {coss.shape}')
            
        if ncf < ncov:
            print('Covariance matrix is larger than correlation function. Trying to cut')
            coss = coss[:, :ncf]
            coss = coss[:ncf, :]

        w = (rr>rmin) & (rr<rmax)
        rr = rr[w]
        cf = cf[w]
        coss = coss[:, w]
        coss = coss[w, :]
        
        print(f' After cutting {rmin:.1f} < r < {rmax:.1f}:')
        print(' Size cf:', cf.size)
        print(' Size cov:', coss.shape[0])

        self.rr = rr
        self.r = np.unique(rr)
        self.cf = cf
        self.coss = coss
        self.nmul = rr.size//self.r.size
        print('Covariance matrix is positive definite?', np.all(np.linalg.eigvals(coss)>0))
        self.icoss = np.linalg.inv(coss)
        if nmocks:
            correction = (1 - (cf.size + 1.)/(nmocks-1))
            self.icoss *= correction
    
class Chi2: 

    def __init__(self, data=None, model=None, parameters=None, options=None):
        self.data = data
        self.model = model
        self.parameters = parameters
        self.options = options
        if options['fit_broadband']:
            self.setup_broadband_H()
        self.best_pars = None
        #print(parameters)

    def get_model(self, r, pars=None):
        if pars is None:
            pars = self.best_pars
        model = self.model.get_xi_multipoles(r, pars, self.options)
        return model.ravel()
    
    def setup_broadband_H(self, r=None, bb_min=None, bb_max=None):
        ''' Setup analytical solution for best-fit polynomial nuisance terms
	        http://pdg.lbl.gov/2016/reviews/rpp2016-rev-statistics.pdf eq. 39.22 and surrounding
        '''
        if r is None:
            r = self.data.rr
        if bb_min is None:
            bb_min = self.options['bb_min']
        if bb_max is None:
            bb_max = self.options['bb_max']

        rr = np.unique(r)
        nmul = r.size//rr.size
        power = np.arange(bb_min, bb_max+1) 
        H = rr[:, None]**power 
        H = np.kron(np.eye(nmul), H)
        self.H = H
        return H

    def get_broadband(self, bb_pars, r=None, H=None):

        H = self.setup_broadband_H(r) if H is None else H 
        return H.dot(bb_pars)

    def fit_broadband(self, residual, icoss, H):
       
        if hasattr(self, 'inv_HWH'):
            inv_HWH = self.inv_HWH 
        else:
            inv_HWH = np.linalg.inv(H.T.dot(icoss.dot(H)))
            self.inv_HWH = inv_HWH

        bb_pars = inv_HWH.dot(H.T.dot(icoss.dot(residual)))

        return bb_pars

    def __call__(self, p):
        ''' Compute chi2 for a set of free parameters (and only the free parameters!)
        '''
        pars = {}
        i = 0
        for par in self.parameters:
            if self.parameters[par]['fixed']:
                pars[par] = self.parameters[par]['value']
            else:
                pars[par] = p[i]
                limit_low = self.parameters[par]['limit_low']
                if not limit_low is None and p[i]<limit_low:
                    return np.inf
                limit_upp = self.parameters[par]['limit_upp']
                if not limit_upp is None and p[i]>limit_upp:
                    return np.inf
                i+=1

        model = self.get_model(self.data.r, pars)
        residual = self.data.cf - model
        inv_cov = self.data.icoss

        if self.options['fit_broadband']:
            bb_pars = self.fit_broadband(residual, inv_cov, self.H)
            bb = self.get_broadband(bb_pars, H=self.H)
            residual -= bb

        chi2 = np.dot(residual, np.dot(inv_cov, residual))

        for par in pars:
            if par in self.parameters and 'prior_mean' in self.parameters[par]:
                mean = self.parameters[par]['prior_mean']
                sigma = self.parameters[par]['prior_sigma']
                chi2 += ((pars[par]-mean)/sigma)**2

        return chi2

    def log_prob(self, p):
        #print(p)
        return -0.5*self.__call__(p)    

    def fit(self):

        #-- Initialise iMinuit dictionaty for initial guess of parameters
        #-- to be fitted, excluding those fixed.
        minuit_options = {}
        pars_to_fit_values = []
        pars_to_fit_name = []
        for par in self.parameters:
            if self.parameters[par]['fixed'] == True: 
                continue
            pars_to_fit_name.append(par)
            pars_to_fit_values.append(self.parameters[par]['value'])
            minuit_options['error_'+par] = self.parameters[par]['error']
            minuit_options['limit_'+par] = (self.parameters[par]['limit_low'], 
                                            self.parameters[par]['limit_upp'])

        mig = iminuit.Minuit.from_array_func(self, tuple(pars_to_fit_values),
                                            name = tuple(pars_to_fit_name),
                             print_level=1, errordef=1, throw_nan=False,
                             **minuit_options)
        #print(mig.get_param_states())
        #mig.tol = 0.01
        imin = mig.migrad()
        print(mig.get_param_states())

        best_pars = {}
        for par in self.parameters:
            best_pars[par] = {}
            if self.parameters[par]['fixed']:
                best_pars[par]['value'] = self.parameters[par]['value']
                best_pars[par]['error'] = 0
            else:
                best_pars[par]['value'] = mig.values[par]
                best_pars[par]['error'] = mig.errors[par]

        if self.options['fit_broadband']==True:
            print('\nBroadband terms')
            pars = {par: best_pars[par]['value'] for par in best_pars}    
            model = self.get_model(self.data.r, pars)
            residual = self.data.cf - model
            inv_cov = self.data.icoss
            bb_pars = self.fit_broadband(residual, inv_cov, self.H)
            self.bb_pars = bb_pars
            ibb = np.arange(self.options['bb_max']-self.options['bb_min']+1)
            bb_name = []
            bb_name+= [f'bb_{i}_mono' for i in ibb]
            if self.options['ell_max']>=2:
                bb_name+= [f'bb_{i}_quad' for i in ibb]
            if self.options['ell_max']>=4:
                bb_name+= [f'bb_{i}_hexa' for i in ibb]
            for bb, bbn in zip(bb_pars, bb_name):
                best_pars[bbn] = {'value': bb, 'error': 0}
                print(bbn, bb)

        #mig.hesse()
        print('\nApproximate correlation coefficients:')
        print(mig.matrix(correlation=True))
        self.mig = mig
        #self.imin = imin
        self.is_valid = imin[0]['is_valid']
        self.best_pars = best_pars
        self.chi2min = mig.fval
        self.ndata = self.data.cf.size
        self.npars = len(pars_to_fit_name)
        if self.options['fit_broadband']:
            self.npars += bb_pars.size
        #self.covariance = mig.covariance
        self.rchi2min = self.chi2min/(self.ndata-self.npars)
        print(f'\n chi2/(ndata-npars) = {self.chi2min:.2f}/({self.ndata}-{self.npars}) = {self.rchi2min:.2f}') 

    def get_correlation_coefficient(self, par_par1, par_par2):

        if not hasattr(self, 'covariance'):
            print('Chi2 was not yet minimized')
            return
        
        cov = self.covariance
        var1 = cov[par_par1, par_par1]
        var2 = cov[par_par2, par_par2]
        cov12 = cov[par_par1, par_par2]
        corr_coeff = cov12/np.sqrt(var1*var2)
        return corr_coeff
        

    def plot_bestfit(self, fig=None, model_only=0, scale_r=2, label=None, figsize=(10, 4)):

        nmul = self.options['ell_max']//2+1
        r = self.data.r*1
        cf = self.data.cf*1
        dcf = np.sqrt(np.diag(self.data.coss*1))
        r_model = np.linspace(r.min(), r.max(), 200)
        pars = {par: self.best_pars[par]['value'] for par in self.best_pars}
        cf_model = self.get_model(r_model, pars)
        if hasattr(self, 'bb_pars'):
            bb_model = self.get_broadband(self.bb_pars, r=np.tile(r_model, nmul))
            cf_model += bb_model
            bb=True
        else:
            bb=False

        if fig is None:
            fig, axes = plt.subplots(nrows=1, ncols=nmul, figsize=figsize)
        else:
            axes = fig.get_axes()

        for i in range(nmul):
            try:
                ax = axes[i]
            except:
                ax = axes
            y_data  =  cf[i*r.size:(i+1)*r.size]
            dy_data = dcf[i*r.size:(i+1)*r.size]
            y_model = cf_model[i*r_model.size:(i+1)*r_model.size]
            y_data *= r**scale_r
            dy_data *= r**scale_r
            y_model *= r_model**scale_r 
            if bb:
                b_model = bb_model[i*r_model.size:(i+1)*r_model.size]
                b_model *= r_model**scale_r

            if not model_only:
                ax.errorbar(r, y_data, dy_data, fmt='o', ms=4)
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(r_model, y_model, color=color, label=label)
            #if bb:
            #    ax.plot(r_model, b_model, '--', color=color)

            if scale_r!=0:
                ax.set_ylabel(r'$r^{%d} \xi_{%d}$ [$h^{%d}$ Mpc$^{%d}]$'%\
                              (scale_r, i*2, -scale_r, scale_r))
            else:
                ax.set_ylabel(r'$\xi_{%d}$'%(i*2), fontsize=16)
            ax.set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$', fontsize=12)

        return fig

    def scan1d(self, par_name='alpha', par_min=0.8, par_max=1.2, par_nsteps=400):

        #-- Initialise chi2 grid
        par_grid = np.linspace(par_min, par_max, par_nsteps)
        chi2_grid = np.zeros(par_nsteps)

        for i in range(par_nsteps):
            self.parameters[par_name]['value'] = par_grid[i]
            self.parameters[par_name]['fixed'] = True

            minuit_options = {}
            pars_to_fit_values = []
            pars_to_fit_name = []
            for par in self.parameters:
                if self.parameters[par]['fixed'] == True: 
                    continue
                pars_to_fit_name.append(par)
                pars_to_fit_values.append(self.best_pars[par]['value'])
                minuit_options['error_'+par] = self.best_pars[par]['error']
                minuit_options['limit_'+par] = (self.parameters[par]['limit_low'], 
                                                self.parameters[par]['limit_upp'])

            mig = iminuit.Minuit.from_array_func(self, tuple(pars_to_fit_values),
                                            name = tuple(pars_to_fit_name),
                             print_level=0, errordef=1, throw_nan=False,
                             **minuit_options)
            mig.migrad()
            print( 'scanning: %s = %.5f  chi2 = %.4f'%(par_name, par_grid[i], mig.fval))
            chi2_grid[i] = mig.fval

        return par_grid, chi2_grid

    def scan_2d(self, par_names=['at','ap'], \
                par_min=[0.8, 0.8], \
                par_max=[1.2, 1.2], \
                par_nsteps=[40, 40] ):

        #-- Initialise chi2 grid
        par0 = par_names[0]
        par1 = par_names[1]
        par_grid0 = np.linspace(par_min[0], par_max[0], par_nsteps[0])
        par_grid1 = np.linspace(par_min[1], par_max[1], par_nsteps[1])
        chi2_grid = np.zeros(par_nsteps)

        for i in range(par_nsteps[0]):
            self.parameters[par0]['value'] = par_grid0[i]
            self.parameters[par0]['fixed'] = True
            for j in range(par_nsteps[1]):
                self.parameters[par1]['value'] = par_grid1[j]
                self.parameters[par1]['fixed'] = True

                minuit_options = {}
                pars_to_fit_values = []
                pars_to_fit_name = []
                for par in self.parameters:
                    if self.parameters[par]['fixed'] == True: 
                        continue
                    pars_to_fit_name.append(par)
                    pars_to_fit_values.append(self.best_pars[par]['value'])
                    minuit_options['error_'+par] = self.best_pars[par]['error']
                    minuit_options['limit_'+par] = (self.parameters[par]['limit_low'], 
                                                    self.parameters[par]['limit_upp'])

                mig = iminuit.Minuit.from_array_func(self, tuple(pars_to_fit_values),
                                                name = tuple(pars_to_fit_name),
                                print_level=0, errordef=1, throw_nan=False,
                                **minuit_options)
                mig.migrad()
                print( 'scanning: %s = %.5f   %s = %.5f    chi2 = %.4f'%\
                        (par0, par_grid0[i], par0, par_grid1[j], mig.fval))
                chi2_grid[i, j] = mig.fval

        return par_grid0, par_grid1, chi2_grid

    def export_bestfit_parameters(self, fout):

        fout = open(fout, 'w')
        print(f'chi2  {self.chi2min}', file=fout)
        print(f'ndata {self.ndata}', file=fout)
        print(f'npars {self.npars}', file=fout)
        print(f'rchi2 {self.rchi2min}', file=fout)
       
        for p in self.best_pars:
            print(p,          self.best_pars[p]['value'], file=fout)
            print(p+'_error', self.best_pars[p]['error'], file=fout)

        fout.close()

    def export_covariance(self, fout):

        fout = open(fout, 'w')
        print('# par_par1 par_par2 covariance corr_coeff', file=fout)
        cov = self.covariance
        for k in cov:
            corr = cov[k]/np.sqrt(cov[(k[0], k[0])]*cov[(k[1], k[1])])
            print(f'{k[0]} {k[1]} {cov[k]} {corr}', file=fout)
        fout.close()  

    def export_model(self, fout):

        nmul = self.options['ell_max']//2+1
        nr = 200
        r_model = np.linspace(self.data.r.min(), self.data.r.max(), nr)
        pars = {par: self.best_pars[par]['value'] for par in self.best_pars}  
        cf_model = self.get_model(r_model, pars)

        if hasattr(self, 'bb_pars'):
            bb_model = self.get_broadband(self.bb_pars, r=np.tile(r_model, nmul))
            cf_model += bb_model
            bb=True
        else:
            bb=False

        cf_model = cf_model.reshape((nmul, nr)) 
        if bb:
            bb_model = bb_model.reshape((nmul, nr)) 
        
        fout = open(fout, 'w')
        line = '#r mono '
        line += 'quad '*(self.options['ell_max']>=2)
        line += 'hexa '*(self.options['ell_max']>=4)
        if bb:
            line += 'bb_mono '
            line += 'bb_quad '*(self.options['ell_max']>=2)
            line += 'bb_hexa '*(self.options['ell_max']>=4)
        print(line, file=fout)

        for i in range(nr):
            line = f'{r_model[i]}  '
            for l in range(nmul):
                line += f'{cf_model[l, i]}  ' 
            if bb:
                for l in range(nmul):
                    line += f'{bb_model[l, i]}  ' 
            print(line, file=fout)
        fout.close()


