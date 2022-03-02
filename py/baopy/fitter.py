import numpy as np
import pylab as plt
import pickle
import iminuit
import scipy.linalg
from scipy.optimize import curve_fit

plt.ion()

class Data: 

    def __init__(self, data_file=None, cova_file=None):
        try:
            self.read_numpy(data_file, cova_file)
        except:
            self.read_astropy(data_file, cova_file)

    def read_numpy(self, data_file, cova_file):
        space, ell, scale, y_value = np.loadtxt(data_file, unpack=1, skiprows=1)
        coords = {'space': space.astype(int), 'ell': ell.astype(int), 'scale': scale}
        
        #-- Make sure the covariance and data vector match
        #-- First, create a dictionary 
        s1, l1, x1, s2, l2, x2, cova_12 = np.loadtxt(cova_file, unpack=1, skiprows=1)
        
        self.coords = coords
        self.y_value = y_value
        self.cova = self.match_cova(coords, s1, l1, x1, s2, l2, x2, cova_12)
    
    def read_astropy(self, data_file, cova_file):
        
        from astropy.table import Table 
        data = Table.read(data_file)
        cova = Table.read(cova_file)
        
        coords = {'space': data['space'].data, 
                  'ell':   data['ell'].data,
                  'scale': data['scale'].data}
        y_value = data['correlation'].data 
        cova_12 = cova['covariance'].data
        cova_match = self.match_cova(coords, 
                                     cova['space_1'].data, cova['ell_1'].data, cova['scale_1'].data,
                                     cova['space_2'].data, cova['ell_2'].data, cova['scale_2'].data,
                                     cova_12)

        self.coords = coords
        self.y_value = y_value
        self.cova = cova_match
    
    def match_cova(self, coords, s1, l1, x1, s2, l2, x2, cova_12):
        
        cova_dict = {}
        for i in range(cova_12.size):
            cova_dict[s1[i], l1[i], x1[i], s2[i], l2[i], x2[i]] = cova_12[i]

        #-- Second, fill covariance matrix with only data vector elements, in the same order
        nbins = coords['space'].size
        cova_match = np.zeros((nbins, nbins))
        for i in range(nbins):
            s1 = coords['space'][i]
            l1 = coords['ell'][i]
            x1 = coords['scale'][i]
            for j in range(nbins):
                s2 = coords['space'][j]
                l2 = coords['ell'][j]
                x2 = coords['scale'][j]
                cova_match[i, j] = cova_dict[s1, l1, x1, s2, l2, x2]

        return cova_match


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
        if scale_r == 0:
            titles = [r'$k^2 P_\ell(k)$', r'$r^2 \xi_\ell(k)$']
        else:
            titles = [fr'$k^{scale_r} P_\ell(k)$', fr'$r^{scale_r} \xi_\ell(k)$']

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
        
        self.setup_iminuit()
        self.setup_broadband()

        self.best_pars = None
        self.best_model = None
        self.best_bb_pars = None 
        self.best_broadband = None
        self.chi2min = None
        self.npar = None
        self.ndof = None
        self.rchi2min = None
        
        #-- Names of fields to be saved 
        self.chi_fields = ['parameters', 'options', 
                           'best_pars', 'best_model', 'best_bb_pars', 'best_broadband',
                           'ndata', 'chi2min', 'npar', 'ndof', 'rchi2min',
                           'contours']
        self.param_fields = ['number', 'value', 'error', 'merror', 
                             'lower_limit', 'upper_limit', 'is_fixed']
        self.output = None

    def setup_iminuit(self):

        parameters = self.parameters 
        if parameters is None:
            self.mig = None 
            return 

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
    
    def setup_broadband(self):
        ''' Setup analytical solution for best-fit polynomial nuisance terms
	        http://pdg.lbl.gov/2016/reviews/rpp2016-rev-statistics.pdf eq. 39.22 and surrounding
        '''
        options = self.options
        if not options is None and 'fit_broadband' in options and options['fit_broadband'] == True:
            bb_min = options['bb_min'] if 'bb_min' in options else -2
            bb_max = options['bb_max'] if 'bb_max' in options else 0
        else:
            self.h_matrix = None
            self.norm_matrix = None
            return 

        coords = self.data.coords
        space = coords['space']
        ell = coords['ell']
        scale = coords['scale']

        #-- Ensemble of unique combinations of pairs of (space, ell)
        upairs = np.unique([space, ell], axis=1)
        n_upairs = upairs.shape[1]
        #-- The broadband is a series of powers of scale, these are the exponents
        power = np.arange(bb_min, bb_max+1) 
        #-- Number of broadband parameters
        n_bb_pars = n_upairs * power.size
        #-- Setup matrix, looping over all data coordinates
        h_matrix = np.zeros((scale.size, n_bb_pars))
        for row in range(scale.size):
            #-- For this (space, ell), determine index in the unique ensemble
            col_space_ell = np.where((upairs[0]== space[row]) & (upairs[1]==ell[row]))[0][0]
            #-- Fill the corresponding elements in the h_matrix with scale**power
            for j in range(power.size):
                col = col_space_ell*power.size + j
                h_matrix[row, col] = scale[row]**power[j]

        #-- Compute normalisation of linear solver
        norm_matrix = np.linalg.inv(h_matrix.T @ self.data.inv_cova @ h_matrix)

        self.h_matrix = h_matrix
        self.norm_matrix = norm_matrix

    def fit_broadband(self, residual):
        ''' Solves for broadband parameters using a linear solver
            and the matrices setup in setup_broadband()
        '''
        norm_matrix = self.norm_matrix
        h_matrix = self.h_matrix 
        inv_cova = self.data.inv_cova 

        #-- Eq. 39.22 of the reference
        bb_pars = norm_matrix.dot(h_matrix.T.dot(inv_cova.dot(residual)))

        return bb_pars

    def get_broadband(self, bb_pars):
        ''' Computes the broadband function from a set of parameters
        '''
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
        if not self.h_matrix is None:
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
        if not self.h_matrix is None:
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
        self.ndata = best_model.size 
        self.chi2min = mig.fval
        self.npar = mig.nfit + n_bb_pars
        self.ndof = self.ndata - self.npar
        self.rchi2min = self.chi2min/(self.ndof)
    
    def print_chi2(self):
        print(f'chi2/(ndata-npars) = {self.chi2min:.2f}/({self.ndata}-{self.npar}) = {self.rchi2min:.2f}') 

    def minos(self, parameter_name):
        self.mig.minos(parameter_name)

    def print_minos(self, parameter_name, symmetrise=False, decimals=None):
        if self.output is None:
            self.get_output_from_minuit()

        par_details = self.output['best_pars_details'][parameter_name]
        value = par_details['value']
        error_low, error_upp = par_details['merror']
        error = (error_upp - error_low)/2

        if not decimals is None:
            value = f'{value:.{decimals}f}'
            error_low = f'{-error_low:.{decimals}f}'
            error_upp = f'{error_upp:.{decimals}f}'
            error = f'{error:.{decimals}f}'

        if symmetrise:     
            print(f'{parameter_name}: {value} +/- {error}')
        else:
            print(f'{parameter_name}: {value} + {error_upp} - {error_low}')

    def get_contours(self, parameter_name_1, parameter_name_2, confidence_level=0.685, n_points=30):
        if self.output is None:
            self.get_output_from_minuit()
        output = self.output 

        contour_xy = self.mig.mncontour(parameter_name_1, parameter_name_2, 
                                    cl=confidence_level, size=n_points)
        
        if not 'contours' in output:
            output['contours'] = {}
        key = (parameter_name_1, parameter_name_2)
        if not key in output['contours']:
            output['contours'][key] = {}
        output['contours'][key][confidence_level] = contour_xy

    def plot_contours(self, parameter_name_1, parameter_name_2):

        if self.output is None or not 'contours' in self.output:
            print('Error: Need to compute contours first.')
            return

        contours = self.output['contours'][parameter_name_1, parameter_name_2]
        confidence_levels = np.sort(list(contours.keys()))

        plt.figure()      
        for confidence_level in confidence_levels[::-1]:
            contour = contours[confidence_level]
            plt.fill(contour[:, 0], contour[:, 1], alpha=0.3, color='C0', label=f'{confidence_level}')
        plt.xlabel(parameter_name_1)
        plt.ylabel(parameter_name_2)
        #plt.legend()

    def get_output_from_minuit(self):
        ''' Converts outputs from iMinuit to a simple dictionary,
            which is used to save the results. 
        '''
        output = {}
        
        for field in self.chi_fields:
            try:
                output[field] = self.__getattribute__(field)
            except:
                pass
        details = {}
        for parameter in self.mig.params:
            details[parameter.name] = {}
            for field in self.param_fields:
                details[parameter.name][field] = parameter.__getattribute__(field)

        output['best_pars_details'] = details
        self.output = output

    def plot(self, f=None, axs=None, scale_r=2, label=None, figsize=(10, 4)):
        ''' Plots the data and its best-fit model 
        '''
        if self.best_model is None:
            print('No best-fit model found. Please run Chi2.fit()')
            return
        if self.data is None:
            print('No data found. Please read a data and covariance file.')
            return 

        f, axs = self.data.plot(f=f, axs=axs, scale_r=scale_r, 
                                y_model=self.best_model,
                                figsize=figsize, label=label)
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

    def save(self, filename):
        if self.output is None:
            self.get_output_from_minuit()
        pickle.dump(self.output, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        output = pickle.load(open(filename, 'rb'))
        chi = Chi2()
        #-- fill chi with output 
        #-- todo
        for field in chi.chi_fields:
            chi.__setattr__(field, output[field])
        
        chi.output = output
        return chi