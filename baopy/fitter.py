''' Module containing the main fitter class and a data handler '''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.table import Table
import iminuit

class Chi2: 
    ''' Fitter class that minimises the $\chi^2$ 
    
        It connects a Data object and a Model object,
        minimisation used iMinuit

    '''

    def __init__(self, 
        data=None, 
        model=None, 
        parameters=None, 
        options=None):
            
        self.data = data
        self.model = model
        self.parameters = parameters
        self.options = options
        
        self.setup_iminuit()
        self.setup_broadband()

        self.best_pars_value = None
        self.best_pars_error = None
        self.best_model = None
        self.best_bb_pars = None 
        self.best_broadband = None
        self.chi2min = None
        self.npar = None
        self.ndof = None
        self.rchi2min = None
        
        #-- Names of fields to be saved 
        self.chi_fields = [ 'parameters', 
                            'options', 
                            'best_pars_value', 
                            'best_pars_error', 
                            'best_model', 
                            'best_bb_pars', 
                            'best_broadband',
                            'ndata', 
                            'chi2min', 
                            'npar', 
                            'ndof', 
                            'rchi2min',
                            'contours']
        #-- Minos fields to be saved 
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
            mig.errors[par] = 0 if 'fixed' in par_dict and par_dict['fixed'] else mig.errors[par]
            mig.fixed[par] = par_dict['fixed'] if 'fixed' in par_dict else False
            limit_low = par_dict['limit_low'] if 'limit_low' in par_dict else None
            limit_upp = par_dict['limit_upp'] if 'limit_upp' in par_dict else None
            mig.limits[par] = (limit_low, limit_upp)
            
        self.mig = mig 

    def get_model(self, pars):
        coords = self.data.coords
        model_values = self.model.get_multipoles(coords['space'], 
                                                coords['ell'], 
                                                coords['scale'], 
                                                pars)
        return model_values
    
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
        ''' Compute chi2 for a set of free parameters 
            (and only the free parameters!)

            Input
            -----
            p: list of parameter values

            Returns
            -------
            chi2: float, value of chi2 for these parameters
        '''

        #-- Create dictionary with parameter names, which is giveb to Model class
        pars = {}
        i = 0
        for i, par in enumerate(self.parameters):
            pars[par] = p[i]
            i+=1

        #-- Compute model
        model_values = self.get_model(pars)
        
        #-- Compute residuals    
        values = self.data.values
        residual = values - model_values

        #-- Add broadband function
        if not self.h_matrix is None:
            bb_pars = self.fit_broadband(residual)
            broadband = self.get_broadband(bb_pars)
            residual -= broadband

        #-- Compute chi2
        inv_cova = self.data.inv_cova
        chi2 = np.dot(residual, np.dot(inv_cova, residual))

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
        best_pars_value = {k: mig.params[k].value for k in mig.parameters}
        best_pars_error = {k: mig.params[k].error for k in mig.parameters}
        best_model = self.get_model(best_pars_value)

        #-- Add broadband 
        if not self.h_matrix is None:
            best_bb_pars = self.fit_broadband(self.data.values - best_model)
            best_broadband = self.get_broadband(best_bb_pars)
            n_bb_pars = best_bb_pars.size
        else:
            best_broadband = best_model*0
            best_bb_pars = None 
            n_bb_pars = 0

        self.best_pars_value = best_pars_value
        self.best_pars_error = best_pars_error
        self.best_model = best_model + best_broadband
        self.best_bb_pars = best_bb_pars 
        self.best_broadband = best_broadband

        self.mig = mig
        self.ndata = best_model.size 
        self.chi2min = mig.fval
        self.npar = mig.nfit + n_bb_pars
        self.ndof = self.ndata - self.npar
        self.rchi2min = self.chi2min/(self.ndof)
    
    def print(self):
        print(self.mig)

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

    def plot(self, f=None, axs=None, power_k=1, power_r=2, figsize=(6, 6)):
        ''' Plots the data and its best-fit model 
        '''
        if self.best_model is None:
            print('No best-fit model found. Please run Chi2.fit()')
            return
        if self.data is None:
            print('No data found. Please read a data and covariance file.')
            return 

        f, axs = self.data.plot(f=f, axs=axs, 
                                power_k=power_k, power_r=power_r, 
                                y_model=self.best_model,
                                figsize=figsize)
        return f, axs

    def save(self, filename):
        if self.output is None:
            self.get_output_from_minuit()
        pickle.dump(self.output, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        output = pickle.load(open(filename, 'rb'))
        chi = Chi2()
        for field in chi.chi_fields:
            if field in output:
                chi.__setattr__(field, output[field])
        
        chi.output = output
        return chi




class Sampler:
    '''First attempt to have a Monte Carlo Markov Chain sampler
    '''

    def __init__(self, chi, 
        sampler_name='emcee', 
        nsteps=1000,
        nsteps_burn=100, 
        nwalkers=10, 
        use_pool=False,
        seed=0):
        ''' Initialises the sampler from a Chi object

        Parameters
        ----------
        chi : Chi
            Chi object instance
        sampler_name : str
            Two options ``emcee`` or ``zeus``
        nsteps : int
            Number of steps per chain
        nsteps_burn : int 
            Number of steps to be considered as burn-in
        nwalkers: int
            Number of parallel chains
        use_pool : bool
            Use python multithreading
        seed : int 
            Seed for random sampler 

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

        parameters = chi.parameters
        fixed = []
        for par in parameters:
            fixed.append('fixed' in parameters[par] and parameters[par]['fixed'] == True)

        npars = len(parameters)
        npars_free = npars-np.sum(fixed)
        pars = np.zeros(npars)

        #-- Use limits to set the start point for random walkers
        print('Setting up starting point of walkers:')
        np.random.seed(seed)
        start = np.zeros((nwalkers, npars_free))
        limits = np.zeros((npars_free, 2))
        i=0
        for j, par in enumerate(parameters):
            pars[j] = parameters[par]['value']
            if fixed[j]:
                continue
            limit_low = parameters[par]['limit_low']
            limit_upp = parameters[par]['limit_upp']
            value = chi.best_pars_value[par]
            error = chi.best_pars_error[par]
            #print(f' "{par}" between ', limit_low, limit_upp)
            #start[:, i] = np.random.rand(nwalkers)*(limit_upp-limit_low)+limit_low
            print(f' "{par}" with mean ', value, 'and dispersion', 3*error)
            start[:, i] = np.random.randn(nwalkers)*3*error+value
            limits[i] = [limit_low, limit_upp]
            i+=1 
            
        self.parameters = parameters
        self.fixed = fixed 
        self.limits = limits
        self.npars = npars 
        self.npars_free = npars_free 
        self.pars = pars
        self.use_pool = use_pool 
        self.nwalkers = nwalkers
        self.start = start
        self.nsteps = nsteps 
        self.nsteps_burn = nsteps_burn
        self.sampler_name = sampler_name
        self.sampling_function = sampling_function
        self.chi = chi

    def run(self):

        fixed = self.fixed
        limits = self.limits  
        npars = self.npars 
        pars = self.pars
        chi = self.chi 

        global log_prob
        def log_prob(p_free):
            #print(p_free)
            p_all = []
            j=0
            for i in range(npars):
                if fixed[i]==False:
                    if p_free[j] < limits[j, 0] or p_free[j] > limits[j, 1]:
                        return -np.inf
                    p_all.append(p_free[j])
                    j+=1
                else:
                    p_all.append(pars[i])
            #print(p_all)
            return chi.log_prob(p_all)

        if self.use_pool == True:
            from multiprocessing import Pool, cpu_count
            print(f'Using multiprocessing with {cpu_count()} cores')
            with Pool() as pool:
                sampler = self.sampling_function(self.nwalkers, self.npars_free, log_prob, pool=pool)
                state = sampler.run_mcmc(self.start, self.nsteps_burn, progress=True)
                sampler.reset()
                sampler.run_mcmc(state, self.nsteps, progress=True)

        else:
            print('Using a single core')
            sampler = self.sampling_function(self.nwalkers, self.npars_free, log_prob)
            state = sampler.run_mcmc(self.start, self.nsteps_burn, progress=True)
            sampler.reset()
            sampler.run_mcmc(state, self.nsteps, progress=True)
            
        #if sampler_name == 'zeus':
        #    sampler.summary

        chain = sampler.get_chain(flat=True)
        self.sampler = sampler
        self.chain = chain

    def save(self, filename):
        chain = self.chain
        parameters = self.parameters
        
        tab=Table({})
        for i, par in enumerate(parameters):
            tab[par]=chain[:,i] 
            
        tab.write(filename,overwrite=True)
