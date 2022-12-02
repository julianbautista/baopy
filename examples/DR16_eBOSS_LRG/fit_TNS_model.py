import baopy.fitter
import baopy.models
import baopy.data

#-- Read data files and covariance in ecsv format
dat = baopy.data.Data(
        data_file='Correlations_data_NGCSGC_prerecon.ecsv',
        cova_file='ezmock_LRGpCMASS_NGCSGC_covariance.ecsv')

#-- Select multipoles and scale ranges (space==0 (FS) and space==1 (CS))
space, ell, scale = dat.coords['space'], dat.coords['ell'], dat.coords['scale']
cuts = (space == 1 ) & ((ell==0) | (ell==2)| (ell==4)) & ((scale >= min) & (scale <= max))

#-- Invert covariance matrix after cuts and apply correction factors 
#-- nmocks = number of mocks used to compute cov matrix
dat.apply_cuts(cuts)
dat.inverse_cova(nmocks=1000)

#-- Read power spectra, 1-loop bias terms and RSD correction terms
#-- Those terms are computed with pyRegPT : https://github.com/adematti/pyregpt
mod = baopy.models.RSD_TNS(
        pk_regpt_file= 'regpt/pk_camb_z0.698_challenge_pk2loop.txt',
        bias_file    = 'regpt/pk_camb_z0.698_challenge_bias1loop.txt',
        a2loop_file  = 'regpt/pk_camb_z0.698_challenge_A2loop.txt',
        b2loop_file  = 'regpt/pk_camb_z0.698_challenge_B2loop.txt')

#-- Read Window Function Multipoles, if no WF is needed don't use this function 
mod.read_window_function(dir_dat+'Window_NGCSGC_public.txt')

#-- Setup parameters, which ones are fixed, limits
#-- if space ==1 : shot_noise should be fixed to 0
parameters= {
        'apar'       :{'value':1.,   'error': 0.1,  'limit_low': 0.5,     'limit_upp': 1.5,    'fixed': False}, 
        'aper'       :{'value':1.,   'error': 0.1,  'limit_low': 0.5,     'limit_upp': 1.5,    'fixed': False},
        'b1'         :{'value':1.,   'error': 0.1,  'limit_low': 0.,      'limit_upp': 10.,    'fixed': False},
        'b2'         :{'value':1,    'error': 0.1,  'limit_low': -10.,    'limit_upp': 10.,    'fixed': False},
        'f'          :{'value':0.5,  'error': 0.1,  'limit_low': 0.,      'limit_upp': 3.,     'fixed': False},
        'shot_noise' :{'value':0.,   'error': 0.1,  'limit_low': -5000.,  'limit_upp': 5000.,  'fixed': False},
        'sigma_fog'  :{'value':1.,   'error': 0.1,  'limit_low': 0.,      'limit_upp': 20.,    'fixed': False},
        }

#-- Some extra options, e.g., broadband
options = {'fit_broadband': False, 
        'bb_min': -1, # Gil-Marin et al. uses -1
        'bb_max': 1, # Gil-Marin et al. uses 1
        }

#-- Initialise the fitter
chi = baopy.fitter.Chi2(data=dat, model=mod, parameters=parameters)
chi.fit()
chi.print_chi2()
print(chi.mig)

#-- Compute precise errors for some parameters
chi.minos('apar')
chi.minos('aper')
chi.minos('f')
chi.minos('b1')
chi.minos('b2')
#-- Print the errors
chi.print_minos('apar', symmetrise=False, decimals=3)
chi.print_minos('aper', symmetrise=False, decimals=3)
chi.print_minos('f', symmetrise=False, decimals=3)
chi.print_minos('b1', symmetrise=False, decimals=3)
chi.print_minos('b2', symmetrise=False, decimals=3)

#-- Save results to file
chi.save(results_file)
#-- Plot best-fit model and save it 
chi.plot()
plt.savefig(plot_file)

#-- run mcmc with iminuit best-fit as starting point
mcmc=baopy.fitter.Sampler(chi, 
                        sampler_name='emcee', 
                        nsteps=5000,
                        nsteps_burn=500, 
                        nwalkers=50, 
                        use_pool=True,
                        seed=0)
mcmc.run()

#-- save mcmc chains 
mcmc.save('chain_mcmc.ecsv')