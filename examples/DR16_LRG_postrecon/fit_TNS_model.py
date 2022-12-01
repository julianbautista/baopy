from astropy.table import Table 
import baopy.bao_fitter
import baopy.fitter
import baopy.models
import baopy.data

#-- Read data files and covariance in ecsv format
dat = baopy.data.Data(
        data_file='data.ecsv',
        cova_file='cov.ecsv')

#-- Select multipoles and scale ranges (space==0 (FS) and space==1 (CS))
space, ell, scale = dat.coords['space'], dat.coords['ell'], dat.coords['scale']
cuts = (space == 0 ) & ((ell==0) | (ell==2)| (ell==4)) & ((scale >= min) & (scale <= max))

#-- Invert covariance matrix after cuts and apply correction factors 
dat.apply_cuts(cuts)
dat.match_cova()
dat.inverse_cova(nmocks=nmocks)

#-- Read power spectra, 1-loop bias terms and RSD correction terms
mod = baopy.models.RSD_TNS(
        pk_regpt_file= 'Pk2loop.txt',
        bias_file    = 'Bias1loop.txt',
        a2loop_file  = 'A2loop.txt',
        b2loop_file  = 'B2loop.txt')

#-- Setup parameters, which ones are fixed, limits
#-- if space ==1 : shot_noise should be fixed to 0
parameters= {
        'alpha_para' :{'value':1.,   'error': 0.1,  'limit_low': 0.5,     'limit_upp': 1.5,    'fixed': False}, 
        'alpha_perp' :{'value':1.,   'error': 0.1,  'limit_low': 0.5,     'limit_upp': 1.5,    'fixed': False},
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
chi.minos('alpha_para')
chi.minos('alpha_perp')
chi.minos('f')
chi.minos('b1')
chi.minos('b2')
#-- Print the errors
chi.print_minos('alpha_para', symmetrise=False, decimals=3)
chi.print_minos('alpha_perp', symmetrise=False, decimals=3)
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

#-- create a table and save mcmc chains
tab = Table({
'apar':mcmc.chain[:,0],
'aper':mcmc.chain[:,1],
'b1':mcmc.chain[:,2],
'b2':mcmc.chain[:,3],
'f':mcmc.chain[:,4], 
'Ng':mcmc.chain[:,5], 
'sigma_fog':mcmc.chain[:,6]})
tab.write('chain_mcmc.ecsv',overwrite=True)
