import matplotlib.pyplot as plt
import baopy.bao_fitter

#-- Read data files and covariance
dat = baopy.bao_fitter.Data(data_file='PowerSpectra_data_NGCSGC_postrecon.ecsv',
        cova_file='ezmock_LRGpCMASS_NGCSGC_recon_covariance.ecsv')

#-- Select multipoles and scale ranges from Gil-Marín et al. 2020
space, ell, scale = dat.coords['space'], dat.coords['ell'], dat.coords['scale']
cuts = (space == 0 ) & ((ell==0) | (ell==2)) & ((scale > 0.02) & (scale < 0.3))
#cuts |= (space == 1) & ((ell==0) | (ell==2)) & ((scale > 50) & (scale < 150))
dat.apply_cuts(cuts)

#-- Invert covariance matrix after cuts and apply correction factors 
dat.inverse_cova(nmocks=1000)

#-- Read linear power spectrum from file
mod = baopy.bao_fitter.Model('pk_camb_z0.698_challenge.txt')
mod.read_window_function('Window_NGCSGC_public.txt')

#-- Setup parameters, which ones are fixed, limits
parameters = {'alpha_para':{'value':1.,     'error':0.1,  'limit_low': 0.5, 'limit_upp': 1.5, 'fixed': False}, 
              'alpha_perp':{'value':1.,     'error':0.1,  'limit_low': 0.5, 'limit_upp': 1.5, 'fixed': False},
              'bias'      :{'value':2.3,    'error': 0.1, 'limit_low': 1,   'limit_upp': 4.,  'fixed': False},
              'beta'      :{'value':0.35,   'fixed':False}, 
              'sigma_rec' :{'value':15.,    'fixed':True}, 
              'sigma_para':{'value':7.0 ,   'fixed':True}, #-- Value used in Gil-Marín et al. 2020
              'sigma_perp':{'value':2.0,    'fixed':True}, #-- Value used in Gil-Marín et al. 2020
              'sigma_s'   :{'value':0.,     'fixed':True} 
              }

#-- Some extra options, e.g., broadband
options = {'fit_broadband': True, 
           'bb_min': -1, # Gil-Marin et al. uses -1
           'bb_max': 1, # Gil-Marin et al. uses 1
           }

#-- Initialise the fitter
chi = baopy.bao_fitter.Chi2(data=dat, model=mod, parameters=parameters, options=options)
chi.fit()
chi.print_chi2()

#-- Compute precise errors for the alphas
chi.minos('alpha_para')
chi.minos('alpha_perp')

#-- Print the errors
chi.print_minos('alpha_perp', symmetrise=False, decimals=3)
chi.print_minos('alpha_para', symmetrise=False, decimals=3)

#-- Plot best-fit model and save it 
chi.plot(scale_r=1)

#-- Save results to file
chi.save('results_fit_power_spectrum.pkl')

