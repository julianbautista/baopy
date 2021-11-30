import matplotlib.pyplot as plt
import baopy.bao_fitter 

#-- Read data files and covariance
dat = baopy.bao_fitter.Data(data_file='Correlations_data_NGCSGC_postrecon.ecsv',
        cova_file='ezmock_LRGpCMASS_NGCSGC_recon_covariance.ecsv')

#-- Select multipoles and scale ranges
space, ell, scale = dat.coords['space'], dat.coords['ell'], dat.coords['scale']
cuts = (space == 0 ) & ((ell==0) | (ell==2)) & ((scale > 0.02) & (scale < 0.3))
cuts |= (space == 1) & ((ell==0) | (ell==2)) & ((scale > 50) & (scale < 150))
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
              'beta':      {'value':0.35,   'fixed':True}, 
              'sigma_rec' :{'value':15.,    'fixed':True}, 
              'sigma_para':{'value':7. ,  'fixed':True}, #-- Value used in Gil-Marín et al. 2020
              'sigma_perp':{'value':2.,   'fixed':True}, #-- Value used in Gil-Marín et al. 2020
              #'sigma_para':{'value':7.31 ,  'fixed':True}, #-- Value used Bautista et al. 2020
              #'sigma_perp':{'value':5.53,   'fixed':True}, #-- Value used in Bautista et al. 2020
              'sigma_s'   :{'value':0.,     'fixed':True} 
              }

#-- Some extra options, e.g., broadband
#-- Bautista  et al. uses [-2, -1, 0]
#-- Gil-Marin et al. uses [-1,  0, 1]
options = {'fit_broadband': True, 
           'bb_min': -2, 
           'bb_max': 1}

#-- Initialise the fitter
chi = baopy.bao_fitter.Chi2(data=dat, model=mod, parameters=parameters, options=options)
chi.fit()
chi.print_chi2()

#-- Compute precise errors for the alphas
chi.minos('alpha_perp')
chi.minos('alpha_para')

#-- Print the errors
chi.print_minos('alpha_perp', decimals=5)
chi.print_minos('alpha_para', decimals=5)

#-- Plot best-fit model and save it 
chi.plot()

#-- Save results to file
chi.save('results_fit_joint.pkl')