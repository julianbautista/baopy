import matplotlib.pyplot as plt
import baopy.fitter
import baopy.models 

#-- Read data files and covariance
dat = baopy.fitter.Data(data_file='PowerSpectra_data_NGCSGC_postrecon.ecsv',
        cova_file='ezmock_LRGpCMASS_NGCSGC_recon_covariance.ecsv')

#-- Select multipoles and scale ranges from Gil-Marín et al. 2020
space, ell, scale = dat.coords['space'], dat.coords['ell'], dat.coords['scale']
cuts = (space == 0 ) & ((ell==0) | (ell==2)) & ((scale > 0.02) & (scale < 0.3))
#cuts |= (space == 1) & ((ell==0) | (ell==2)) & ((scale > 50) & (scale < 150))
dat.apply_cuts(cuts)

#-- Invert covariance matrix after cuts and apply correction factors 
dat.inverse_cova(nmocks=1000)

#-- Read linear power spectrum from file
mod = baopy.models.BAO('pk_camb_z0.698_challenge.txt')
mod.read_window_function('Window_NGCSGC_public.txt')

#-- Setup parameters, which ones are fixed, limits
parameters = {'alpha_para':{'value':1.,     'error':0.1,  'limit_low': 0.5, 'limit_upp': 1.5, 'fixed': False}, 
              'alpha_perp':{'value':1.,     'error':0.1,  'limit_low': 0.5, 'limit_upp': 1.5, 'fixed': False},
              'bias'      :{'value':2.3,    'error': 0.1, 'limit_low': 1,   'limit_upp': 4.,  'fixed': False},
              'beta'      :{'value':0.35,   'fixed':True}, 
              'sigma_rec' :{'value':15.,    'fixed':True}, 
              'sigma_para':{'value':7.0,   'fixed':True}, #-- Value used in Gil-Marín et al. 2020
              'sigma_perp':{'value':2.0,    'fixed':True}, #-- Value used in Gil-Marín et al. 2020
              'sigma_fog' :{'value':0.,     'fixed':True} 
              }

#-- Some extra options, e.g., broadband
options = {'fit_broadband': True, 
           'bb_min': -2, # Gil-Marin et al. uses -1
           'bb_max': 1, # Gil-Marin et al. uses 1
           }

#-- Initialise the fitter
print('Computing best-fit parameters...')
chi = baopy.fitter.Chi2(data=dat, model=mod, parameters=parameters, options=options)
chi.fit()
chi.print_chi2()

#-- Compute precise errors for the alphas
print('Computing precise error bars...')
chi.minos('alpha_para')
chi.minos('alpha_perp')

#-- Print the errors
chi.print_minos('alpha_perp', symmetrise=False, decimals=3)
chi.print_minos('alpha_para', symmetrise=False, decimals=3)

#-- Plot best-fit model and save it 
chi.plot(power_k=1)
plt.savefig('results_fit_power_spectrum.pdf')

#-- Get contours of 1 and 2 sigma
print('Computing 2D contours...')
chi.get_contours('alpha_perp', 'alpha_para', confidence_level=0.68, n_points=30)
chi.get_contours('alpha_perp', 'alpha_para', confidence_level=0.95, n_points=30)

#-- Plot contours
chi.plot_contours('alpha_perp', 'alpha_para')
plt.savefig('results_fit_power_spectrum_contours.pdf')

#-- Save results to file
chi.save('results_fit_power_spectrum.pkl')

