import matplotlib.pyplot as plt
import baopy.bao_fitter

#-- Read data files and covariance
dat = baopy.bao_fitter.Data(data_file='PowerSpectra_data_NGC_postrecon_slx.txt',
         cova_file='Covariance_NGC_postrecon_slx.txt')

#-- Select multipoles and scale ranges from Gil-Marín et al. 2020
cuts = (((dat.coords['ell'] == 0) & (dat.coords['scale'] > 0.02) & (dat.coords['scale'] < 0.3)) | 
        ((dat.coords['ell'] == 2) & (dat.coords['scale'] > 0.02) & (dat.coords['scale'] < 0.3)) )
        #| ((dat.coords['ell'] == 4) & (dat.coords['scale'] > 50) & (dat.coords['scale']<150)))
dat.apply_cuts(cuts)
dat.inverse_cova(nmocks=1000)

#-- Read linear power spectrum from file
mod = baopy.bao_fitter.Model('pk_camb_z0.698_challenge.txt')
mod.read_window_function('Window_NGC_public.txt')

#-- Setup parameters, which ones are fixed, limits
parameters = {'alpha_para':{'value':1.,     'error':0.1,  'limit_low': 0.5, 'limit_upp': 1.5, 'fixed': False}, 
              'alpha_perp':{'value':1.,     'error':0.1,  'limit_low': 0.5, 'limit_upp': 1.5, 'fixed': False},
              'bias'      :{'value':2.3,    'error': 0.1, 'limit_low': 1,   'limit_upp': 4.,  'fixed': False},
              'beta':      {'value':0.35,   'fixed':False}, 
              'sigma_rec' :{'value':15.,    'fixed':True}, 
              'sigma_para':{'value':7.0 ,  'fixed':True}, 
              'sigma_perp':{'value':2.0,   'fixed':True}, 
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
#plt.savefig('DR16_LRGpk_postrecon_NGC_bestfitmodel_window.pdf')

chi.save('chi_pk.pkl')

