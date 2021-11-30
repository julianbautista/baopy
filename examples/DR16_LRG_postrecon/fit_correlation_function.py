import matplotlib.pyplot as plt
import baopy.bao_fitter

#-- Read data files and covariance
dat = baopy.bao_fitter.Data(data_file='CorrelationFunction_data_COMB_postrecon.ecsv',
        cova_file='ezmock_LRGpCMASS_NGCSGC_recon_covariance.ecsv')

#-- Select multipoles and scale ranges
space, ell, scale = dat.coords['space'], dat.coords['ell'], dat.coords['scale']
#cuts = (space == 0 ) & ((ell==0) | (ell==2)) & ((scale > 0.002) & (scale < 0.3))
cuts = (space == 1) & ((ell==0) | (ell==2)) & ((scale > 50) & (scale < 150))
dat.apply_cuts(cuts)

#-- Invert covariance matrix after cuts and apply correction factors 
dat.inverse_cova(nmocks=1000)

#-- Read linear power spectrum from file
mod = baopy.bao_fitter.Model('pk_camb_z0.698_challenge.txt')

#-- Setup parameters, which ones are fixed, limits
parameters = {'alpha_para':{'value':1.,     'error':0.1,  'limit_low': 0.5, 'limit_upp': 1.5, 'fixed': False}, 
              'alpha_perp':{'value':1.,     'error':0.1,  'limit_low': 0.5, 'limit_upp': 1.5, 'fixed': False},
              'bias'      :{'value':2.3,    'error': 0.1, 'limit_low': 1,   'limit_upp': 4.,  'fixed': False},
              'beta':      {'value':0.35,   'fixed':True}, 
              'sigma_rec' :{'value':15.,    'fixed':True}, 
              'sigma_para':{'value':7.31 ,  'fixed':True}, 
              'sigma_perp':{'value':5.53,   'fixed':True}, 
              'sigma_s'   :{'value':0.,     'fixed':True} 
              }

#-- Some extra options, e.g., broadband
options = {'fit_broadband': True, 
           'bb_min': -2, 
           'bb_max': 0}

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
chi.plot()
#plt.savefig('DR16_LRGxi_postrecon_bestfitmodel.pdf')

#-- Save results to file
chi.save('chi_xi.pkl')
