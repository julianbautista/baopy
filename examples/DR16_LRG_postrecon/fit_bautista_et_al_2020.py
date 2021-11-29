import matplotlib.pyplot as plt
import baopy.bao_fitter

#-- Read data files and covariance
dat = baopy.bao_fitter.Data(data_file='Data_LRGxi_NGCSGC_0.6z1.0_postrecon_slx.txt',
           cova_file='Covariance_LRGxi_NGCSGC_0.6z1.0_postrecon_slx.txt')

#-- Select multipoles and scale ranges
cuts = (((dat.coords['ell'] == 0) & (dat.coords['scale'] > 50) & (dat.coords['scale'] < 150)) | 
        ((dat.coords['ell'] == 2) & (dat.coords['scale'] > 50) & (dat.coords['scale'] < 150)) )
        #| ((dat.coords['ell'] == 4) & (dat.coords['scale'] > 50) & (dat.coords['scale']<150)))
dat.apply_cuts(cuts)
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
