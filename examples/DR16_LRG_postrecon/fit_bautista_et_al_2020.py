import matplotlib.pyplot as plt
from baopy.bao_fitter import Data, Model, Chi2 

#-- Read data files and covariance
d = Data(data_file='Data_LRGxi_NGCSGC_0.6z1.0_postrecon_slx.txt',
         cova_file='Covariance_LRGxi_NGCSGC_0.6z1.0_postrecon_slx.txt')

#-- Select multipoles and scale ranges
cuts = (((d.coords['ell'] == 0) & (d.coords['scale'] > 50) & (d.coords['scale'] < 150)) | 
        ((d.coords['ell'] == 2) & (d.coords['scale'] > 50) & (d.coords['scale'] < 150)) )
        #| ((d.coords['ell'] == 4) & (d.coords['scale'] > 50) & (d.coords['scale']<150)))
d.apply_cuts(cuts)
d.inverse_cova(nmocks=1000)

#-- Read linear power spectrum from file
m = Model('pk_camb_z0.698_challenge.txt')

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
c = Chi2(data=d, model=m, parameters=parameters, options=options)
c.fit()

#-- Compute precise errors for the alphas
print(c.mig.minos('alpha_para', 'alpha_perp'))

#-- Plot best-fit model and save it 
c.plot()
plt.savefig('DR16_LRG_postrecon_bestfitmodel.pdf')