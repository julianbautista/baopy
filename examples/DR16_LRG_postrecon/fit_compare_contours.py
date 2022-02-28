import baopy.fitter
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

#-- Read results from :
#-- fit_correlation_function.py
#-- fit_power_spectrum.py
#-- fit_joint.py

chi_xi = baopy.fitter.Chi2.load('results_fit_correlation_function.pkl')

chi_pk = baopy.fitter.Chi2.load('results_fit_power_spectrum.pkl')

chi_pkxi = baopy.fitter.Chi2.load('results_fit_joint.pkl')


names = [r'$P_\ell$', r'$\xi_\ell$', 'Joint']
chis = [chi_pk, chi_xi, chi_pkxi]

f, ax = plt.subplots(1, 1)

for i in range(len(chis)):
    #-- Read best-fit values
    alpha_perp = chis[i].output['best_pars']['alpha_perp']
    alpha_para = chis[i].output['best_pars']['alpha_para']
    #-- Read contours
    contours = chis[i].output['contours']['alpha_perp', 'alpha_para']
    
    #-- 2-sigma contour
    contour = contours[0.95]
    polygon = Polygon(contour, edgecolor=f'C{i}', facecolor='none', ls='--', lw=2)
    ax.add_patch(polygon)

    #-- 1-sigma contour
    contour = contours[0.68]
    polygon = Polygon(contour, edgecolor=f'C{i}', facecolor='none', ls='-', lw=2, label=names[i])
    ax.add_patch(polygon)
    
    ax.autoscale_view()
    
    ax.plot(alpha_perp, alpha_para, f'C{i}o')

ax.set_xlabel(r'$\alpha_\perp$', fontsize=12)
ax.set_ylabel(r'$\alpha_\parallel$', fontsize=12)
ax.legend()
ax.grid(linestyle=':')
ax.set_aspect('equal')

plt.savefig('results_compare_contours.pdf')