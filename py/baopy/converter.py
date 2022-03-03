import numpy as np
from astropy.table import Table 

def convert(k=None, pk_mono=None, pk_quad=None, pk_hexa=None,
            r=None, xi_mono=None, xi_quad=None, xi_hexa=None):
            
    data = []
    if not k is None:
        s = 0 
        for ell, mult in zip([0, 2, 4], [pk_mono, pk_quad, pk_hexa]):
            if mult is None:
                continue
            for i in range(mult.size):
                data.append([s, ell, k[i], mult[i]])

    if not r is None:
        s = 1 
        for ell, mult in zip([0, 2, 4], [xi_mono, xi_quad, xi_hexa]):
            if mult is None:
                continue
            for i in range(mult.size):
                data.append([s, ell, r[i], mult[i]])
    data = np.array(data)

    data_out = Table(data=data, 
                    names=['space', 'ell', 'scale', 'correlation'], 
                    dtype=[int, int, float, float])
    return data_out
