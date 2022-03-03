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

def covariance(list_of_files):
    ''' Reads a list of files in ecsv format (defined above) and
        computes a mean and covariance matrix to be exported in ecsv 
        format for baopy
    '''

    #-- Reads files and stores correlations
    correlations = []
    for fi in list_of_files:
        data = Table.read(fi)
        correlations.append(data['correlation'])
    correlations = np.array(correlations)

    #-- Computes average and covariance matrix
    mean = np.mean(correlations, axis=0)
    cova = np.cov(correlations.T)

    #-- Makes a table for average
    mean_out = data.copy()
    mean_out['correlation'] = mean

    #-- Makes a table for covariance
    coords = data['space', 'ell', 'scale']
    nbins = len(coords)
    rows = []
    for i in range(nbins):
        row_i = [coords[i][key] for key in ['space', 'ell', 'scale']]
        for j in range(nbins):
            row_j = [coords[j][key] for key in ['space', 'ell', 'scale']]
            rows.append(row_i+row_j+[cova[i, j]])
    rows = np.array(rows)

    cova_out = Table(data=rows, 
        names=['space_1', 'ell_1', 'scale_1', 'space_2', 'ell_2', 'scale_2', 'covariance'],
        dtype=[int, int, float, int, int, float, float])
    
    return mean_out, cova_out
    


