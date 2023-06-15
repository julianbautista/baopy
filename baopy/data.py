''' Module handling data vectors and covariance matrices '''
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table 



class Data: 
    ''' Class dealing with data vector and covariance matrices 
        
        It contains functions to mask bins or plotting, and it 
        is mostly used in the Chi class

        Convention is :

        - space: integer, 0 for Fourier-space or 1 for Configuration-space
        - ell: multipole order  
        - scale: wavenumber ``k`` in units of h/Mpc for Fourier-space or separation ``r`` in units of Mpc/h for Configuration-space
        
        Use ``baopy.data.convert()`` to convert your data to this format !

    '''

    def __init__(self, data_file=None, cova_file=None):
        ''' Initialises the object by reading data vector and covariances
        They have to be in .ecsv format, which can be easily done with 
        ``baopy.data.convert()`` and ``baopy.data.covariance()``.

        Parameters
        ----------
        
        data_file : str, optional 
            Name of (.ecsv) file containing data vector and its coordinates
        cova_file : str, optional 
            Name of (.ecsv) file containing covariance matrix 

        '''

        self.coords = None
        self.values = None 
        self.cova = None 
        if data_file is not None:
            self.read_data(data_file)
        if cova_file is not None:
            self.read_cova(cova_file)
            self.match_cova()

    def read_data(self, data_file):
        ''' Reads data vector from .ecsv format
            
            Parameters 
            ----------
            data_file: str 
                Name of .ecsv file containing data vector 

        '''
        print(f'Reading data vector from: {data_file}')
        data = Table.read(data_file)
        coords = {  'space': data['space'].data,
                    'ell':   data['ell'].data,
                    'scale': data['scale'].data
                 }
        values = data['correlation'].data

        self.coords = coords
        self.values = values

    def read_cova(self, cova_file):
        ''' Reads covariance matrix from .ecsv format

            Parameters
            ----------

            cova_file: str 
                Name of .ecsv file containing covariance matrix and its coordinates
        '''
        print(f'Reading covariance from: {cova_file}')
        cova = Table.read(cova_file)

        cova_coords = { 'space_1': cova['space_1'].data,
                        'ell_1':   cova['ell_1'].data,
                        'scale_1': cova['scale_1'].data,
                        'space_2': cova['space_2'].data,
                        'ell_2':   cova['ell_2'].data,
                        'scale_2': cova['scale_2'].data
                     }
        cova_values = cova['covariance'].data
        
        self.cova_coords = cova_coords 
        self.cova_values = cova_values
    
    def match_cova(self):
        ''' Makes sure that data vector and covariance have matching coordinates 
        
        '''

        assert self.coords is not None 
        assert self.cova_coords is not None 

        data_coords = self.coords
        cova_coords = self.cova_coords 
        cova_values = self.cova_values

        cova_dict = {}
        for i in range(cova_values.size):
            s1 = cova_coords['space_1'][i]
            l1 = cova_coords['ell_1'][i]
            x1 = cova_coords['scale_1'][i]
            s2 = cova_coords['space_2'][i]
            l2 = cova_coords['ell_2'][i]
            x2 = cova_coords['scale_2'][i]           
            cova_dict[s1, l1, x1, s2, l2, x2] = cova_values[i]

        #-- Second, fill covariance matrix with only data vector elements, in the same order
        nbins = len(data_coords['space'])

        cova_match = np.zeros((nbins, nbins))
        for i in range(nbins):
            s1 = data_coords['space'][i]
            l1 = data_coords['ell'][i]
            x1 = data_coords['scale'][i]
            for j in range(nbins):
                s2 = data_coords['space'][j]
                l2 = data_coords['ell'][j]
                x2 = data_coords['scale'][j]
                cova_match[i, j] = cova_dict[s1, l1, x1, s2, l2, x2]

        self.cova = cova_match


    def apply_cuts(self, cuts):
        ''' Applies cuts to data vector and covariance matrix consistently

            Parameters
            ----------
            cuts: boolean np.array 
                Numpy array with the mask to be applied to data vector and 
                covariance matrix

        '''

        coords = self.coords
        self.coords = {k: coords[k][cuts] for k in coords}
        self.values = self.values[cuts]
        if self.cova is not None:
            cova = self.cova[:, cuts]
            self.cova = cova[cuts]
    
    def inverse_cova(self, nmocks=0):
        ''' 
        Computes inverse of the covariance matrix
        and applies Hartlap correction factor
            
        Parameters
        ----------
        nmocks: int, optional 
            Number of mocks used in the construction of covariance matrix
    
        '''

        inv_cova = np.linalg.inv(self.cova)
        correction = 1
        if nmocks > 0:
            #-- Hartlap correction
            correction = (1 - (self.values.size + 1.)/(nmocks-1))
            inv_cova *= correction
        
        self.hartlap_correction = correction  
        self.inv_cova = inv_cova
    
    def plot(self, y_model=None, f=None, axs=None, power_k=1, power_r=2, fill_between=False,
        alpha=None, figsize=(7, 8)):
        ''' 
        Plotting function

        Plots the multipoles in both spaces (if available) with error bars 

        Parameters
        ----------
        y_model : np.array, optional
            Array containing the model evaluated at the data coordinates
        f : plt.figure, optional
            Figure object already created and to be reused
        axs : array of Axes, optional
            Axes array already created and to be reused
        power_k : int
            Scales the plotted power spectra by ``k**power_k``
        power_r : int 
            Scales the plotted correlation function by ``r**power_r``
        fill_between : bool 
            If true, plots a shaded area instead of points with error bars

        '''
        
        coords = self.coords
        space = np.unique(coords['space'])
        n_space = space.size
        ell = np.unique(coords['ell'])
        n_ell = ell.size

        values = self.values

        if self.cova is not None:
            data_error = np.sqrt(np.diag(self.cova))
        else:
            data_error = None

        if f is None:
            f, axs = plt.subplots(ncols=n_space, nrows=n_ell, figsize=figsize, sharex='col', squeeze=False)

        xlabels = [r'$k$ [$h$ Mpc$^{-1}$]', r'$r$ [$h^{-1}$ Mpc]']
        if power_k == 0:
            title_k = r'$P_\ell(k)$'
        elif power_k == 1:
            title_k = fr'$k P_\ell(k)$'
        else:
            title_k = r'$k^{{{power_k}}} P_\ell(k)$'.format(power_k=power_k)

        if power_r == 0:
            title_r =  r'$\xi_\ell(k)$'
        elif power_r == 1:
            title_r = fr'$r \xi_\ell(k)$'
        else:
            title_r =  fr'$r^{{{power_r}}} \xi_\ell(k)$'
        titles = [title_k, title_r]


        for col in range(n_space):
            w_space = coords['space'] == space[col]
            for row in range(n_ell):
                w_ell = coords['ell'] == ell[row]
                w = w_space & w_ell
                x = coords['scale'][w]
                power_x = power_k if space[col] == 0 else power_r
                y_value = values[w]*x**power_x
                ax = axs[row, col]
                
                if data_error is not None:
                    y_error = data_error[w]*x**power_x
                    if fill_between: 
                        ax.fill_between(x, y_value-y_error, y_value+y_error, color=f'C{row}', alpha=alpha)
                    else: 
                        ax.errorbar(x, y_value, y_error, fmt='o', color=f'C{row}', alpha=alpha)
                else:
                    ax.plot(x, y_value, color=f'C{row}', alpha=alpha)
                if not y_model is None:
                    axs[row, col].plot(x, y_model[w]*x**power_x, color=f'C{row}')
                ax.grid(ls=':', color='k', alpha=0.3)
            axs[row, col].set_xlabel(xlabels[space[col]])
            axs[0, col].set_title(titles[space[col]])

        return f, axs
        
    def plot_error(self, y_model=None, f=None, axs=None, power_k=1, power_r=1, fill_between=False,
        alpha=None, color=None, ls=None, figsize=(7, 8)):
        ''' 
        Plotting function for uncertainties only

        Plots the uncertainties of multipoles in both spaces (if available) with error bars 

        Parameters
        ----------
        y_model : np.array, optional
            Array containing the model evaluated at the data coordinates
        f : plt.figure, optional
            Figure object already created and to be reused
        axs : array of Axes, optional
            Axes array already created and to be reused
        power_k : int
            Scales the plotted power spectra by ``k**power_k``
        power_r : int 
            Scales the plotted correlation function by ``r**power_r``
        fill_between : bool 
            If true, plots a shaded area instead of points with error bars

        '''
        
        coords = self.coords
        space = np.unique(coords['space'])
        n_space = space.size
        ell = np.unique(coords['ell'])
        n_ell = ell.size

        values = self.values
        data_error = np.sqrt(np.diag(self.cova))
 
        if f is None:
            f, axs = plt.subplots(ncols=n_space, nrows=n_ell, figsize=figsize, sharex='col', squeeze=False)

        xlabels = [r'$k$ [$h$ Mpc$^{-1}$]', r'$r$ [$h^{-1}$ Mpc]']
        if power_k == 0:
            title_k = r'$P_\ell(k)$'
        elif power_k == 1:
            title_k = fr'$k P_\ell(k)$'
        else:
            title_k = r'$k^{{{power_k}}} P_\ell(k)$'.format(power_k=power_k)

        if power_r == 0:
            title_r =  r'$\xi_\ell(k)$'
        elif power_r == 1:
            title_r = fr'$r \xi_\ell(k)$'
        else:
            title_r =  fr'$r^{{{power_r}}} \xi_\ell(k)$'
        titles = [title_k, title_r]


        for col in range(n_space):
            w_space = coords['space'] == space[col]
            for row in range(n_ell):
                w_ell = coords['ell'] == ell[row]
                w = w_space & w_ell
                x = coords['scale'][w]
                power_x = power_k if space[col] == 0 else power_r
                y_value = data_error[w]*x**power_x
                ax = axs[row, col]
                c = f'C{row}' if color is None else color
                ax.plot(x, y_value, color=c, alpha=alpha, ls=ls)

                if not y_model is None:
                    axs[row, col].plot(x, y_model[w]*x**power_x, color=c, ls=ls)
                
                ax.grid(ls=':', color='k', alpha=0.3)
            axs[row, col].set_xlabel(xlabels[space[col]])
            axs[0, col].set_title(titles[space[col]])

        return f, axs
    
    def plot_cova(self, normalise=True, **kwargs):
        ''' Plots the covariance matrix using pcolormeh 
        
        Parameters
        ----------
        normalise : bool,
            If true, normalises the covariance and shows the correlation coefficients
            
        '''

        cova = self.cova 
        corr = cova/np.sqrt(np.outer(np.diag(cova), np.diag(cova)))
 
        z = corr if normalise else cova
        plt.figure()
        plt.pcolormesh(z, **kwargs)
        plt.colorbar()



def convert(k=None, pk_mono=None, pk_quad=None, pk_hexa=None,
            r=None, xi_mono=None, xi_quad=None, xi_hexa=None):
    ''' Converts multipoles into a astropy Table 
        following convention described in ``baopy.data.Data``
        Returns object ready to be exported in .ecsv format. 
    
    Parameters
    ----------
    k : array, optional
        Wavenumber in units of h/Mpc for Fourier space
    r : array, optional 
        Separation in units of Mpc/h for Configuration space 
    pk_mono, pk_quad, pk_hexa: arrays, optional
        Monopole, quadrupole, hexadecapole of the power spectrum
    xi_mono, xi_quad, xi_hexa: arrays, optional 
        Monopole, quadrupole, hexadecapole of the correlation function 

    Returns
    -------
    data_out : astropy.table.Table 
        Table containing multipoles ready to be exported in .ecsv format
    '''

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

def covariance_from_realisations(list_of_files):
    ''' 
    Reads a list of files in .ecsv format 
    (exported by ``baopy.data.convert()``) 
    and computes a mean and covariance matrix. 
    To be exported in .ecsv format. 
    
    Parameters
    ----------
    list_of_files: list
        List of str containing the paths to all realisations 

    Returns
    -------
    mean_out : astropy.table.Table
        Contains the average of all realisations
    cova_out : astropy.table.Table 
        Containts the covariance matrix 

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
    
