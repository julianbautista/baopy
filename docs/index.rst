.. baopy documentation master file, created by
   sphinx-quickstart on Mon Jul 25 23:42:53 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to baopy's documentation!
=================================

**baopy** is a package for fitting two-point statistics of cosmological galaxy surveys,
either in Fourier space (power-spectrum) or configuration space (correlation function).

It contains models for fitting for the baryon acoustic oscillations (BAO) or 
for the redshift-space distortions (RSD). 

Requirements
============

It requires the following packages::

  numpy
  scipy
  iminuit
  astropy
  hankl
  

Installation
============

To install **baopy**:: 

  git clone https://github.com/julianbautista/baopy 

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user

Testing
=======

There are some examples of BAO fits in::

  cd examples/DR16_eBOSS_LRG 

Then run one of the fits::

  python fit_correlation_function.py 

or::

  python fit_power_spectrum.py 


.. toctree::
  :maxdepth: 1
  :caption: API

  api/baopy