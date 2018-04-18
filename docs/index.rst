nestcheck
=========

.. image:: https://travis-ci.org/ejhigson/nestcheck.svg?branch=master
    :target: https://travis-ci.org/ejhigson/nestcheck
.. image:: https://coveralls.io/repos/github/ejhigson/nestcheck/badge.svg?branch=master
	:target: https://coveralls.io/github/ejhigson/nestcheck?branch=master
.. image:: https://readthedocs.org/projects/nestcheck/badge/?version=latest
	:target: http://nestcheck.readthedocs.io/en/latest/?badge=latest
.. image:: https://api.codeclimate.com/v1/badges/a99a88d28ad37a79dbf6/maintainability
   :target: https://codeclimate.com/github/codeclimate/codeclimate/maintainability
   :alt: Maintainability
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/ejhigson/nestcheck/LICENSE

``nestcheck`` provides python utilities for analysing nested sampling runs and estimating numerical uncertainties. This includes implementations of the diagnostic tests and plots described in "Diagnostic Tests for Nested Sampling Calculations" (`Higson et al. 2018
<https://arxiv.org/abs/TBC>`_) and
"Sampling Errors in Nested Sampling Parameter Estimation" (`Higson et al. 2017
<https://doi.org/10.1214/17-BA1075>`_). For more information see the papers and the documentation. 
``nestcheck`` is fully compatible with `dynamic nested sampling
<https://arxiv.org/abs/1704.03459>`_.

Currently functions for processing output from ``MultiNest``, ``PolyChord`` and ``perfectns`` are available, and more nested sampling software packages can be added. ``nestcheck`` requires information about the iso-likelihood contours within which dead points were sampled ("born"), which is needed to split nested sampling runs into their constituent single live point runs ("threads"); see `Higson et al. 2017 <https://doi.org/10.1214/17-BA1075>`_ for more details. *Producing birth contour output files with MultiNest requires v3.11 or later, and with PolyChord requires v1.13 or later and the setting "write_dead"=True.*

IPython notebooks containing example usage of ``nestcheck`` can be found in the `examples folder
<https://github.com/ejhigson/nestcheck/tree/master/examples>`_. This includes the code used to make the results in the diagnostics tests paper (`Higson et al. 2018
<https://arxiv.org/abs/TBC>`_).

Documentation
-------------

.. toctree::
   :maxdepth: 2

   install
   demos/quickstart_demo
   api

Attribution
-----------

If this code is useful for your research, please cite the two papers on which it is based - the bibtex is:

.. code-block:: tex

    @article{higson2018a,
    title={Diagnostic Tests for Nested Sampling Calculations},
    author={Higson, Edward and Handley, Will and Hobson, Mike and Lasenby, Anthony},
    journal={arXiv preprint arXiv:TBC},
    url={TBC},
    year={2018}
    }

    @article{higson2017a,
    title={Sampling Errors in Nested Sampling Parameter Estimation},
    author={Higson, Edward and Handley, Will and Hobson, Mike and Lasenby, Anthony},
    doi={doi:10.1214/17-BA1075},
    journal={Bayesian Analysis},
    url={https://doi.org/10.1214/17-BA1075},
    year={2017}
    }


Contributions
-------------

Contributions are welcome! Please use numpy-style docstrings, and make sure the tests pass and  before issuing a pull request.

Authors & License
-----------------

Copyright 2018 Edward Higson (MIT Licence).
