nestcheck
=========

.. image:: https://travis-ci.org/ejhigson/nestcheck.svg?branch=master
    :target: https://travis-ci.org/ejhigson/nestcheck
.. image:: https://coveralls.io/repos/github/ejhigson/nestcheck/badge.svg?branch=master
	:target: https://coveralls.io/github/ejhigson/nestcheck?branch=master
.. image:: https://readthedocs.org/projects/nestcheck/badge/?version=latest
	:target: http://nestcheck.readthedocs.io/en/latest/?badge=latest
.. image:: https://api.codeclimate.com/v1/badges/7fdfe74eb8256020c780/maintainability
    :target: https://codeclimate.com/github/ejhigson/nestcheck/maintainability
    :alt: Maintainability
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/ejhigson/nestcheck/blob/master/LICENSE

``nestcheck`` provides python utilities for analysing nested sampling runs and estimating numerical uncertainties. This includes implementations of the diagnostic tests and plots described in:

- "Diagnostic Tests for Nested Sampling Calculations" (`Higson et al., 2018 <https://arxiv.org/abs/1804.06406>`_);
- "Sampling Errors in Nested Sampling Parameter Estimation" (`Higson et al., 2017 <https://doi.org/10.1214/17-BA1075>`_).

To get started, see the `installation instructions <http://nestcheck.readthedocs.io/en/latest/install.html>`_ and the `quickstart demo <http://nestcheck.readthedocs.io/en/latest/demos/quickstart_demo.html>`_. More examples, including the code used to make the results and plots in `Higson et al. (2018) <https://arxiv.org/abs/1804.06406>`_, can be found in the `examples folder <https://github.com/ejhigson/nestcheck/tree/master/examples>`_.

Compatible nested sampling software
-----------------------------------

Currently ``nestcheck.data_processing`` has functions to load results from:

- `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest/>`_ >=v3.11;
- `PolyChord <https://ccpforge.cse.rl.ac.uk/gf/project/polychord/>`_ >=v1.13;
- `dyPolyChord <https://github.com/ejhigson/dyPolyChord>`_ (same output format as PolyChord);
- `dynesty <https://github.com/joshspeagle/dynesty>`_;
- `perfectns <https://github.com/ejhigson/perfectns>`_.

You can easily add input functions for other nested sampling software packages. Note that ``nestcheck`` requires information about the iso-likelihood contours within which dead points were sampled ("born"), which is needed to split nested sampling runs into their constituent single live point runs ("threads"); see `Higson et al. (2017) <https://doi.org/10.1214/17-BA1075>`_ for more details.
``nestcheck`` is fully compatible with `dynamic nested sampling <https://arxiv.org/abs/1704.03459>`_, in which the number of live points is varied to increase calculation accuracy.


Documentation contents
----------------------

.. toctree::
   :maxdepth: 2

   install
   demos/quickstart_demo
   api


Attribution
-----------

If this code is useful for your academic research, please cite the two papers on which it is based. The BibTeX is:

.. code-block:: tex

    @article{higson2018diagnostic,
    title={Diagnostic Tests for Nested Sampling Calculations},
    author={Higson, Edward and Handley, Will and Hobson, Mike and Lasenby, Anthony},
    journal={arXiv preprint arXiv:1804.06406},
    url={1804.06406},
    year={2018}}

    @article{higson2018sampling,
    title={Sampling Errors in Nested Sampling Parameter Estimation},
    author={Higson, Edward and Handley, Will and Hobson, Mike and Lasenby, Anthony},
    doi={doi:10.1214/17-BA1075},
    journal={Bayesian Analysis},
    url={https://doi.org/10.1214/17-BA1075},
    year={2018}}


Changelog
---------

The changelog for each release can be found at https://github.com/ejhigson/nestcheck/releases.

Contributions
-------------

Contributions are welcome! Development takes place on github:

- source code: https://github.com/ejhigson/nestcheck;
- issue tracker: https://github.com/ejhigson/nestcheck/issues.

When creating a pull request, please try to make sure the tests pass and use numpy-style docstrings.

If you have any questions or suggestions please get in touch (e.higson@mrao.cam.ac.uk).

Authors & License
-----------------

Copyright 2018 Edward Higson and contributors (MIT Licence).
