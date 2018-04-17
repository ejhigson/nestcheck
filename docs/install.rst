.. _install:

Installation
============

.. (Not yet set up) ``nestcheck`` can be installed with `pip <http://www.pip-installer.org/>`_:

.. .. code-block:: bash

..    pip install nestcheck

Install ``nestcheck`` and its dependencies by cloning `the git
repository <https://github.com/ejhigson/nestcheck>`_:

.. code-block:: bash

    git clone https://github.com/ejhigson/nestcheck.git
    cd nestcheck
    python setup.py install


Dependencies
------------

``nestcheck`` requires

 - ``numpy`` >=1.13;
 - ``scipy`` >=1.0.0;
 - ``matplotlib`` >=2.1.0;
 - ``fgivenx`` >=1.1.4;
 - ``pandas`` >=0.21.0;
 - ``tqdm`` >=4.11.

Currently, ``nestcheck`` can process MultiNest and PolyChord output files.
These must include infomation about the steps at which points were sampled ("born") is required as it is needed to split nested sampling runs into their constituent single live point runs ("threads"); see `Higson et al. 2017 <https://doi.org/10.1214/17-BA1075>`_ for more details.
*Producing these requires MultiNest >= v3.11 and PolyChord >= v1.13.*

Tests
-----

You can run the test suite using `nose
<http://nose.readthedocs.org/>`_. From the root of the ``nestcheck`` directory, run:

.. code-block:: bash

    nosetests

To also get code coverage information (requires the ``coverage`` package)

.. code-block:: bash

    nosetests --with-coverage --cover-erase --cover-package=nestcheck

If all the tests pass, the install should be working.
