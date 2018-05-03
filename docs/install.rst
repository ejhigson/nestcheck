.. _install:

Installation
============

``nestcheck`` is compatible with python 2.7 and >=3.4, and can be installed with `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

   pip install nestcheck


Alternatively, you can download the latest version and install it by cloning `the git
repository <https://github.com/ejhigson/nestcheck>`_:

.. code-block:: bash

    git clone https://github.com/ejhigson/nestcheck.git
    cd nestcheck
    python setup.py install

Both of these methods also automatically install any of ``nestcheck``'s dependencies which are not already satisfied by your system.


Dependencies
------------

``nestcheck`` requires:

 - ``numpy`` >=1.13;
 - ``scipy`` >=1.0.0;
 - ``matplotlib`` >=2.1.0;
 - ``fgivenx`` >=1.1.4;
 - ``pandas`` >=0.21.0;
 - ``tqdm`` >=4.11.


Note also that producing the birth contour output files needed for ``nestcheck`` analysis using `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest/>`_ requires v3.11 or later, and using `PolyChord <https://ccpforge.cse.rl.ac.uk/gf/project/polychord/>`_ requires v1.13 or later and the setting "write_dead"=True (its default value).


Tests
-----

You can run the test suite with `nose <http://nose.readthedocs.org/>`_. From the root of the ``dyPolyChord`` directory, run:

.. code-block:: bash

    nosetests

To also get code coverage information (this requires the ``coverage`` package), use:

.. code-block:: bash

    nosetests --with-coverage --cover-erase --cover-package=nestcheck

If all the tests pass, the install should be working.
