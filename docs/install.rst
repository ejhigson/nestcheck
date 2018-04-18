.. _install:

Installation
============

.. (Not yet set up) ``nestcheck`` can be installed with `pip <http://www.pip-installer.org/>`_:

.. .. code-block:: bash

..    pip install nestcheck


``nestcheck`` and its dependencies can be installed by cloning `the git
repository <https://github.com/ejhigson/nestcheck>`_:

.. code-block:: bash

    git clone https://github.com/ejhigson/nestcheck.git
    cd nestcheck
    python setup.py install


It is compatible with python 2.7 and >=3.4.

Dependencies
------------

``nestcheck`` requires:

 - ``numpy`` >=1.13;
 - ``scipy`` >=1.0.0;
 - ``matplotlib`` >=2.1.0;
 - ``fgivenx`` >=1.1.4;
 - ``pandas`` >=0.21.0;
 - ``tqdm`` >=4.11.

All of the dependencies are on `PiPy <https://pypi.org/>`_ and are automatically installed by the above procedures.
Note also that producing the birth contour output files needed for ``nestcheck`` analysis with MultiNest requires v3.11 or later, and with PolyChord requires v1.13 or later and the setting "write_dead"=True.*

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
