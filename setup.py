#!/usr/bin/env python
"""nestcheck setup."""
import os
import setuptools


def read_file(fname):
    """
    For using the README file as the long description.
    Taken from https://pythonhosted.org/an_example_pypi_project/setuptools.html
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(name='nestcheck',
                 version='0.0.0',
                 author='Edward Higson',
                 author_email='ejhigson@gmail.com',
                 description=('Diagnostic tests for nested sampling '
                              'calculations'),
                 url='https://github.com/ejhigson/nestcheck',
                 long_description=read_file('README.md'),
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   'matplotlib>=2.1.0',
                                   'fgivenx',  # >=1.1.4',
                                   'pandas>=0.21.0',
                                   'tqdm>=4.11'],
                 test_suite='nose.collector',
                 tests_require=['nose', 'coverage'],
                 extras_require={'docs': ['numpydoc']},
                 packages=['nestcheck'])
