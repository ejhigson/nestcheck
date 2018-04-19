#!/usr/bin/env python
"""
nestcheck setup module.

Based on https://github.com/pypa/sampleproject/blob/master/setup.py.
"""
import os
import setuptools


def get_long_description():
    """Get PyPI long description from the .rst file."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, '.pypi_long_desc.rst')) as readme_file:
        long_description = readme_file.read()
    return long_description


setuptools.setup(name='nestcheck',
                 version='0.0.1',
                 description=('Error analysis, diagnostic tests and plots for '
                              'nested sampling calculations.'),
                 long_description=get_long_description(),
                 long_description_content_type='text/markdown',
                 url='https://github.com/ejhigson/nestcheck',
                 author='Edward Higson',
                 author_email='ejhigson@gmail.com',
                 license='MIT',
                 keywords='nested-sampling sampling error-analysis',
                 classifiers=[  # Optional
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'Topic :: Software Development :: Build Tools',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 2',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Topic :: Scientific/Engineering :: Astronomy',
                     'Topic :: Scientific/Engineering :: Physics',
                     'Topic :: Scientific/Engineering :: Visualization',
                     'Topic :: Scientific/Engineering :: Information Analysis',
                 ],
                 packages=['nestcheck'],
                 # futures is in standard library for python >= 3.2, but
                 # include it for compatibility with python 2.7.
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   'matplotlib>=2.1.0',
                                   'fgivenx>=1.1.4',
                                   'pandas>=0.21.0',
                                   'tqdm>=4.11',
                                   'futures'],
                 test_suite='nose.collector',
                 tests_require=['nose', 'coverage'],
                 # Force sphinx<1.6 to avoid readthedocs error as per this
                 # https://github.com/rtfd/readthedocs.org/issues/3148
                 extras_require={'docs': ['sphinx<1.6', 'numpydoc',
                                          'sphinx-rtd-theme', 'nbsphinx']},
                 project_urls={  # Optional
                     'Docs': 'http://nestcheck.readthedocs.io/en/latest/',
                     'Source': 'https://github.com/ejhigson/nestcheck'})
