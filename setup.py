#!/usr/bin/env python
"""
nestcheck setup module.

Based on https://github.com/pypa/sampleproject/blob/master/setup.py.
"""
import os
import setuptools


def get_package_dir():
    """Get the file path for nestcheck's directory."""
    return os.path.abspath(os.path.dirname(__file__))


def get_long_description():
    """Get PyPI long description from the .rst file."""
    pkg_dir = get_package_dir()
    with open(os.path.join(pkg_dir, '.pypi_long_desc.rst')) as readme_file:
        long_description = readme_file.read()
    return long_description


def get_version():
    """Get single-source __version__."""
    pkg_dir = get_package_dir()
    with open(os.path.join(pkg_dir, 'nestcheck/_version.py')) as ver_file:
        string = ver_file.read()
    return string.strip().replace('__version__ = ', '').replace('\'', '')


setuptools.setup(name='nestcheck',
                 version=get_version(),
                 description=('Error analysis, diagnostic tests and plots for '
                              'nested sampling calculations.'),
                 long_description=get_long_description(),
                 long_description_content_type='text/x-rst',
                 url='https://github.com/ejhigson/nestcheck',
                 author='Edward Higson',
                 author_email='e.higson@mrao.cam.ac.uk',
                 license='MIT',
                 keywords='nested-sampling sampling error-analysis',
                 classifiers=[  # Optional
                     'Development Status :: 5 - Production/Stable',
                     'Intended Audience :: Science/Research',
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
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   'matplotlib>=2.1.0',
                                   'fgivenx>=2.1.11',
                                   'pandas>=0.21.0',
                                   'tqdm>=4.11'],
                 test_suite='nose.collector',
                 tests_require=['nose', 'coverage'],
                 extras_require={
                     'docs': ['sphinx', 'numpydoc', 'sphinx-rtd-theme',
                              'nbsphinx>=0.3.3'],
                     ':python_version == "2.7"': ['futures']},
                 project_urls={  # Optional
                     'Docs': 'http://nestcheck.readthedocs.io/en/latest/'})
