---
title: 'nestcheck: error analysis, diagnostic tests and plots for nested sampling calculations'
tags:
  - Python
  - nested sampling
  - dynamic nested sampling
  - Bayesian inference
  - error analysis
authors:
  - name: Edward Higson
    orcid: 0000-0001-8383-4614
    affiliation: "1, 2"
affiliations:
 - name: Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, UK
   index: 1
 - name: Kavli Institute for Cosmology, Madingley Road, Cambridge, CB3 0HA, UK
   index : 2
date: 4 August 2018
bibliography: paper.bib
---

# Summary

Nested sampling [@Skilling2006] is a popular Monte Carlo method for Bayesian analysis which, given some likelihood and prior, provides both samples from the posterior distribution and an estimate of the Bayesian evidence.
Due to the distinctive manner in which the nested sampling algorithm explores the parameter space, it produces posterior samples with different statistical properties to those generated from alternative techniques such as Markov chain Monte Carlo (MCMC)-based approaches.
As a result, posterior inferences and estimates of their associated uncertainties require methods specific to nested sampling.

``nestcheck`` is a Python package for analysing samples produced by nested sampling, and estimating uncertainty on posterior inferences.
Most importantly, ``nestcheck`` contains fast and well-tested implementations of the error analysis methods introduced in [@Higson2017a] and the diagnostic tests and plots described in [@Higson:2018a].
The code has been profiled for computational efficiency and uses fast ``numpy`` functions and parallelisation with ``concurrent.futures``.
The diagnostic plots make use of the ``matplotlib`` [@Hunter2007] and ``fgivenx`` [@zenodofgivenx] packages.

``nestcheck`` can analyse samples from the popular ``MultiNest`` [@Feroz2008; @Feroz2009; @Feroz2013] and ``PolyChord`` [@Handley2015a; @Handley2015b] packages, and functions for loading samples from other software packages with different formats can easily be added.
``nestcheck`` is also compatible with samples produced by the dynamic nested sampling algorithm [@Higson2017b], and its functions for storing and manipulating nested sampling output are used by the ``dyPolyChord`` [@zenododypolychord] and ``perfectns`` [@zenodoperfectns] dynamic nested sampling packages.

``nestcheck`` is designed to allow nested sampling software users to quickly calculate results and uncertainty estimates, as well as to apply diagnostics for checking their software has explored the posterior correctly.
It was used for the diagnostic tests and plots in [@Higson:2018a], and for error analysis in [@Higson2017b] and [@Higson2018b].
An earlier version of the code was used in the analysis of black hole mergers in [@Chua2018].
The source code for ``nestcheck`` has been archived to Zenodo [@zenodonestcheck].

# Acknowledgements

I am grateful to Will Handley, Anthony Lasenby and Mike Hobson for their help and advice.

# References
