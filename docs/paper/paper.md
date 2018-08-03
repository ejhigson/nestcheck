---
title: 'nestcheck: error analysis, diagnostic tests and plots for nested sampling calculations'
tags:
  - Python
  - nested sampling
  - dynamic nested sampling
  - Bayesian inference
  - error analysis
  - astronomy
authors:
  - name: Edward Higson
    orcid: 0000-0001-8383-4614
    affiliation: "1, 2"
affiliations:
 - name: Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, UK
   index: 1
 - name: Kavli Institute for Cosmology, Madingley Road, Cambridge, CB3 0HA, UK
   index : 2

date: 3 August 2018
bibliography: paper.bib
---

# Summary

Nested sampling [@Skilling2006] is a popular Monte Carlo method for Bayesian analysis which, given some likelihood and prior, provides both samples from the posterior distribution and an estimate of the Bayesian evidence.
Due to the distinctive manner in which the nested sampling algorithm explores the parameter space, the statistical properties of the samples it produces are different to samples from alternative techniques such as Markov chain Monte Carlo (MCMC)-based approaches.
As a result, posterior inferences and estimates of their associated uncertainties require methods specific to nested sampling which are non-trivial to implement.

``nestcheck`` is a pure Python package for analysing samples produced by nested sampling, and estimating uncertainty on posterior inferences.
Most importantly, ``nestcheck`` contains fast and well-tested implementations of the error analysis methods introduced in [@Higson2017a] and the diagnostic tests and plots described in [@Higson:2018a].
The code has been profiled for computational efficiency and makes use of fast ``numpy`` functions and parallelisation with ``concurrent.futures``.
The diagnostic plots make use of the ``matplotlib`` [@Hunter2007] and ``fgivenx`` [@zenodofgivenx] packages.

The functions are compatible with output from the popular ``MultiNest`` [@Feroz2008; @Feroz2009; @Feroz2013] and __`PolyChord`__ [@Handley2015a; @Handley2015b] packages.
``nestcheck`` is also compatible with samples produced by the dynamic nested sampling algorithm [@Higson2017b], and is used in the ``dyPolyChord`` [@zenododypolychord] and ``perfectns`` [@zenodoperfectns] dynamic nested sampling packages.

``nestcheck`` is intended as a convenient analysis tool for nested sampling software users; estimates of results' uncertainties are particularly important when reporting scientific results.
It was used for the diagnostic tests and plots in [@Higson:2018a], and for error analysis in [@Higson2017b] and [@Higson2018b].
An earlier version of the code was used in the analysis of black hole mergers in [@Chua2018].
The source code for ``nestcheck`` has been archived to Zenodo with the linked DOI: [@zenodonestcheck]

# Acknowledgements

We acknowledge help and advice from Will Handley, Anthony Lasenby and Mike Hobson.

# References
