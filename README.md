# Strategic Classification with Non-Linear Classifiers
Code for experiments in the paper Strategic Classification with Non-Linear Classifiers by Benyamin Trachtenberg and Nir Rosenfeld.

# Abstract
In strategic classification, the standard supervised learning setting is extended
to support the notion of strategic user behavior in the form of costly feature
manipulations made in response to a classifier. While standard learning supports
a broad range of model classes, the study of strategic classification has, so far,
been dedicated mostly to linear classifiers. This work aims to expand the horizon
by exploring how strategic behavior manifests under non-linear classifiers and
what this implies for learning. We take a bottom-up approach showing how non-
linearity affects decision boundary points, classifier expressivity, and model classes
complexity. A key finding is that universal approximators (e.g., neural nets) are
no longer universal once the environment is strategic. We demonstrate empirically
how this can create performance gaps even on an unrestricted model class.

# Expressivity Experiment 
To test whether the expressivity of a given $h$ increases or decreases practically, we randomly generate degree-k polynomials ($k \in [1...10]$) and determine whether the degree, $k_\Delta$, of the polynomial approximation of $h_\Delta$ is larger or smaller than $k$.

# Approximation Experiment
To demonstrate the practical implications of the limited maximal strategic accuracy, we generate synthetic data with varying levels of data seperability and then calculate the maximum strategic and linear accuracy.
