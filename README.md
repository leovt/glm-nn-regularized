# Regularized GLM–Neural Network Hybrids for Insurance Pricing

## Project Outline

Full outline: [CAS Innovation Project Outline](deliverables/outline/CAS_Innovation_Project_Outline.md)

Literature links: [literature.md](literature.md)

## Motivation & Topic

Wüthrich et al.'s *AI Tools for Actuaries* presents two complementary ways to combine GLMs with neural networks: the Combined Actuarial Neural Network (CANN), which freezes the GLM and adds a correction network as a skip connection, and LocalGLMnet, which lets a network learn locally-varying GLM coefficients.
Both preserve interpretability, but via different architectural choices.
This project explores a third option: training the GLM and a small correction network jointly, with the correction branch (initialized near zero) shrunk toward the pure-GLM solution via an explicit regularization penalty (ridge, with LASSO as an optional extension).
Regularization strength becomes a tunable dial between GLM-like interpretability and FNN-like flexibility, rather than a hard architectural constraint.

## Objective & Contribution

Rather than comparing model variants on predictive performance alone, the project assesses whether a plain GLM structure should be preferred unless additional nonlinear structure yields a sufficient improvement in likelihood, and whether asymmetric regularization can encode this preference within a single jointly-trained architecture.
Predictive performance is evaluated via out-of-sample Poisson deviance (see Evaluation).

The central result is how predictive performance and departure from GLM structure move together as λ varies, illustrated by dependence plots at representative values.

## Data

The data archive was downloaded from [AI Tools for Actuaries data](https://people.math.ethz.ch/~wueth/AITools/Data.zip) on 22 August 2026 and stored locally as `data/Data.zip`.

SHA-256:

```text
3175fab0efa56ac5383a8803ef6c9243b1dc2c581e03809f684fbd70dde88c47  data/Data.zip
```

## Repository Structure

TODO.
