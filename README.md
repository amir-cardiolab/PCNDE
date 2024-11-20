# PCNDE

---

This repository contains the Julia/ipynb codes for the following paper:

**Physics-constrained coupled neural differential equations for one dimensional blood flow modeling**

Hunor Csala, Arvind Mohan, Daniel Livescu, Amirhossein Arzani

Preprint on arXiv: [https://arxiv.org/abs/2411.05631](https://arxiv.org/abs/2411.05631)

---


The src directory contains helper function for calculating numerical derivatives with different schemes and for training with different optimizers.

The trained_weights directory contains the trained weights for the momentum and continuity equation models in .jld2 format.

PCNDE_train_QS.jl has the momentum and continuity equation training.

PCNDE_InferenceFitPressure.ipynb has the PCNDE model in inference mode, and the pressure model in training/inference mode. This jupyter notebook also has some of the plots from the paper, for example parts of Fig. 4, Fig. 5, Fig. 6 and Fig. 7.
