# Symmetry-Reduced Model Reduction for Shift-Equivariant Systems via Operator Inference

This repository provides a Python implementation of the **symmetry-reduced operator inference** framework for learning projection-based reduced-order models (ROMs) of shift-equivariant systems—particularly for partial differential equations.
The framework is **data-driven** and **non-intrusive**, making it a viable candidate for model reduction of "black-box" systems.

Our method builds upon:

- **Operator Inference**, as introduced by [Peherstorfer and Willcox (2016)](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
- **Symmetry reduction techniques**, following [Rowley and Marsden (2000)](https://www.sciencedirect.com/science/article/pii/S0167278900000427).

This work has been submitted to *Advances in Computational Mathematics* and is currently under review. A preprint is available on [arXiv:2507.18780](https://arxiv.org/abs/2507.18780).

# Running the codes

To run the demo of building SR-OpInf ROMs on the Kuramoto-Sivashinsky system, please:

- Set the root folder /SROpInf as your working directory
- Run the ks.py in the subfolder /src
