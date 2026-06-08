"""
ks.py --- source code for the Kuramoto-Sivashinsky equation class on the 1D periodic grid with different shift/interpolation/spatial derivative operators
"""

import numpy as np
from typing import Union
from SROpInf.typing import Vector, Matrix
from SROpInf.grids.grid1d import Grid1DUniformSpectral, Grid1DCubicSpline
from SROpInf.models.model import FullOrderModel

class KuramotoSivashinsky(FullOrderModel):
    """Class for the Kuramoto-Sivashinsky equation on the 1D periodic grid with different shift/interpolation/spatial derivative operators.
    u_t + u u_x + u_xx + nu u_xxxx = 0
    Since the primal variable is u, and the energy-based inner product is based only on L2 norm:
    <u, u>_E = <u, u>_L2,
    we can use the same inner product defined on the grid class
    so we don't need to override the 'inner_product' or the 'apply_sqrt_inner_product_mass' method from the base 'FullOrderModel' class.
    """
    def __init__(self,
                grid: Union[Grid1DUniformSpectral, Grid1DCubicSpline],
                nu: float):
        # IMPORTANT: populate poly_operators BEFORE super().__init__() so that
        # FullOrderModel.__init__ caches num_poly_terms = len(poly_operators) correctly.
        # Otherwise num_poly_terms stays at 0 and rhs() returns all zeros.

        self.nu = nu
        # when nu = 4/87, Lx = 2 pi,
        # the KS equation exhibits traveling beating waves after t = 120,
        # when starting from sine and cosine perturbations at t = 0.
        self.poly_operators = {
            1: self._linear,
            2: self._bilinear,
        }
        super().__init__(grid)
        # Precompute the 3/2-rule zero-padding constants used by _bilinear on spectral grids
        # (Orszag 1971). Cubic-spline grid has no Fourier representation, so the spline branch
        # of _bilinear stays a plain nodal product.
        if hasattr(self.grid, "kx"):
            self.grid_pad = Grid1DUniformSpectral(self.grid.Lx, 3 * self.grid.nx // 2)
        else:
            self.grid_pad = None

    def _linear(self, q: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Evaluate the linear operator -u_xx - nu u_xxxx on each column of q,
        where q has shape (nx, M) with M columns representing M vectorized states,
        or q has shape (nx,) representing a single state."""
        return -1 * self.grid.diff_x(q, order = 2) - self.nu * self.grid.diff_x(q, order = 4)

    def _bilinear(self, q1: Union[Vector, Matrix], q2: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Evaluate the symmetrized bilinear operator -0.5 * (u1 u2_x + u2 u1_x).

        On a spectral grid, the nodal product u * u_x aliases (high modes wrap back into the
        linearly unstable band of KS, causing blow-up). We therefore use Orszag's 3/2-rule
        zero-padding (Orszag 1971): zero-pad the Fourier coefficients of u1, u2 to length 3N/2,
        do the multiplication in the finer physical space, then FFT back and truncate to N modes.
        This is exactly alias-free for the lower-N output modes.

        On a cubic-spline grid the multiplication is performed directly in nodal space
        (no analogous Fourier-aliasing concern applies)."""
        if q1.shape != q2.shape:
            raise ValueError(f"q1 and q2 must have the same shape, got {q1.shape} and {q2.shape}")
        if self.grid_pad is None:
            return -0.5 * (q1 * self.grid.diff_x(q2, order=1) + q2 * self.grid.diff_x(q1, order=1))
        else:
            q1_pad, q2_pad = self.grid.pad(q1), self.grid.pad(q2) # step a
            output_pad = -0.5 * (q1_pad * self.grid_pad.diff_x(q2_pad, order=1) + q2_pad * self.grid_pad.diff_x(q1_pad, order=1))
            output_truncated = self.grid.truncate(output_pad) # step c
            return output_truncated
        
    