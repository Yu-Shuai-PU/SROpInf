"""
grid1d.py --- source code for 1D periodic grid with different shift/interpolation/spatial derivative operators

classes:
    Grid1DUniformSpectral: 1D periodic grid with spectral shift/interpolation/spatial derivative operators
    Grid1DCubicSpline: 1D periodic grid with cubic spline shift/interpolation/spatial derivative operators
"""

import numpy as np
import scipy as sp
from abc import ABC, abstractmethod
from typing import Union
from SROpInf.typing import Vector, Matrix, ComplexVector, ComplexMatrix

def fft(q: Union[Vector, Matrix]) -> Union[ComplexVector, ComplexMatrix]:
    """Compute the Fourier transform of q (vector or matrix) along the spatial dimension (axis 0).
    For even nx, the Nyquist mode (at index 0 after fftshift) is zeroed out."""
    q_hat = np.fft.fftshift(np.fft.fft(q, axis=0), axes=0) / q.shape[0]
    q_hat[0] = 0.0
    return q_hat

def ifft(q_hat: Union[ComplexVector, ComplexMatrix]) -> Union[Vector, Matrix]:
    """Compute the inverse Fourier transform of q_hat (vector or matrix) along the spatial dimension (axis 0)."""
    return np.fft.ifft(np.fft.ifftshift(q_hat, axes=0), axis=0).real * q_hat.shape[0]

class Grid1D(ABC):
    """Base class for 1D periodic grid with different shift/interpolation/spatial derivative operators."""
    def __init__(self, Lx: float):
        self.Lx = Lx
        
    @abstractmethod    
    def inner_product(self, q1: Union[Vector, Matrix], q2: Union[Vector, Matrix]) -> Union[float, Matrix]:
        """Compute the inner product of q1 (vector or matrix) and q2 using the appropriate quadrature weights."""
        pass
    
    @abstractmethod
    def apply_sqrt_inner_product_mass(self, A: Union[Vector, Matrix], action: str = "forward") -> Union[Vector, Matrix]:
        """Apply the square root R of the inner-product mass matrix M, defined by M = R^T R, so that
            <q1, q2> = q1^T M q2 = (R q1)^T (R q2) = <R q1, R q2>_l2.
        This convention matches NiSRH-ROM (see NiSRH-ROM/src/nisrh/models/fom.py docstring of rhs_scaled
        and examples/LNS/lns3d.py apply_sqrt_inner_product_mass).
            action = "forward":           apply R       to A   (scales M-IP into l2-IP)
            action = "inverse":           apply R^{-1}  to A   (recovers physical coords)
            action = "transpose":         apply R^T     to A   (M-adjoint diagnostics)
            action = "inverse_transpose": apply R^{-T}  to A   (M-adjoint diagnostics)
        Only 'forward' and 'inverse' are load-bearing in SR-OpInf production paths; 'transpose' and
        'inverse_transpose' are provided for API parity with NiSRH-ROM. Works for 1D and 2D inputs.
        """
        pass
    
    def norm(self, q: Union[Vector, Matrix]) -> Union[float, Vector]:
        """Compute the norm of q (vector or matrix) using the appropriate quadrature weights."""
        if q.ndim == 1:
            return np.sqrt(self.inner_product(q, q))
        elif q.ndim == 2:
            return np.sqrt(np.diag(self.inner_product(q, q)))
        else:
            raise ValueError(f"Input q must be 1D or 2D, got shape {q.shape!r}")
        
    @abstractmethod
    def shift_x(self, q: Union[Vector, Matrix], c: Union[float, Vector]) -> Union[Vector, Matrix]:
        """Shift the input q (vector or matrix)by c (scalar or vector) using the appropriate shift operator."""
        pass
    
    @abstractmethod
    def diff_x(self, q: Union[Vector, Matrix], order: int = 1) -> Union[Vector, Matrix]:
        """Compute the spatial derivative of the input q (vector or matrix) using the appropriate differentiation operator."""
        pass

class Grid1DUniformSpectral(Grid1D):
    """Child class for 1D periodic grid with equispaced grid points and spectral shift/interpolation/spatial derivative operators."""
    def __init__(self, Lx: float, nx: int):
        super().__init__(Lx)
        self.nx = nx
        if self.nx % 2 != 0:
            raise ValueError("nx must be even for spectral methods to avoid Nyquist mode issues.")
        self.x = np.linspace(0.0, Lx, nx, endpoint=False)
        self.dx = Lx / self.nx
        self.kx = 2 * np.pi * np.linspace(-self.nx // 2, self.nx // 2 - 1, self.nx, dtype=int, endpoint=True) / self.Lx
      
    def inner_product(self, q1: Union[Vector, Matrix], q2: Union[Vector, Matrix]) -> Union[float, Matrix]:
        """Compute the normalized inner product of q1 (vector or matrix) and q2 using the trapezoidal rule:
        <q1, q2> = 1/Lx * int_0^Lx q1(x) * q2(x) dx, approximated by sum_i q1[i] * q2[i] * dx / Lx = sum_i q1[i] * q2[i] / nx.
        Works for both 1D (returns scalar) and 2D (returns Gram-style matrix) inputs via numpy's matmul semantics.
        """
        return (q1.T @ q2) / self.nx

    def apply_sqrt_inner_product_mass(self, A: Union[Vector, Matrix], action: str = "forward") -> Union[Vector, Matrix]:
        """For the uniform spectral grid the inner-product mass is M = (1/nx) I, so the square root
        R satisfying M = R^T R is R = (1/sqrt(nx)) I (diagonal, hence R = R^T).
            forward, transpose:           A -> A / sqrt(nx)
            inverse, inverse_transpose:   A -> A * sqrt(nx)
        """
        if action == "forward" or action == "transpose":
            return A / np.sqrt(self.nx)
        elif action == "inverse" or action == "inverse_transpose":
            return A * np.sqrt(self.nx)
        else:
            raise ValueError(f"action must be 'forward', 'inverse', 'transpose', or 'inverse_transpose', got {action!r}")
        
    def shift_x(self, q: Union[Vector, Matrix], c: Union[float, Vector]) -> Union[Vector, Matrix]:
        """Shift the input q (vector or matrix) by c (scalar or vector) using the spectral shift operator:
        q_shifted(x) = T_c[q](x) = q(x - c) = F^{-1} (F(q) * exp(-1j * kx * c)), where F is the Fourier transform and kx are the wavenumbers."""
        kx = self.kx.reshape((-1,) + (1,) * (q.ndim - 1))   # (nx,) for 1D q, (nx, 1) for 2D q
        return ifft(fft(q) * np.exp(-1j * kx * c))
        
    def diff_x(self, q: Union[Vector, Matrix], order: int = 1) -> Union[Vector, Matrix]:
        """Compute the spatial derivative of the input q (vector or matrix) of given order using the spectral differentiation operator:
        d^order q / dx^order = F^{-1} ((1j * kx)^order * F(q))."""
        kx = self.kx.reshape((-1,) + (1,) * (q.ndim - 1))   # (nx,) for 1D q, (nx, 1) for 2D q
        return ifft(fft(q) * (1j * kx) ** order)

    def pad(self, q: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """3/2-rule padding (steps a-c). Returns q_pad at npad collocation points."""
        q_freq = fft(q)
        npad = (3 * self.nx) // 2
        pad_start = (npad - self.nx) // 2              # offset of the original N modes inside the padded array (fftshifted order)
        shape_pad = (npad,) + q.shape[1:]
        q_pad_freq = np.zeros(shape_pad, dtype=complex)
        q_pad_freq[pad_start:pad_start + self.nx] = q_freq
        return ifft(q_pad_freq)
    
    def truncate(self, q_padded: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """3/2-rule truncation (steps e-g). Maps an npad-point physical-space field back to N points
        by FFT'ing to npad-point Fourier, keeping only the central N modes [-N/2, N/2-1], and IFFT'ing
        to N-point physical. Modes |k| >= N/2 (which may contain aliased contributions on the npad
        grid) are discarded -- this is the step that actually kills the aliasing energy."""
        q_pad_freq = fft(q_padded)
        npad = q_pad_freq.shape[0]
        pad_start = (npad - self.nx) // 2
        q_freq = q_pad_freq[pad_start:pad_start + self.nx]
        return ifft(q_freq)

class Grid1DCubicSpline(Grid1D):
    """Child class for 1D periodic grid with not necessarily equispaced grid points and cubic-spline based shift/interpolation/spatial derivative operators."""
    def __init__(self, Lx: float, x: Vector):
        super().__init__(Lx)
        self.x = x
        self.nx = len(x)
        # Per-interval spacings (cyclic, non-uniform): h[j] = x[j+1] - x[j], with h[N-1] = (x[0]+Lx) - x[N-1].
        self.h = np.diff(np.append(x, x[0] + Lx))
        # Spline tridiag system A M = R F (cyclic, non-uniform aware). Precompute A, R, A_inv, and the composed M_operator.
        self.A_matrix = self._get_A_matrix(self.h)
        self.R_matrix = self._get_R_matrix(self.h)
        self.A_inv = sp.linalg.inv(self.A_matrix)
        self.M_operator = self.A_inv @ self.R_matrix                              # F -> M, dense (nx, nx)
        # First-derivative operator at nodes (closed-form spline S'(x_k)).
        self.D1_matrix = self._get_D1_matrix(self.h, self.M_operator)             # F -> S'(x_k), dense (nx, nx)
        # Exact Galerkin mass matrix W such that <f, g> = F^T W G.
        self.ip_mass_matrix = self._get_inner_product_mass_matrix(self.h, self.M_operator, self.Lx)
        # Square-root factor R (upper-triangular) of the inner-product mass: W = R^T R.
        # This matches the NiSRH-ROM convention; forward action is R @ A (no transpose).
        # scipy.linalg.cholesky(., lower=False) returns the upper-triangular factor directly.
        self.ip_mass_R = sp.linalg.cholesky(self.ip_mass_matrix, lower=False)     # upper-tri R, W = R^T R
        self.ip_mass_R_transpose = self.ip_mass_R.T                               # R^T, lower-triangular
        self.ip_mass_R_inv = sp.linalg.inv(self.ip_mass_R)                        # R^{-1}, dense (nx, nx)
        self.ip_mass_R_inv_transpose = self.ip_mass_R_inv.T                       # R^{-T}, dense (nx, nx)

    @staticmethod
    def _build_cyclic_tridiag(diag: Vector, off: Vector) -> Matrix:
        """Assemble a cyclic symmetric tridiagonal matrix T of size (N, N):
            T[j, j]           = diag[j]
            T[j, (j+1) % N]   = off[j]     (also placed at the transposed position)
        The wrap-around at j=N-1 contributes to the (N-1, 0) and (0, N-1) entries.
        """
        N = len(diag)
        T = np.diag(diag).astype(float)
        i = np.arange(N)
        j = (i + 1) % N
        T[i, j] += off
        T[j, i] += off
        return T

    def _get_A_matrix(self, h: Vector) -> Matrix:
        """Cyclic tridiag A on the LHS of the spline second-derivative system A M = R F:
            A[j, j-1] = h[j-1] / 6,
            A[j, j  ] = (h[j-1] + h[j]) / 3,
            A[j, j+1] = h[j] / 6.
        Derived from C^2 continuity of the periodic cubic spline at each node.
        """
        return self._build_cyclic_tridiag((np.roll(h, 1) + h) / 3, h / 6)

    def _get_R_matrix(self, h: Vector) -> Matrix:
        """Cyclic tridiag R on the RHS of the spline second-derivative system A M = R F:
            R[j, j-1] = 1 / h[j-1],
            R[j, j  ] = -(1 / h[j-1] + 1 / h[j]),
            R[j, j+1] = 1 / h[j].
        Together with A, it encodes the standard divided-difference RHS of the spline equations.
        """
        inv_h = 1.0 / h
        return self._build_cyclic_tridiag(-(np.roll(inv_h, 1) + inv_h), inv_h)

    def _get_D1_matrix(self, h: Vector, M_op: Matrix) -> Matrix:
        """First-derivative operator at nodes from the analytic spline derivative:
            (D1 q)[k] = S'(x_k) = (q[k+1] - q[k]) / h[k] - h[k]/3 * M[k] - h[k]/6 * M[k+1],
        where M = M_op @ q. Decomposes as D1 = E + C @ M_op where
            E[k, k] = -1/h[k],  E[k, k+1] = 1/h[k]      (forward difference)
            C[k, k] = -h[k]/3,  C[k, k+1] = -h[k]/6     (curvature correction).
        Wrap-around: index (k+1) is taken modulo nx. D1 is generally non-symmetric and dense.
        """
        N = len(h)
        k = np.arange(N)
        k_next = (k + 1) % N
        inv_h = 1.0 / h
        E = np.zeros((N, N))
        E[k, k] = -inv_h
        E[k, k_next] = inv_h
        C = np.zeros((N, N))
        C[k, k] = -h / 3
        C[k, k_next] = -h / 6
        return E + C @ M_op

    def _get_inner_product_mass_matrix(self, h: Vector, M_op: Matrix, Lx: float) -> Matrix:
        """Assemble the exact Galerkin mass matrix W such that
            <f, g> = F^T W G = (1/Lx) * int_0^Lx S_f(x) S_g(x) dx,
        where S_f, S_g are the periodic cubic spline interpolants of F, G.

        On each interval [x_k, x_{k+1}] of length h_k, the spline can be written in local coords
        xi = x - x_k, eta = h_k - xi, as
            S(xi) = f_k * (eta/h_k) + f_{k+1} * (xi/h_k)
                  + M_k     * (eta^3 - h_k^2 * eta) / (6 h_k)
                  + M_{k+1} * (xi^3  - h_k^2 * xi ) / (6 h_k).
        Analytical integration of the product of two such splines over the interval yields a
        4x4 local mass matrix in coords (f_k, f_{k+1}, M_k, M_{k+1}) with three blocks:
            LL (linear-linear):  diag h_k/3,      off h_k/6
            LC (linear-cubic):   diag -h_k^3/45,  off -7 h_k^3/360
            CC (cubic-cubic):    diag 2 h_k^5/945, off 31 h_k^5/15120.
        Summed cyclically over intervals, these give global tridiags A (=LL), L (=LC), B (=CC).
        Substituting M = M_op @ F (where M_op = A_inv @ R) gives
            W = (A + L M_op + M_op^T L + M_op^T B M_op) / Lx.
        W is symmetric (each summand is symmetric or appears with its transpose) and SPD (Gram
        matrix of the L^2 inner product on the cubic-spline interpolant subspace).
        """
        # LL block: identical structure to the spline LHS A.
        A = self._build_cyclic_tridiag((np.roll(h, 1) + h) / 3, h / 6)
        # LC (cross) block.
        h3 = h ** 3
        L = self._build_cyclic_tridiag(-(np.roll(h3, 1) + h3) / 45, -7 * h3 / 360)
        # CC block.
        h5 = h ** 5
        B = self._build_cyclic_tridiag(2 * (np.roll(h5, 1) + h5) / 945, 31 * h5 / 15120)
        # Assemble.
        W = A + L @ M_op + M_op.T @ L + M_op.T @ B @ M_op
        return W / Lx

    def inner_product(self, q1: Union[Vector, Matrix], q2: Union[Vector, Matrix]) -> Union[float, Matrix]:
        """Compute the normalized inner product of q1 (vector or matrix) and q2 using the spline-based Galerkin mass matrix:
        <q1, q2> = q1^T W q2 = (1/Lx) * int_0^Lx S_{q1}(x) S_{q2}(x) dx.
        Works for both 1D (returns scalar) and 2D (returns Gram-style matrix) inputs.
        """
        return q1.T @ self.ip_mass_matrix @ q2

    def apply_sqrt_inner_product_mass(self, A: Union[Vector, Matrix], action: str = "forward") -> Union[Vector, Matrix]:
        """Apply R (or its transpose / inverse / inverse-transpose) where R is the upper-triangular
        Cholesky factor of W satisfying W = R^T R. Then ||v||_W^2 = v^T R^T R v = ||R v||_2^2, so
        the forward action R @ . maps W-orthogonality into standard l2-orthogonality. This matches
        the NiSRH-ROM convention.
            forward:           R    @ A
            inverse:           R_inv @ A      (= R^{-1} @ A)
            transpose:         R.T  @ A
            inverse_transpose: R_inv.T @ A    (= R^{-T} @ A)
        Works for both 1D (Vector) and 2D (Matrix) inputs.
        """
        if action == "forward":
            return self.ip_mass_R @ A
        elif action == "inverse":
            return self.ip_mass_R_inv @ A
        elif action == "transpose":
            return self.ip_mass_R_transpose @ A
        elif action == "inverse_transpose":
            return self.ip_mass_R_inv_transpose @ A
        else:
            raise ValueError(f"action must be 'forward', 'inverse', 'transpose', or 'inverse_transpose', got {action!r}")

    def shift_x(self, q: Union[Vector, Matrix], c: Union[float, Vector]) -> Union[Vector, Matrix]:
        """Shift the input q (vector or matrix) by c (scalar or vector) using the spline-based shift operator:
            q_shifted(x_j) = T_c[q](x_j) = S_q(x_j - c)  (with periodic wrap of the argument),
        where S_q is the periodic cubic spline interpolant of q.

        Supports three input shape combinations:
            (1) q.ndim=1 (nx,) and c scalar             -> result (nx,)
            (2) q.ndim=2 (nx, nt) and c scalar          -> result (nx, nt), same shift on every column
            (3) q.ndim=2 (nx, nt) and c.shape=(nt,)     -> result (nx, nt), per-column shifts

        Algorithm: for each target point (x_j - c) mod Lx,
            (i) locate the host interval k via searchsorted on self.x,
            (ii) evaluate the analytic cubic spline polynomial on that interval.
        All steps are vectorized; no Python-level loop over nodes or columns.
        """
        M = self.M_operator @ q                                                   # spline 2nd-derivatives at nodes; same shape as q

        c_arr = np.asarray(c)
        if c_arr.ndim == 0:
            target_x = (self.x - c_arr) % self.Lx                                 # (nx,)
        else:
            target_x = (self.x[:, None] - c_arr[None, :]) % self.Lx               # (nx, nt)

        # Locate host interval k for each target.
        k = np.searchsorted(self.x, target_x, side='right') - 1                   # shape matches target_x
        k_next = (k + 1) % self.nx

        h_local = self.h[k]
        xi = target_x - self.x[k]                                                  # local coord in [0, h_local)
        eta = h_local - xi

        # Gather (q, M) values at the endpoints of each target's host interval.
        if k.ndim == 1:
            f_left = q[k]
            f_right = q[k_next]
            M_left = M[k]
            M_right = M[k_next]
            if q.ndim == 2:
                # broadcast 1D xi/eta/h_local against 2D f_left/f_right/M_left/M_right
                xi = xi[:, None]
                eta = eta[:, None]
                h_local = h_local[:, None]
        else:  # k.ndim == 2 (per-column shifts), q must be 2D
            f_left = np.take_along_axis(q, k, axis=0)
            f_right = np.take_along_axis(q, k_next, axis=0)
            M_left = np.take_along_axis(M, k, axis=0)
            M_right = np.take_along_axis(M, k_next, axis=0)

        # Analytic cubic spline polynomial on the host interval.
        h2 = h_local * h_local
        return ((f_left * eta + f_right * xi) / h_local
                + M_left  * (eta * eta * eta - h2 * eta) / (6.0 * h_local)
                + M_right * (xi  * xi  * xi  - h2 * xi ) / (6.0 * h_local))

    def diff_x(self, q: Union[Vector, Matrix], order: int = 1) -> Union[Vector, Matrix]:
        """Compute the spatial derivative of q of given non-negative order using option-B dispatch:
            order = 0: identity (returns q).
            order = 1: D1 @ q  (analytic spline S'(x_k), 4th-order accurate on uniform smooth periodic data).
            order = 2: M_op @ q  (analytic spline S''(x_k) = M, 2nd-order accurate at nodes).
            order >= 3: D1^(order-2) @ (M_op @ q)  (repeated 1st-derivative composition on top of M).
        For SR-OpInf, only order in {1, 2} is load-bearing (phase condition / diagnostics). Higher
        orders are supported for API completeness but lose accuracy with each extra D1 composition
        and should not be relied upon for $u^{(\\geq 3)}$ in production paths --- use the spectral
        grid (Grid1DUniformSpectral) for those.
        """
        if order < 0:
            raise ValueError(f"diff_x order must be a non-negative integer, got {order!r}")
        if order == 1:
            return self.D1_matrix @ q
        result = self.M_operator @ q # order = 2 starting point
        for _ in range(order - 2):
            result = self.D1_matrix @ result
        return result
