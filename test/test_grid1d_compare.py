"""test_grid1d_compare.py --- pytest suite contrasting Grid1DCubicSpline vs Grid1DUniformSpectral
on the same uniform grid.

Premise: when the cubic spline grid is given a uniform node placement, all of
(inner_product, diff_x, shift) must converge to the corresponding spectral result as nx grows.
The spectral baseline is exact (to machine precision) on smooth band-limited periodic data,
so |spline - spectral| at uniform nx is essentially the spline-induced discretization error,
which follows the same O(h^alpha) law as the standalone spline tests verify.

Expected alpha (uniform grid, band-limited smooth periodic test function with modes k=1, 2):
    inner_product             alpha = 4
    diff_x order = 1          alpha = 4   (nodal superconvergence on uniform)
    diff_x order = 2 (M_op)   alpha = 2
    shift                     alpha = 4

Slope tolerance: ±0.5.

Run:
    cd SROpInf && python3 -m pytest test/test_grid1d_compare.py -v
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from SROpInf.grids.grid1d import Grid1DCubicSpline, Grid1DUniformSpectral  # noqa: E402


PI = np.pi
LX = 2 * PI


def _u_ref(x, Lx=LX):
    """Band-limited test signal: modes k=1 and k=2, well within Nyquist for nx >= 8."""
    a = 2 * PI / Lx
    b = 4 * PI / Lx
    return np.cos(a * x) + 0.3 * np.sin(b * x)


def _u_ref_deriv(x, order, Lx=LX):
    a = 2 * PI / Lx
    b = 4 * PI / Lx
    return a**order * np.cos(a * x + order * PI / 2) \
         + 0.3 * b**order * np.sin(b * x + order * PI / 2)


_U_IP_EXACT = 0.5 * 1.0**2 + 0.5 * 0.3**2  # = 0.545


def _loglog_slope(nxs, errs):
    nxs = np.asarray(nxs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    keep = errs > 0
    if keep.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0 / nxs[keep]), np.log(errs[keep]), 1)
    return slope


# ---------------------------------------------------------------------------
# Spectral baseline sanity (precondition for the comparison to be meaningful)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spectral_grid64():
    return Grid1DUniformSpectral(LX, 64)


def test_spectral_inner_product_is_exact(spectral_grid64):
    u = _u_ref(spectral_grid64.x)
    assert abs(spectral_grid64.inner_product(u, u) - _U_IP_EXACT) < 1e-13


def test_spectral_diff_x_order1_is_exact(spectral_grid64):
    u = _u_ref(spectral_grid64.x)
    err = np.linalg.norm(spectral_grid64.diff_x(u, 1) - _u_ref_deriv(spectral_grid64.x, 1))
    assert err < 1e-12


def test_spectral_diff_x_order2_is_exact(spectral_grid64):
    u = _u_ref(spectral_grid64.x)
    err = np.linalg.norm(spectral_grid64.diff_x(u, 2) - _u_ref_deriv(spectral_grid64.x, 2))
    assert err < 1e-11


def test_spectral_shift_is_exact(spectral_grid64):
    u = _u_ref(spectral_grid64.x)
    c = 0.37
    err = np.linalg.norm(spectral_grid64.shift_x(u, c) - _u_ref(spectral_grid64.x - c))
    assert err < 1e-12


# ---------------------------------------------------------------------------
# Batched 2D consistency at one fixed nx (snapshot-matrix style usage)
# ---------------------------------------------------------------------------
def test_batched_diff_x_close_at_nx_64():
    nx = 64
    x = np.linspace(0.0, LX, nx, endpoint=False)
    g_sp = Grid1DUniformSpectral(LX, nx)
    g_cs = Grid1DCubicSpline(LX, x)

    nt = 4
    rng = np.random.default_rng(42)
    a = 2 * PI / LX
    coeffs = rng.normal(size=(3, nt))
    U = sum(coeffs[k - 1] * np.cos(k * a * x[:, None] + 0.3 * (k - 1)) for k in range(1, 4))
    diff = np.linalg.norm(g_cs.diff_x(U, 1) - g_sp.diff_x(U, 1)) / np.sqrt(nx * nt)
    # band-limited modes 1..3, spline O(h^4) at nx=64: error ~ (1/64)^4 * (some const) << 1
    assert diff < 1e-2


def test_batched_shift_with_per_column_c_close_at_nx_64():
    nx = 64
    x = np.linspace(0.0, LX, nx, endpoint=False)
    g_sp = Grid1DUniformSpectral(LX, nx)
    g_cs = Grid1DCubicSpline(LX, x)

    nt = 4
    rng = np.random.default_rng(42)
    a = 2 * PI / LX
    coeffs = rng.normal(size=(3, nt))
    U = sum(coeffs[k - 1] * np.cos(k * a * x[:, None] + 0.3 * (k - 1)) for k in range(1, 4))
    c_vec = np.array([0.10, 0.25, -0.30, 0.50])
    diff = np.linalg.norm(g_cs.shift_x(U, c_vec) - g_sp.shift_x(U, c_vec)) / np.sqrt(nx * nt)
    assert diff < 1e-3


# ---------------------------------------------------------------------------
# Convergence of |spline - spectral|
# ---------------------------------------------------------------------------
_NXS = (16, 32, 64, 128, 256, 512)
_SLOPE_TOL = 0.5


def _collect_diff_errors(op, c=0.37):
    """For each nx, return ||cs.op(u) - sp.op(u)||_2 / sqrt(nx) (or |scalar diff| for ip)."""
    errs = []
    for nx in _NXS:
        x = np.linspace(0.0, LX, nx, endpoint=False)
        g_sp = Grid1DUniformSpectral(LX, len(x))
        g_cs = Grid1DCubicSpline(LX, x)
        u = _u_ref(x)
        if op == "ip":
            errs.append(abs(g_cs.inner_product(u, u) - g_sp.inner_product(u, u)))
        elif op == "d1":
            errs.append(float(np.linalg.norm(g_cs.diff_x(u, 1) - g_sp.diff_x(u, 1)) / np.sqrt(nx)))
        elif op == "d2":
            errs.append(float(np.linalg.norm(g_cs.diff_x(u, 2) - g_sp.diff_x(u, 2)) / np.sqrt(nx)))
        elif op == "shift":
            errs.append(float(np.linalg.norm(g_cs.shift_x(u, c) - g_sp.shift_x(u, c)) / np.sqrt(nx)))
        else:
            raise ValueError(op)
    return errs


def test_compare_convergence_inner_product():
    errs = _collect_diff_errors("ip")
    alpha = _loglog_slope(_NXS, errs)
    assert abs(alpha - 4.0) <= _SLOPE_TOL, \
        f"|cs-sp| inner_product: expected alpha~4, got {alpha:.2f}, errs={errs}"


def test_compare_convergence_diff_x_order1():
    errs = _collect_diff_errors("d1")
    alpha = _loglog_slope(_NXS, errs)
    assert abs(alpha - 4.0) <= _SLOPE_TOL, \
        f"|cs-sp| diff_x order=1: expected alpha~4, got {alpha:.2f}, errs={errs}"


def test_compare_convergence_diff_x_order2():
    errs = _collect_diff_errors("d2")
    alpha = _loglog_slope(_NXS, errs)
    assert abs(alpha - 2.0) <= _SLOPE_TOL, \
        f"|cs-sp| diff_x order=2: expected alpha~2, got {alpha:.2f}, errs={errs}"


def test_compare_convergence_shift():
    errs = _collect_diff_errors("shift")
    alpha = _loglog_slope(_NXS, errs)
    assert abs(alpha - 4.0) <= _SLOPE_TOL, \
        f"|cs-sp| shift: expected alpha~4, got {alpha:.2f}, errs={errs}"
