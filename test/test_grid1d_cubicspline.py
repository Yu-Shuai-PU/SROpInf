"""test_grid1d_cubicspline.py --- pytest suite for Grid1DCubicSpline.

Verifies:
    Structural sanity (uniform + non-uniform grid):
        - W is symmetric, SPD
        - M_op @ 1 = 0  (spline of constant has zero curvature)
        - D1 @ 1   = 0  (spline derivative of constant is zero)
        - <1, 1>_W = 1  exactly

    2D batched consistency:
        - shift(Q, c_vec) column-wise equals shift(Q[:,t], c_vec[t])

    Convergence (smooth periodic test signal cos + 0.3 sin) — slope tolerance ±0.5:
        operator             uniform alpha    non-uniform alpha    reason
        inner_product (W)    4                4                    Galerkin mass on spline interpolant
        diff_x order=1       4                3                    nodal superconvergence only on uniform
        diff_x order=2       2                2                    natural spline 2nd derivative
        shift                4                4                    spline interpolation error at shifted points

Run:
    cd SROpInf && python3 -m pytest test/test_grid1d_cubicspline.py -v
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from SROpInf.grids.grid1d import Grid1DCubicSpline  # noqa: E402


PI = np.pi
LX = 2 * PI

# Reference signal: u(x) = cos(2pi x/L) + 0.3 sin(4pi x/L)
def _u_ref(x, Lx=LX):
    a = 2 * PI / Lx
    b = 4 * PI / Lx
    return np.cos(a * x) + 0.3 * np.sin(b * x)


def _u_ref_deriv(x, order, Lx=LX):
    a = 2 * PI / Lx
    b = 4 * PI / Lx
    return a**order * np.cos(a * x + order * PI / 2) \
         + 0.3 * b**order * np.sin(b * x + order * PI / 2)


# Exact <u, u> = (1/L) int u^2 dx. Cross terms vanish by orthogonality of distinct trig modes.
_U_IP_EXACT = 0.5 * 1.0**2 + 0.5 * 0.3**2


# ---------------------------------------------------------------------------
# Grid factories
# ---------------------------------------------------------------------------
def _make_uniform_grid(nx, Lx=LX):
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    return Grid1DCubicSpline(Lx, x)


def _make_nonuniform_grid(nx, Lx=LX, seed=0, ratio=2.0):
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(1.0, ratio, nx)
    gaps *= Lx / gaps.sum()
    x = np.concatenate([[0.0], np.cumsum(gaps[:-1])])
    return Grid1DCubicSpline(Lx, x)


_FACTORY = {"uniform": _make_uniform_grid, "non-uniform": _make_nonuniform_grid}


# ---------------------------------------------------------------------------
# Log-log slope helper
# ---------------------------------------------------------------------------
def _loglog_slope(nxs, errs):
    nxs = np.asarray(nxs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    keep = errs > 0
    if keep.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0 / nxs[keep]), np.log(errs[keep]), 1)
    return slope


# ---------------------------------------------------------------------------
# Structural tests (parametrized over grid type)
# ---------------------------------------------------------------------------
@pytest.fixture(params=["uniform", "non-uniform"])
def grid64(request):
    return _FACTORY[request.param](64)


def test_W_symmetric(grid64):
    W = grid64.ip_mass_matrix
    assert np.allclose(W, W.T)


def test_W_spd(grid64):
    W = grid64.ip_mass_matrix
    assert np.linalg.eigvalsh(W).min() > 0


def test_M_operator_kills_constant(grid64):
    """The spline second derivative of a constant function is identically zero."""
    M_one = grid64.M_operator @ np.ones(grid64.nx)
    assert np.linalg.norm(M_one) < 1e-10


def test_D1_kills_constant(grid64):
    """The spline first derivative of a constant function is identically zero."""
    D1_one = grid64.D1_matrix @ np.ones(grid64.nx)
    assert np.linalg.norm(D1_one) < 1e-10


def test_constant_inner_product_unity(grid64):
    """<1, 1>_W must equal 1 exactly (normalized L2 inner product of constant)."""
    ones = np.ones(grid64.nx)
    assert np.isclose(grid64.inner_product(ones, ones), 1.0)


def test_batched_shift_matches_columnwise(grid64):
    """grid.shift_x(Q, c_vec)[:, t] must equal grid.shift_x(Q[:, t], c_vec[t])."""
    nx = grid64.nx
    rng = np.random.default_rng(42)
    nt = 4
    Q = rng.standard_normal((nx, nt))
    c_vec = np.array([0.1, 0.5, -0.3, 0.7])
    batched = grid64.shift_x(Q, c_vec)
    for t in range(nt):
        col = grid64.shift_x(Q[:, t], c_vec[t])
        assert np.allclose(batched[:, t], col, atol=1e-12), f"column {t}"


# ---------------------------------------------------------------------------
# Convergence tests (parametrized over grid type + operator)
# ---------------------------------------------------------------------------
_NXS = (16, 32, 64, 128, 256, 512)
_SLOPE_TOL = 0.5  # |empirical - expected| <= 0.5 is the accepted band

# Expected alpha[grid_kind][operator]
_EXPECTED_ALPHA = {
    "uniform":     {"ip": 4.0, "d1": 4.0, "d2": 2.0, "shift": 4.0},
    "non-uniform": {"ip": 4.0, "d1": 3.0, "d2": 2.0, "shift": 4.0},  # d1 drops by 1 on non-uniform
}


def _collect_errors(grid_kind, op, c=0.37):
    """Return list of errors at each nx in _NXS, for the given operator."""
    factory = _FACTORY[grid_kind]
    errs = []
    for nx in _NXS:
        g = factory(nx)
        u = _u_ref(g.x)
        if op == "ip":
            errs.append(abs(g.inner_product(u, u) - _U_IP_EXACT))
        elif op == "d1":
            errs.append(float(np.linalg.norm(g.diff_x(u, 1) - _u_ref_deriv(g.x, 1)) / np.sqrt(nx)))
        elif op == "d2":
            errs.append(float(np.linalg.norm(g.diff_x(u, 2) - _u_ref_deriv(g.x, 2)) / np.sqrt(nx)))
        elif op == "shift":
            errs.append(float(np.linalg.norm(g.shift_x(u, c) - _u_ref(g.x - c)) / np.sqrt(nx)))
        else:
            raise ValueError(op)
    return errs


@pytest.mark.parametrize("grid_kind", ["uniform", "non-uniform"])
def test_convergence_inner_product(grid_kind):
    expected = _EXPECTED_ALPHA[grid_kind]["ip"]
    errs = _collect_errors(grid_kind, "ip")
    alpha = _loglog_slope(_NXS, errs)
    assert abs(alpha - expected) <= _SLOPE_TOL, \
        f"inner_product on {grid_kind}: expected alpha~{expected}, got {alpha:.2f}, errs={errs}"


@pytest.mark.parametrize("grid_kind", ["uniform", "non-uniform"])
def test_convergence_diff_x_order1(grid_kind):
    expected = _EXPECTED_ALPHA[grid_kind]["d1"]
    errs = _collect_errors(grid_kind, "d1")
    alpha = _loglog_slope(_NXS, errs)
    assert abs(alpha - expected) <= _SLOPE_TOL, \
        f"diff_x order=1 on {grid_kind}: expected alpha~{expected}, got {alpha:.2f}, errs={errs}"


@pytest.mark.parametrize("grid_kind", ["uniform", "non-uniform"])
def test_convergence_diff_x_order2(grid_kind):
    expected = _EXPECTED_ALPHA[grid_kind]["d2"]
    errs = _collect_errors(grid_kind, "d2")
    alpha = _loglog_slope(_NXS, errs)
    assert abs(alpha - expected) <= _SLOPE_TOL, \
        f"diff_x order=2 on {grid_kind}: expected alpha~{expected}, got {alpha:.2f}, errs={errs}"


@pytest.mark.parametrize("grid_kind", ["uniform", "non-uniform"])
def test_convergence_shift(grid_kind):
    expected = _EXPECTED_ALPHA[grid_kind]["shift"]
    errs = _collect_errors(grid_kind, "shift")
    alpha = _loglog_slope(_NXS, errs)
    assert abs(alpha - expected) <= _SLOPE_TOL, \
        f"shift on {grid_kind}: expected alpha~{expected}, got {alpha:.2f}, errs={errs}"
