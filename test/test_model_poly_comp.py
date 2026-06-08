"""test_model_poly_comp.py --- pytest suite verifying FOM + SR-FOM + SR-ROM correctness
across all polynomial-term combinations.

Design note (post-refactor): the polynomial operators live on a plain `FullOrderModel`
(`base_fom`); a `SymmetryReducedScaledFullOrderModel` wraps that base FOM and adds the
symmetry-reduction terms. So the FOM-side checks below target the base FOM, while the SR-side
checks target the SR-FOM that wraps it.

For each poly_comp in {[0], [1], [2], [0,1], [0,2], [1,2], [1,2,3]}, this file checks:

    FOM side (FullOrderModel)
      - linear(q)        : returns zero if 1 not in poly_operators (no KeyError)
      - nonlinear(q)     : iterates poly_operators.items() (no KeyError on non-consecutive
                           keys), calls the k=0 op as a function
      - rhs(q)           : finite, equals linear(q) + nonlinear(q)

    SR side (SymmetryReducedScaledFullOrderModel)
      - rhs(Rq)          : finite, no infinite recursion, equals
                           rhs_poly + shift_speed * advection
      - shift_speed_numer(Rq) : finite scalar, equals -<q_template_dx_scaled, rhs_poly(Rq)>

    ROM side (SymmetryReducedScaledReducedOrderModel)
      - SR-FOM.project(poly_comp, phi) : runs to completion; tensor shapes match the
                                         (k+1)-rank for RHS and k-rank for shift_numer
      - rom.rhs_z(z) / rom.shift_speed_numer(z) / rom.rhs_zc(t, zc) : finite, correct shapes

Each combination targets one of the four bugs fixed in the model.py refactor:
    - poly_comp = [0]     exercises the k=0 callable convention
    - poly_comp = [2]     exercises the linear()=0 fallback (no key 1)
    - poly_comp = [0, 1]  exercises the non-consecutive-keys iteration in nonlinear()
    - poly_comp = [1]     would have hit the shift_speed_numer recursion bug pre-fix

Run from repo root:
    cd SROpInf && python3 -m pytest test/test_model_poly_comp.py -v
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from SROpInf.grids.grid1d import Grid1DUniformSpectral  # noqa: E402
from SROpInf.models.model import (  # noqa: E402
    FullOrderModel,
    SymmetryReducedScaledFullOrderModel,
)


PI = np.pi
NX = 16
LX = 2 * PI


# ---------------------------------------------------------------------------
# Deterministic test operators (so we can do exact numerical cross-checks).
# ---------------------------------------------------------------------------
# Fixed seed so every test run is reproducible.
_RNG = np.random.default_rng(0)
_A_LIN = _RNG.standard_normal((NX, NX)) * 0.1
_F0_VEC = _RNG.standard_normal(NX) * 0.3
_TEMPLATE_DX = _RNG.standard_normal(NX)
_TEMPLATE_DXX = _RNG.standard_normal(NX)


def _F0():
    """Constant (k=0) operator: takes no args, returns a fixed vector."""
    return _F0_VEC.copy()


def _F1(q):
    """Linear (k=1) operator: A_lin @ q. Supports 1D and 2D q via numpy matmul."""
    return _A_LIN @ q


def _F2(q1, q2):
    """Quadratic (k=2) operator: element-wise q1 * q2 * 0.1 (symmetric)."""
    return q1 * q2 * 0.1


def _F3(q1, q2, q3):
    """Cubic (k=3) operator: element-wise q1 * q2 * q3 * 0.01 (symmetric)."""
    return q1 * q2 * q3 * 0.01


_POLY_OP_LIBRARY = {0: _F0, 1: _F1, 2: _F2, 3: _F3}


# ---------------------------------------------------------------------------
# Minimal concrete FOM / SR-FOM builders.
# ---------------------------------------------------------------------------
class _DummyFOM(FullOrderModel):
    """Concrete FullOrderModel populated with a chosen subset of poly_operators."""

    def __init__(self, grid, poly_ops):
        # poly_operators must be set BEFORE super().__init__() per the project idiom.
        self.poly_operators = poly_ops
        super().__init__(grid)


def _make_fom(grid, poly_comp):
    """Plain base FOM carrying the requested polynomial operators."""
    poly_ops = {k: _POLY_OP_LIBRARY[k] for k in poly_comp}
    return _DummyFOM(grid, poly_ops)


def _make_sr_fom(grid, poly_comp):
    """Symmetry-reduced, scaled FOM wrapping a base FOM with the requested poly operators."""
    base = _make_fom(grid, poly_comp)
    return SymmetryReducedScaledFullOrderModel(
        grid,
        base_fom=base,
        q_template_dx_scaled=_TEMPLATE_DX,
        q_template_dxx_scaled=_TEMPLATE_DXX,
    )


@pytest.fixture(scope="module")
def grid():
    return Grid1DUniformSpectral(LX, NX)


# ---------------------------------------------------------------------------
# Test inputs: state q, scaled state Rq, latent state z, etc. Deterministic.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def state_q(grid):
    return _RNG.standard_normal(NX) * 0.1


@pytest.fixture(scope="module")
def state_Rq(grid, state_q):
    return grid.apply_sqrt_inner_product_mass(state_q, action="forward")


@pytest.fixture(scope="module")
def phi_basis(grid):
    r = 3
    return _RNG.standard_normal((NX, r)) * 0.1


@pytest.fixture(scope="module")
def latent_z():
    r = 3
    return _RNG.standard_normal(r) * 0.1


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------
POLY_COMP_CASES = [
    [0],
    [1],
    [2],
    [0, 1],
    [0, 2],
    [1, 2],
    [1, 2, 3],
]


def _case_id(poly_comp):
    # Underscores keep the case ID valid as a Python identifier so `pytest -k <id>` works.
    return "k_" + "_".join(str(k) for k in poly_comp)


# ---------------------------------------------------------------------------
# FOM-side tests (FullOrderModel)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_fom_linear_zero_fallback(grid, poly_comp, state_q):
    """linear(q) must return zeros when poly_operators[1] is absent, and equal F1(q) when present."""
    fom = _make_fom(grid, poly_comp)
    lin_q = fom.linear(state_q)
    assert lin_q.shape == state_q.shape
    if 1 in poly_comp:
        assert np.allclose(lin_q, _F1(state_q))
    else:
        assert np.allclose(lin_q, 0.0), \
            f"linear() should be zero when 1 not in poly_operators, got ||.||={np.linalg.norm(lin_q):.3e}"


@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_fom_nonlinear_handles_nonconsecutive_keys(grid, poly_comp, state_q):
    """nonlinear(q) must iterate over poly_operators.items() (not range(2, N+1)),
    so non-consecutive keys (e.g. {0, 1} or {0, 2}) do not raise KeyError.
    """
    fom = _make_fom(grid, poly_comp)
    nl_q = fom.nonlinear(state_q)
    assert nl_q.shape == state_q.shape
    assert np.all(np.isfinite(nl_q))

    # Cross-check: nonlinear == sum_{k>=2} F_k(q,...,q) + (F0() if 0 in poly_ops else 0).
    expected = np.zeros_like(state_q)
    if 0 in poly_comp:
        expected = expected + _F0()
    if 2 in poly_comp:
        expected = expected + _F2(state_q, state_q)
    if 3 in poly_comp:
        expected = expected + _F3(state_q, state_q, state_q)
    assert np.allclose(nl_q, expected), f"nonlinear() value mismatch for poly_comp={poly_comp}"


@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_fom_rhs_finite_no_recursion(grid, poly_comp, state_q):
    """FullOrderModel.rhs(q) must be finite and equal linear(q) + nonlinear(q)."""
    fom = _make_fom(grid, poly_comp)
    rhs_q = fom.rhs(state_q)
    assert rhs_q.shape == state_q.shape
    assert np.all(np.isfinite(rhs_q))
    assert np.allclose(rhs_q, fom.linear(state_q) + fom.nonlinear(state_q))


# ---------------------------------------------------------------------------
# SR-FOM-side tests (SymmetryReducedScaledFullOrderModel)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_srfom_rhs_finite_no_recursion(grid, poly_comp, state_Rq):
    """SR-FOM.rhs(Rq) must complete (no RecursionError, no KeyError) and be finite,
    for any poly_comp combination including non-consecutive keys.
    """
    sr = _make_sr_fom(grid, poly_comp)
    # If recursion bug were back, this would RecursionError before assertions are reached.
    rhs_Rq = sr.rhs(state_Rq)
    assert rhs_Rq.shape == state_Rq.shape
    assert np.all(np.isfinite(rhs_Rq))


@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_srfom_rhs_equals_rhspoly_plus_shift_advection(grid, poly_comp, state_Rq):
    """SR-FOM.rhs(Rq) must equal rhs_poly(Rq) + shift_speed(Rq) * advection(Rq) (its definition)."""
    sr = _make_sr_fom(grid, poly_comp)
    rhs_Rq = sr.rhs(state_Rq)
    rhs_poly = sr.rhs_poly(state_Rq)
    shift = sr.shift_speed(state_Rq)
    adv = sr.advection(state_Rq)
    assert np.allclose(rhs_Rq, rhs_poly + shift * adv), \
        f"sr.rhs decomposition broken for poly_comp={poly_comp}"


@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_srfom_shift_speed_numer_scalar_finite(grid, poly_comp, state_Rq):
    """shift_speed_numer must be a finite scalar and equal -<template_dx_scaled, rhs_poly(Rq)>."""
    sr = _make_sr_fom(grid, poly_comp)
    s_val = float(sr.shift_speed_numer(state_Rq))
    assert np.isfinite(s_val)
    expected = -np.dot(_TEMPLATE_DX, sr.rhs_poly(state_Rq))
    assert np.isclose(s_val, float(expected)), \
        f"shift_speed_numer != -<template_dx, rhs_poly(Rq)> for poly_comp={poly_comp}"


# ---------------------------------------------------------------------------
# ROM-side tests (project + downstream)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_project_tensor_shapes(grid, poly_comp, phi_basis):
    """project() must run and produce tensors with shapes:
       - rhs_tensor[i]:   (r,)*(k_i+1)   for each k_i in poly_comp
       - shift_tensor[i]: (r,)*k_i       for each k_i in poly_comp
       - tensors[-2]:     (r,)           (dcdt_denom_scaled)
       - tensors[-1]:     (r, r)         (advection_scaled, phi^T D phi)
    """
    sr = _make_sr_fom(grid, poly_comp)
    rom = sr.project(poly_comp, phi_basis)
    r = phi_basis.shape[1]
    n = len(poly_comp)

    assert len(rom.tensors) == 2 * n + 2, \
        f"expected {2*n+2} tensors, got {len(rom.tensors)} for poly_comp={poly_comp}"

    for i, k in enumerate(poly_comp):
        rhs_tensor = rom.tensors[i]
        shift_tensor = rom.tensors[n + i]
        assert rhs_tensor.shape == (r,) * (k + 1), \
            f"rhs tensor[i={i}, k={k}] shape mismatch: {rhs_tensor.shape}"
        assert shift_tensor.shape == (r,) * k, \
            f"shift tensor[i={i}, k={k}] shape mismatch: {shift_tensor.shape}"

    assert rom.tensors[-2].shape == (r,)        # dcdt_denom_scaled
    assert rom.tensors[-1].shape == (r, r)      # advection_scaled


@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_rom_rhs_z_finite(grid, poly_comp, phi_basis, latent_z):
    """rom.rhs_z(z) is finite with shape (r,)."""
    sr = _make_sr_fom(grid, poly_comp)
    rom = sr.project(poly_comp, phi_basis)
    dz = rom.rhs_z(latent_z)
    r = phi_basis.shape[1]
    assert dz.shape == (r,)
    assert np.all(np.isfinite(dz))


@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_rom_shift_speed_numer_scalar(grid, poly_comp, phi_basis, latent_z):
    """rom.shift_speed_numer(z) must be a finite Python float."""
    sr = _make_sr_fom(grid, poly_comp)
    rom = sr.project(poly_comp, phi_basis)
    s = rom.shift_speed_numer(latent_z)
    assert isinstance(s, float)
    assert np.isfinite(s)


@pytest.mark.parametrize("poly_comp", POLY_COMP_CASES, ids=_case_id)
def test_rom_rhs_zc_finite(grid, poly_comp, phi_basis, latent_z):
    """rom.rhs_zc(t, zc) is finite with shape (r+1,) (the joint [dz/dt, dc/dt] vector)."""
    sr = _make_sr_fom(grid, poly_comp)
    rom = sr.project(poly_comp, phi_basis)
    r = phi_basis.shape[1]
    zc = np.append(latent_z, 0.0)
    dzc = rom.rhs_zc(0.0, zc)
    assert dzc.shape == (r + 1,)
    assert np.all(np.isfinite(dzc))


# ---------------------------------------------------------------------------
# Targeted regression tests: each one corresponds to one of the four bugs
# fixed in the refactor, named so a failure tells you which bug regressed.
# ---------------------------------------------------------------------------
def test_bug1_no_recursion_when_only_linear(grid, state_Rq):
    """Regression: shift_speed_numer used to call back into SR-FOM.rhs, causing infinite
    recursion. Trigger condition: poly_operators[1] exists.
    """
    sr = _make_sr_fom(grid, [1])
    sr.rhs(state_Rq)  # would RecursionError if Bug 1 returned
    sr.shift_speed_numer(state_Rq)


def test_bug2_nonconsecutive_keys_zero_and_one(grid, state_q):
    """Regression: nonlinear() iterated range(2, num_poly_terms+1) and crashed with KeyError
    when poly_operators keys were {0, 1} (range tries k=2 which is absent).
    """
    fom = _make_fom(grid, [0, 1])
    nl_q = fom.nonlinear(state_q)
    # Should equal F0() (constant), since no k>=2 op exists.
    assert np.allclose(nl_q, _F0())


def test_bug3_k0_called_as_function(grid, state_q):
    """Regression: nonlinear() used to do `nonlinear_q += self.poly_operators[0]`, treating
    the entry as a vector. The convention is that poly_operators[0] is a CALLABLE returning a
    vector; nonlinear() must invoke it.
    """
    fom = _make_fom(grid, [0])
    nl_q = fom.nonlinear(state_q)
    assert np.allclose(nl_q, _F0_VEC)


def test_bug4_linear_returns_zero_when_no_key_1(grid, state_q):
    """Regression: linear() used to KeyError when poly_operators[1] was absent. Now it should
    return zeros so that purely nonlinear systems (poly_comp without 1) can still compose rhs.
    """
    fom = _make_fom(grid, [0, 2])
    lin_q = fom.linear(state_q)
    assert np.allclose(lin_q, 0.0)
    # And rhs(q) should still be finite (no KeyError from linear).
    assert np.all(np.isfinite(fom.rhs(state_q)))
