"""test_grid1d_demo.py --- side-by-side comparison of Grid1DUniformSpectral and
Grid1DCubicSpline on three illustrative periodic test functions of increasing roughness:

    1. cossin:  f(x) = cos(2 pi x/L) * sin(8 pi x/L),          L = 2 pi
                = 0.5 sin(10 pi x/L) + 0.5 sin(6 pi x/L)        (product-to-sum)
                Band-limited (modes k=3 and k=5). Spectral is exact to roundoff for any
                nx >= 12; spline error is the pure cubic-spline approximation error.

    2. expsin:  f(x) = exp(sin(2 pi x/L)),                     L = 4 pi
                Analytic but NOT band-limited. Fourier coefficients decay super-exponentially
                (Bessel-like). Spectral exhibits exponential / super-algebraic convergence
                that saturates to roundoff at modest nx; spline stays algebraic O(h^4).

    3. abssin:  f(x) = |sin(2 pi x/L)|,                         L = 2 pi
                C^0 but f' has jump discontinuities at x = 0, L/2 (kinks). Spectral suffers
                Gibbs-style pollution near the kinks; spline interpolation also degrades there.
                A non-smooth case to contrast against the two smooth ones above.

Each example produces:
    - Section 1: snapshot of inner_product / shift / diff_x at one moderate nx, with errors
                 vs analytic reference for both grids.
    - Section 2: convergence table as nx grows.
    - Section 3: a figure (3 rows x 2 cols) saved as test_grid1d_demo_<name>.png:
                 (A, A') shift values and pointwise error
                 (B, B') diff_x order=1 values and pointwise error
                 (C, summary) convergence log-log + text summary

Run from anywhere:
    python3 SROpInf/test/test_grid1d_demo.py
"""

import os
import sys
from typing import Sequence, Callable
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from SROpInf.grids.grid1d import Grid1DCubicSpline, Grid1DUniformSpectral  # noqa: E402

PI = np.pi


# ---------------------------------------------------------------------------
# Three example definitions. Each example is a dict with keys:
#   name, title, info, L, c, nx_value, f, f_deriv, f_shift, f_ip_exact.
# ---------------------------------------------------------------------------

# ----- Example 1: cossin -----
def _cossin_f(x, L):
    return np.cos(2 * PI * x / L) * np.sin(8 * PI * x / L)

def _cossin_deriv(x, L, order):
    """f = 0.5 sin(a x) + 0.5 sin(b x) with a = 6 pi/L, b = 10 pi/L."""
    a = 6 * PI / L
    b = 10 * PI / L
    return 0.5 * (a ** order) * np.sin(a * x + order * PI / 2) \
         + 0.5 * (b ** order) * np.sin(b * x + order * PI / 2)

# ----- Example 2: expsin -----
I0_2 = 2.2795853023360672  # modified Bessel I_0(2); matches scipy.special.i0(2.0)

def _expsin_f(x, L):
    return np.exp(np.sin(2 * PI * x / L))

def _expsin_deriv(x, L, order):
    a = 2 * PI / L
    u = a * x
    g = np.exp(np.sin(u))
    if order == 0:
        return g
    if order == 1:
        return a * np.cos(u) * g
    if order == 2:
        return a * a * (np.cos(u) ** 2 - np.sin(u)) * g
    raise NotImplementedError(f"expsin: closed-form derivative of order {order} not provided")

# ----- Example 3: abssin -----
def _abssin_f(x, L):
    return np.abs(np.sin(2 * PI * x / L))

def _abssin_deriv(x, L, order):
    """f = |sin(omega x)|, omega = 2 pi / L. f' has jumps at sin = 0; pointwise convention here
    returns 0 at the kinks (the average of the two one-sided limits +/- omega cos)."""
    a = 2 * PI / L
    u = a * x
    s = np.sin(u)
    if order == 0:
        return np.abs(s)
    if order == 1:
        return a * np.cos(u) * np.sign(s)
    if order == 2:
        return -a * a * np.abs(s)
    raise NotImplementedError(f"abssin: closed-form derivative of order {order} not provided")


def _shift_fn(f: Callable):
    """Build T_c f(x) = f(x - c) given an evaluator f(x, L)."""
    return lambda x, L, c: f(x - c, L)


EXAMPLES = [
    {
        "name": "cossin",
        "title": "f(x) = cos(2 pi x/L) * sin(8 pi x/L)",
        "info": "band-limited (modes k=3, k=5); spectral exact to roundoff",
        "L": 2 * PI, "c": 0.37, "nx_value": 32,
        "f": _cossin_f, "f_deriv": _cossin_deriv, "f_shift": _shift_fn(_cossin_f),
        "f_ip_exact": 0.25,
        "ip_label": "1/4",
    },
    {
        "name": "expsin",
        "title": "f(x) = exp(sin(2 pi x/L))",
        "info": "analytic, non-band-limited; spectral converges super-algebraically",
        "L": 4 * PI, "c": 0.73, "nx_value": 32,
        "f": _expsin_f, "f_deriv": _expsin_deriv, "f_shift": _shift_fn(_expsin_f),
        "f_ip_exact": I0_2,
        "ip_label": "I_0(2)",
    },
    {
        "name": "abssin",
        "title": "f(x) = |sin(2 pi x/L)|",
        "info": "C^0 with kinks at x = 0, L/2; spectral has Gibbs, spline degrades near kinks",
        "L": 2 * PI, "c": 0.37, "nx_value": 64,
        "f": _abssin_f, "f_deriv": _abssin_deriv, "f_shift": _shift_fn(_abssin_f),
        "f_ip_exact": 0.5,
        "ip_label": "1/2",
    },
]


# ---------------------------------------------------------------------------
# Diagnostics formatting helpers.
# ---------------------------------------------------------------------------
def err_metrics(num: np.ndarray, ref: np.ndarray) -> dict:
    diff = num - ref
    rms = float(np.linalg.norm(diff) / np.sqrt(diff.size))
    inf = float(np.max(np.abs(diff)))
    ref_rms = float(np.linalg.norm(ref) / np.sqrt(ref.size))
    rel = rms / ref_rms if ref_rms > 0 else float("nan")
    return {"inf": inf, "rms": rms, "rel_rms": rel}


def row(label, m):
    return f"    {label:<22s} inf={m['inf']:.3e}  rms={m['rms']:.3e}  rel_rms={m['rel_rms']:.3e}"


def loglog_rate(nxs, errs):
    nxs = np.asarray(nxs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    keep = errs > 0
    if keep.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0 / nxs[keep]), np.log(errs[keep]), 1)
    return slope


# ---------------------------------------------------------------------------
# Section 1: snapshot at fixed nx.
# ---------------------------------------------------------------------------
def section_snapshot(ex: dict, nx: int = 64):
    L = ex["L"]; c = ex["c"]
    print(f"\n--- Section 1 [{ex['name']}]: snapshot at nx = {nx}, L = {L:.4f}, shift c = {c} ---")
    x = np.linspace(0.0, L, nx, endpoint=False)
    g_sp = Grid1DUniformSpectral(L, len(x))
    g_cs = Grid1DCubicSpline(L, x)
    u = ex["f"](x, L)

    # inner product
    ip_ex = ex["f_ip_exact"]
    ip_sp = g_sp.inner_product(u, u)
    ip_cs = g_cs.inner_product(u, u)
    print(f"\n  inner_product <f, f>  (exact = {ip_ex:.12f} = {ex['ip_label']})")
    print(f"    spectral             = {ip_sp:.12e}    |err| = {abs(ip_sp - ip_ex):.3e}")
    print(f"    cubic spline         = {ip_cs:.12e}    |err| = {abs(ip_cs - ip_ex):.3e}")
    print(f"    |spline - spectral|  = {abs(ip_cs - ip_sp):.3e}")

    # shift
    s_ex = ex["f_shift"](x, L, c)
    s_sp = g_sp.shift(u, c); s_cs = g_cs.shift(u, c)
    print(f"\n  shift T_c[f]  (c = {c})")
    print(row("spectral vs exact",     err_metrics(s_sp, s_ex)))
    print(row("cubic spline vs exact", err_metrics(s_cs, s_ex)))
    print(row("spline vs spectral",    err_metrics(s_cs, s_sp)))

    # diff_x order = 1
    d1_ex = ex["f_deriv"](x, L, 1)
    d1_sp = g_sp.diff_x(u, 1); d1_cs = g_cs.diff_x(u, 1)
    print(f"\n  diff_x order = 1")
    print(row("spectral vs exact",     err_metrics(d1_sp, d1_ex)))
    print(row("cubic spline vs exact", err_metrics(d1_cs, d1_ex)))
    print(row("spline vs spectral",    err_metrics(d1_cs, d1_sp)))

    # diff_x order = 2
    d2_ex = ex["f_deriv"](x, L, 2)
    d2_sp = g_sp.diff_x(u, 2); d2_cs = g_cs.diff_x(u, 2)
    print(f"\n  diff_x order = 2")
    print(row("spectral vs exact",     err_metrics(d2_sp, d2_ex)))
    print(row("cubic spline vs exact", err_metrics(d2_cs, d2_ex)))
    print(row("spline vs spectral",    err_metrics(d2_cs, d2_sp)))


# ---------------------------------------------------------------------------
# Section 2: convergence as nx grows.
# ---------------------------------------------------------------------------
def section_convergence(ex: dict, nxs: Sequence[int] = (8, 12, 16, 20, 24, 32, 64, 128, 256, 512)):
    L = ex["L"]; c = ex["c"]
    print(f"\n--- Section 2 [{ex['name']}]: convergence as nx grows ---")
    rows = []
    for nx in nxs:
        x = np.linspace(0.0, L, nx, endpoint=False)
        g_sp = Grid1DUniformSpectral(L, len(x))
        g_cs = Grid1DCubicSpline(L, x)
        u = ex["f"](x, L)
        rows.append({
            "nx": nx,
            "ip_sp": abs(g_sp.inner_product(u, u) - ex["f_ip_exact"]),
            "ip_cs": abs(g_cs.inner_product(u, u) - ex["f_ip_exact"]),
            "s_sp":  float(np.linalg.norm(g_sp.shift(u, c) - ex["f_shift"](x, L, c)) / np.sqrt(nx)),
            "s_cs":  float(np.linalg.norm(g_cs.shift(u, c) - ex["f_shift"](x, L, c)) / np.sqrt(nx)),
            "d1_sp": float(np.linalg.norm(g_sp.diff_x(u, 1) - ex["f_deriv"](x, L, 1)) / np.sqrt(nx)),
            "d1_cs": float(np.linalg.norm(g_cs.diff_x(u, 1) - ex["f_deriv"](x, L, 1)) / np.sqrt(nx)),
            "d2_sp": float(np.linalg.norm(g_sp.diff_x(u, 2) - ex["f_deriv"](x, L, 2)) / np.sqrt(nx)),
            "d2_cs": float(np.linalg.norm(g_cs.diff_x(u, 2) - ex["f_deriv"](x, L, 2)) / np.sqrt(nx)),
        })
    ops = [
        ("inner_product",  "ip_sp", "ip_cs"),
        ("shift",          "s_sp",  "s_cs"),
        ("diff_x order=1", "d1_sp", "d1_cs"),
        ("diff_x order=2", "d2_sp", "d2_cs"),
    ]
    nxs_list = [r["nx"] for r in rows]
    for name, k_sp, k_cs in ops:
        print(f"\n  [{name}]")
        print(f"    {'nx':>6}  {'spectral err':>14}  {'spline err':>14}")
        for r in rows:
            print(f"    {r['nx']:>6d}  {r[k_sp]:>14.4e}  {r[k_cs]:>14.4e}")
        sp_a = loglog_rate(nxs_list, [r[k_sp] for r in rows])
        cs_a = loglog_rate(nxs_list, [r[k_cs] for r in rows])
        print(f"    log-log slope: spectral = {sp_a:.2f}, spline = {cs_a:.2f}")


# ---------------------------------------------------------------------------
# Section 3: visual comparison plot, one figure per example.
# ---------------------------------------------------------------------------
def section_plot(ex: dict,
                 nxs_conv: Sequence[int] = (8, 12, 16, 20, 24, 32, 64, 128, 256, 512),
                 save_dir: str = HERE) -> str:
    L = ex["L"]; c = ex["c"]; nx_value = ex["nx_value"]
    x_dense = np.linspace(0.0, L, 1200, endpoint=False)
    x_n = np.linspace(0.0, L, nx_value, endpoint=False)
    g_sp = Grid1DUniformSpectral(L, nx_value)
    g_cs = Grid1DCubicSpline(L, x_n)
    u_n = ex["f"](x_n, L)

    # BEFORE shift: analytic dense curve + nodal samples (both grids share x_n).
    f_dense_ex = ex["f"](x_dense, L)
    # Internal interpolants: each grid implicitly defines a continuous reconstruction of u_n at
    # arbitrary x via its shift operator. Trick: evaluating the grid's interpolant at x_dense[k]
    # equals shifting u_n by c = -x_dense[k] and reading the j=0 entry, since
    #     (T_{-x_dense[k]} u_n)(x_0) = u_n(x_0 + x_dense[k]) = u_interp(x_dense[k])   (x_0 = 0).
    # Vectorize via 2D q with a per-column c, then take row 0.
    Q_tile = np.broadcast_to(u_n[:, None], (nx_value, len(x_dense)))
    sp_interp_dense = g_sp.shift(np.ascontiguousarray(Q_tile), -x_dense)[0, :]
    cs_interp_dense = g_cs.shift(np.ascontiguousarray(Q_tile), -x_dense)[0, :]

    # shift
    s_dense_ex = ex["f_shift"](x_dense, L, c)
    s_sp = g_sp.shift(u_n, c); s_cs = g_cs.shift(u_n, c)
    s_node_ex = ex["f_shift"](x_n, L, c)
    # diff_x order=1
    d1_dense_ex = ex["f_deriv"](x_dense, L, 1)
    d1_sp = g_sp.diff_x(u_n, 1); d1_cs = g_cs.diff_x(u_n, 1)
    d1_node_ex = ex["f_deriv"](x_n, L, 1)

    # convergence (both grids)
    keys = ("inner_product", "shift", "diff_x order=1", "diff_x order=2")
    rms_sp = {k: [] for k in keys}
    rms_cs = {k: [] for k in keys}
    for nx in nxs_conv:
        xc = np.linspace(0.0, L, nx, endpoint=False)
        gs = Grid1DUniformSpectral(L, nx); gc = Grid1DCubicSpline(L, xc)
        uc = ex["f"](xc, L)
        rms_sp["inner_product"].append(abs(gs.inner_product(uc, uc) - ex["f_ip_exact"]))
        rms_cs["inner_product"].append(abs(gc.inner_product(uc, uc) - ex["f_ip_exact"]))
        rms_sp["shift"].append(np.linalg.norm(gs.shift(uc, c) - ex["f_shift"](xc, L, c)) / np.sqrt(nx))
        rms_cs["shift"].append(np.linalg.norm(gc.shift(uc, c) - ex["f_shift"](xc, L, c)) / np.sqrt(nx))
        rms_sp["diff_x order=1"].append(np.linalg.norm(gs.diff_x(uc, 1) - ex["f_deriv"](xc, L, 1)) / np.sqrt(nx))
        rms_cs["diff_x order=1"].append(np.linalg.norm(gc.diff_x(uc, 1) - ex["f_deriv"](xc, L, 1)) / np.sqrt(nx))
        rms_sp["diff_x order=2"].append(np.linalg.norm(gs.diff_x(uc, 2) - ex["f_deriv"](xc, L, 2)) / np.sqrt(nx))
        rms_cs["diff_x order=2"].append(np.linalg.norm(gc.diff_x(uc, 2) - ex["f_deriv"](xc, L, 2)) / np.sqrt(nx))
    nxs_arr = np.asarray(nxs_conv, dtype=float)

    fig, axes = plt.subplots(4, 2, figsize=(13, 16), constrained_layout=True)

    # Row 0: BEFORE-shift state.
    # (0, 0) input data: analytic f(x) + nodal samples (both grids share x_n, hence one marker set).
    ax = axes[0, 0]
    ax.plot(x_dense, f_dense_ex, "k-", lw=1.2, label="exact $f(x)$")
    ax.plot(x_n, u_n, "ko", ms=5, mfc="white", mec="k", label="nodal samples $f(x_j)$\n(both grids identical)")
    ax.set_title(f"(0) input $f(x)$ BEFORE shift, at $n_x={nx_value}$")
    ax.set_xlabel("x"); ax.set_ylabel("$f$")
    ax.legend(loc="upper right", fontsize=9)

    # (0, 1) implicit interpolant reconstruction: each grid sees the same nodal samples but
    # represents the continuous function differently between nodes.
    ax = axes[0, 1]
    ax.plot(x_dense, f_dense_ex, "k-", lw=1.2, label="exact $f(x)$")
    ax.plot(x_dense, sp_interp_dense, "C0--", lw=1.2, alpha=0.85, label="spectral interpolant")
    ax.plot(x_dense, cs_interp_dense, "C3:", lw=1.5, alpha=0.85, label="spline interpolant")
    ax.plot(x_n, u_n, "ko", ms=4, mfc="white", mec="k", label="nodes")
    ax.set_title(f"(0') implicit interpolants over dense $x$ (nodes agree, between-node differs)")
    ax.set_xlabel("x"); ax.set_ylabel("$f$")
    ax.legend(loc="upper right", fontsize=9)

    # Row A: shift values and pointwise error
    ax = axes[1, 0]
    ax.plot(x_dense, s_dense_ex, "k-", lw=1.2, label="exact $T_c f(x)$")
    ax.plot(x_n, s_sp, "C0o", ms=4, label="spectral (nodes)")
    ax.plot(x_n, s_cs, "C3x", ms=6, label="spline (nodes)")
    ax.set_title(f"(A) $T_c f$ AFTER shift at $n_x={nx_value}$, $c={c}$")
    ax.set_xlabel("x"); ax.set_ylabel("$T_c f$")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1, 1]
    ax.plot(x_n, s_sp - s_node_ex, "C0o-", lw=1, ms=4, label="spectral $-$ exact")
    ax.plot(x_n, s_cs - s_node_ex, "C3x-", lw=1, ms=6, label="spline $-$ exact")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"(A') $T_c f$ pointwise error at nodes")
    ax.set_xlabel("x"); ax.set_ylabel("error")
    ax.legend(loc="upper right", fontsize=9)

    # Row B: diff_x order=1
    ax = axes[2, 0]
    ax.plot(x_dense, d1_dense_ex, "k-", lw=1.2, label="exact $f'(x)$")
    ax.plot(x_n, d1_sp, "C0o", ms=4, label="spectral (nodes)")
    ax.plot(x_n, d1_cs, "C3x", ms=6, label="spline (nodes)")
    ax.set_title(f"(B) diff_x order=1 at $n_x={nx_value}$")
    ax.set_xlabel("x"); ax.set_ylabel("$f'$")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2, 1]
    ax.plot(x_n, d1_sp - d1_node_ex, "C0o-", lw=1, ms=4, label="spectral $-$ exact")
    ax.plot(x_n, d1_cs - d1_node_ex, "C3x-", lw=1, ms=6, label="spline $-$ exact")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"(B') diff_x order=1 pointwise error at nodes")
    ax.set_xlabel("x"); ax.set_ylabel("error")
    ax.legend(loc="upper right", fontsize=9)

    # Row C: convergence log-log + summary text
    ax = axes[3, 0]
    op_styles = {
        "inner_product":  ("C0", "o", "s"),
        "shift":          ("C1", "s", "D"),
        "diff_x order=1": ("C2", "^", "v"),
        "diff_x order=2": ("C3", "P", "X"),
    }
    floor = 1e-17  # guard against zeros on log scale
    for key, (color, mk_sp, mk_cs) in op_styles.items():
        sp_e = np.maximum(np.array(rms_sp[key]), floor)
        cs_e = np.maximum(np.array(rms_cs[key]), floor)
        ax.loglog(nxs_arr, sp_e, color=color, marker=mk_sp, linestyle="--", lw=1.0, ms=5,
                  label=f"{key} (spectral)")
        ax.loglog(nxs_arr, cs_e, color=color, marker=mk_cs, linestyle="-", lw=1.5, ms=6,
                  label=f"{key} (spline)")
    h_arr = 1.0 / nxs_arr
    cs0 = max(rms_cs["inner_product"][0], floor)
    cs0_d2 = max(rms_cs["diff_x order=2"][0], floor)
    ax.loglog(nxs_arr, 0.5 * (h_arr / h_arr[0]) ** 4 * cs0,    "k--", lw=0.7, alpha=0.6, label=r"slope $\propto h^4$")
    ax.loglog(nxs_arr, 0.5 * (h_arr / h_arr[0]) ** 2 * cs0_d2, "k:",  lw=0.7, alpha=0.6, label=r"slope $\propto h^2$")
    ax.set_title("(C) spectral vs spline RMS error vs $n_x$ (log-log)")
    ax.set_xlabel("$n_x$"); ax.set_ylabel("RMS error")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=floor * 0.5)

    # text summary
    ax = axes[3, 1]
    ax.axis("off")
    def _alpha(es):
        es = np.asarray(es, dtype=float); k = es > 0
        if k.sum() < 2: return float("nan")
        s, _ = np.polyfit(np.log(1 / nxs_arr[k]), np.log(es[k]), 1); return s
    lines = [
        f"Example: {ex['name']}",
        f"Title:   {ex['title']}",
        f"Info:    {ex['info']}",
        f"L = {L:.6f},  shift c = {c}",
        f"<f, f> exact = {ex['f_ip_exact']:.12f}  (= {ex['ip_label']})",
        "",
        f"At $n_x = {nx_value}$:",
        f"  shift            max |err|: spectral={np.max(np.abs(s_sp-s_node_ex)):.2e}, spline={np.max(np.abs(s_cs-s_node_ex)):.2e}",
        f"  diff_x order=1   max |err|: spectral={np.max(np.abs(d1_sp-d1_node_ex)):.2e}, spline={np.max(np.abs(d1_cs-d1_node_ex)):.2e}",
        "",
        "Convergence alpha (log-log slope from convergence table):",
        f"  inner_product       sp = {_alpha(rms_sp['inner_product']):>5.2f},  cs = {_alpha(rms_cs['inner_product']):>5.2f}",
        f"  shift               sp = {_alpha(rms_sp['shift']):>5.2f},  cs = {_alpha(rms_cs['shift']):>5.2f}",
        f"  diff_x order=1      sp = {_alpha(rms_sp['diff_x order=1']):>5.2f},  cs = {_alpha(rms_cs['diff_x order=1']):>5.2f}",
        f"  diff_x order=2      sp = {_alpha(rms_sp['diff_x order=2']):>5.2f},  cs = {_alpha(rms_cs['diff_x order=2']):>5.2f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace", fontsize=8.5)

    fig.suptitle(f"Grid1DUniformSpectral vs Grid1DCubicSpline  ---  [{ex['name']}]  {ex['title']}",
                 fontsize=13)

    out_path = os.path.join(save_dir, f"test_grid1d_demo_{ex['name']}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Grid1DUniformSpectral vs Grid1DCubicSpline --- three demonstration examples")
    print("=" * 80)
    saved = []
    for ex in EXAMPLES:
        header = f" Example: {ex['name']}  ({ex['info']}) ".center(80, "#")
        print("\n\n" + header)
        section_snapshot(ex, nx=ex["nx_value"] * 2)
        section_convergence(ex)
        path = section_plot(ex)
        saved.append(path)
        print(f"\n[plot saved to: {path}]")
    print("\n" + "=" * 80)
    print("All three demos complete. PNGs:")
    for p in saved:
        print(f"  {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
