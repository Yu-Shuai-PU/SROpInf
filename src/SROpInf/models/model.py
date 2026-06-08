"""
model.py --- source code for abstract FOM/ROM on the 1D periodic grid with different shift/interpolation/spatial derivative operators

classes:
    FullOrderModel: abstract class for full-order models (FOMs) on the 1D periodic grid with different shift/interpolation/spatial derivative operators
    SymmetryReducedScaledFullOrderModel: abstract class for symmetry-reduced full-order models (FOMs) on the 1D periodic grid with different shift/interpolation/spatial derivative operators
    SymmetryReducedScaledReducedOrderModel: abstract class for symmetry-reduced and scaled reduced-order models (ROMs) on the 1D periodic grid with different shift/interpolation/spatial derivative operators
"""

import numpy as np
import scipy as sp
from typing import Callable, Union, List, Dict
from types import SimpleNamespace
from SROpInf.typing import Vector, Matrix, ROMTensorTuple
from SROpInf.grids.grid1d import Grid1DUniformSpectral, Grid1DCubicSpline
from SROpInf.timestepper import Timestepper, SemiImplicit
import itertools
from string import ascii_lowercase
from scipy.integrate import solve_ivp

class FullOrderModel:
    """Abstract class for full-order models (FOMs) on the 1D periodic grid with different shift/interpolation/spatial derivative operators."""
    def __init__(self,
                grid: Union[Grid1DUniformSpectral, Grid1DCubicSpline]):
        self.grid = grid
        if not hasattr(self, 'poly_operators'):
            self.poly_operators: Dict[int, Callable] = {}
        
    @property
    def num_poly_terms(self) -> int:
        return len(self.poly_operators)    

    def inner_product(self, q1: Union[Vector, Matrix], q2: Union[Vector, Matrix]) -> Union[float, Matrix]:
        """Compute the inner product of q1 (vector or matrix) and q2 using the appropriate quadrature weights."""
        return self.grid.inner_product(q1, q2)
    
    def apply_sqrt_inner_product_mass(self, A: Union[Vector, Matrix], action: str="forward") -> Union[Vector, Matrix]:
        """Apply the square root of the mass matrix of the inner product to the vector or matrix A.
        action = "forward": apply M^(1/2) to A
        action = "inverse": apply M^(-1/2) to A
        """
        return self.grid.apply_sqrt_inner_product_mass(A, action=action)
    
    def norm(self, q: Union[Vector, Matrix]) -> Union[float, Vector]:
        """Compute the norm of q using the appropriate quadrature weights."""
        if q.ndim == 1:
            return np.sqrt(self.inner_product(q, q))
        elif q.ndim == 2:
            return np.sqrt(np.diag(self.inner_product(q, q)))
        else:
            raise ValueError(f"Input q must be 1D or 2D, got shape {q.shape!r}")

    def linear(self, q: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Evaluate the linear term of the RHS on each column of q, where q has shape (nx, M) with M columns representing M vectorized states, or q has shape (nx,) representing a single state."""
        if 1 in self.poly_operators:
            return self.poly_operators[1](q)
        else:
            return np.zeros_like(q)
    
    @property
    def linear_matrix(self) -> Matrix:
        """Dense matrix form of the linear operator L = poly_operators[1].
        Built lazily via the batched trick: L = poly_operators[1] @ I.
        """
        if not hasattr(self, "_L_matrix_cache") or self._L_matrix_cache is None:
            if 1 not in self.poly_operators:
                raise ValueError("linear_matrix requires poly_operators[1] (linear term)")
            # _linear supports 2D batched input -> applying to identity gives the dense L
            self._L_matrix_cache = self.poly_operators[1](np.eye(self.grid.nx))
        return self._L_matrix_cache
    
    def _get_implicit_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        """Return a callable r -> (I - alpha L)^-1 r.
        Grid-agnostic: builds the dense (I - alpha L) and caches its LU factorization.
        Used as the solver_factory passed to semi-implicit timesteppers.
        """
        I_alphaL = np.eye(self.grid.nx) - alpha * self.linear_matrix
        lu_piv = sp.linalg.lu_factor(I_alphaL)
        return lambda rhs: sp.linalg.lu_solve(lu_piv, rhs)

    def get_stepper(self, method: str, dt: float) -> Union[Timestepper, SemiImplicit]:
        """Build a configured timestepper for this FOM. High-level entry point that hides
        timestepper-class constructor signatures (explicit vs semi-implicit).

        Looks up the method name (case-insensitive) in both the explicit Timestepper registry
        and the semi-implicit SemiImplicit registry, then instantiates the matching class with
        the appropriate FOM hooks:
            - explicit (Euler / RK2 / RK4 / ...):  uses self.rhs
            - semi-implicit (RK2CN / RK3CN / ...): uses self.linear, self.nonlinear,
                                                    self._get_implicit_solver as solver_factory

        Example:
            ts = fom.get_stepper(method="RK3CN", dt=0.025)
            u_new = ts.step(u)
        """
        try:
            cls = SemiImplicit.lookup(method)
            return cls(
                dt=dt,
                linear=self.linear,
                nonlinear=self.nonlinear,
                solver_factory=self._get_implicit_solver,
            )
        except NotImplementedError:
            pass
        try:
            cls = Timestepper.lookup(method)
            return cls(dt=dt, rhs=self.rhs)
        except NotImplementedError:
            raise NotImplementedError(
                f"Unknown timestepper method {method!r}. Available: "
                f"explicit = {Timestepper.methods()}, semi-implicit = {SemiImplicit.methods()}"
            )
    
    def nonlinear(self, q: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Evaluate the nonlinear term of the RHS on each column of q, where q has shape (nx, M) with M columns representing M vectorized states, or q has shape (nx,) representing a single state."""
        nonlinear_q = np.zeros_like(q)
        for k, op in self.poly_operators.items():
            if k == 0:
                nonlinear_q += op()           # see Bug 3 below
            elif k >= 2:
                nonlinear_q += op(*([q] * k))
        return nonlinear_q
    
    def rhs(self, q: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Compute the right-hand side of the FOM given state q at time t."""
        return self.linear(q) + self.nonlinear(q)
    
    def rhs_scaled(self, Rq: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Compute the scaled right-hand side of the FOM given scaled state Rq."""
        q = self.apply_sqrt_inner_product_mass(Rq, action="inverse")
        rhs_q = self.rhs(q)  # Assuming time-independent RHS for now
        rhs_q_scaled = self.apply_sqrt_inner_product_mass(rhs_q, action="forward")
        return rhs_q_scaled
    
    def eval_rhs_poly_scaled(self, k: int, args: List[Union[Vector, Matrix]]) -> Union[Vector, Matrix]:
        """Compute the scaled k-th order polynomial term of the RHS given scaled arguments."""
        unscaled_args = [self.apply_sqrt_inner_product_mass(arg, action="inverse") for arg in args]
        unscaled_res = self.poly_operators[k](*unscaled_args)
        return self.apply_sqrt_inner_product_mass(unscaled_res, action="forward")
    
class SymmetryReducedScaledFullOrderModel(FullOrderModel):
    """Abstract class for symmetry-reduced and scaled full-order models (FOMs) on the 1D periodic grid with different shift/interpolation/spatial derivative operators.
    For the building of SR-ROMs only, not for actual time stepping, so don't use get_stepper here."""
    
    def __init__(self,
                grid: Union[Grid1DUniformSpectral, Grid1DCubicSpline],
                base_fom: FullOrderModel,
                q_template_dx_scaled: Vector,
                q_template_dxx_scaled: Vector):
        super().__init__(grid)
        self.base_fom = base_fom
        self.q_template_dx_scaled = q_template_dx_scaled
        self.q_template_dxx_scaled = q_template_dxx_scaled

    def shift_x(self, q_scaled: Union[Vector, Matrix], c: Union[float, Vector]) -> Union[Vector, Matrix]:
        """Shift in SCALED coordinates: unscale -> shift -> rescale. The sqrt-inner-product-mass map R
        does NOT commute with shift_x on non-spectral grids (e.g. cubic spline): shift(R q) != R shift(q).
        So apply R^{-1}, shift the physical field, then re-apply R. On a spectral grid R is a scalar,
        so this leaves the result unchanged; on a cubic-spline grid it corrects the ROM reconstruction."""
        q = self.apply_sqrt_inner_product_mass(q_scaled, action="inverse")
        return self.apply_sqrt_inner_product_mass(self.grid.shift_x(q, c), action="forward")
    
    def diff_x(self, q_scaled: Union[Vector, Matrix], order: int=1) -> Union[Vector, Matrix]:
        """Spatial derivative in SCALED coordinates: unscale -> differentiate -> rescale.
        The sqrt-inner-product-mass map R does NOT commute with diff_x on non-spectral grids
        (e.g. cubic spline): diff_x(R q) != R diff_x(q). So we must apply R^{-1}, differentiate the
        physical field, then re-apply R. On a spectral grid R is a scalar, so this leaves the result
        unchanged; on a cubic-spline grid it corrects the advection / shift-speed terms."""
        q = self.apply_sqrt_inner_product_mass(q_scaled, action="inverse")
        return self.apply_sqrt_inner_product_mass(self.grid.diff_x(q, order=order), action="forward")
    
    def shift_speed_numer(self, q_scaled: Union[Vector, Matrix]) -> Union[float, Vector]:
        """Compute the numerator of the shift speed using the appropriate inner product and template.
        dcdt_numer = -<q_template_dx, rhs(t, q)> = -np.dot(q_template_dx_scaled, rhs_scaled(t, q_scaled))
        """
        return -np.dot(self.q_template_dx_scaled, self.rhs_poly(q_scaled))
    
    def shift_speed_denom(self, q_scaled: Union[Vector, Matrix]) -> Union[float, Vector]:
        """Compute the denominator of the shift speed using the appropriate inner product and template.
        dcdt_denom = <q_template_dx, diff_x(q)> = -<q_template_dxx, q> = -np.dot(q_template_dxx_scaled, q_scaled)
        """
        return -np.dot(self.q_template_dxx_scaled, q_scaled)
    
    def inv_shift_speed_denom(self, q_scaled: Union[Vector, Matrix]) -> Union[float, Vector]:
        """Return (1/D), where D is the denominator of the shift speed."""
        denom = self.shift_speed_denom(q_scaled)
        return 1.0 / denom
    
    def shift_speed(self, q_scaled: Union[Vector, Matrix]) -> Union[float, Vector]:
        """Compute the shift speed using the appropriate inner product and template.
        dcdt = dcdt_numer / dcdt_denom, with Tikhonov regularization if the denominator is too small.
        """
        numer = self.shift_speed_numer(q_scaled)
        inv_denom = self.inv_shift_speed_denom(q_scaled)
        return numer * inv_denom
    
    def rhs(self, q_scaled: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Compute the right-hand side of the FOM given state q at time t."""
        return self.rhs_poly(q_scaled) + self.shift_speed(q_scaled) * self.advection(q_scaled)
    
    def project(self,
                poly_comp: List[int],
                phi_scaled: Matrix,
                shift_speed_denom_threshold: float = 0.0) -> "SymmetryReducedScaledReducedOrderModel":
        """Project the full-order dynamics onto the test and trial subspaces
        Notice that we have user-input linear and bilinear term,
        because this might depend on the choice of inner product and the way we apply mass of the inner product
        """
        r = phi_scaled.shape[1]
        rhs_tensors = []
        shift_numer_tensors = []
        
        for k in poly_comp:
            rhs_shape = (r,) * (k + 1)
            shift_shape = (r,) * k
            tensor_rhs = np.zeros(rhs_shape)
            tensor_shift = np.zeros(shift_shape)
            if k == 0:
                f_scaled_eval = self.eval_rhs_poly_scaled(0, [])
                tensor_rhs[:] = phi_scaled.T @ f_scaled_eval
                tensor_shift = np.array(np.dot(-1 * self.q_template_dx_scaled, f_scaled_eval))
            elif k == 1:
                f_scaled_eval_matrix = self.eval_rhs_poly_scaled(1, [phi_scaled])
                tensor_rhs = phi_scaled.T @ f_scaled_eval_matrix
                tensor_shift = -1 * self.q_template_dx_scaled.T @ f_scaled_eval_matrix    
            else:
                for multi_index in itertools.product(range(r), repeat=k):
                    phi_columns = [phi_scaled[:, idx] for idx in multi_index]
                    f_scaled_eval = self.eval_rhs_poly_scaled(k, phi_columns)

                    reduced_rhs_vec = phi_scaled.T @ f_scaled_eval
                    rhs_target_slice = (slice(None),) + multi_index
                    tensor_rhs[rhs_target_slice] = reduced_rhs_vec
                    
                    reduced_shift_scalar = np.dot(-1 * self.q_template_dx_scaled, f_scaled_eval)
                    tensor_shift[multi_index] = reduced_shift_scalar
            rhs_tensors.append(tensor_rhs)
            shift_numer_tensors.append(tensor_shift)
        tensors = rhs_tensors + shift_numer_tensors
        
        dcdt_denom_scaled = self.shift_speed_denom(phi_scaled)
        advection_scaled = phi_scaled.T @ self.advection(phi_scaled)
        tensors.append(dcdt_denom_scaled)
        tensors.append(advection_scaled)
        
        return SymmetryReducedScaledReducedOrderModel(poly_comp,
                                                    phi_scaled,
                                                    tuple(tensors),
                                                    q_template_dxx_scaled = self.q_template_dxx_scaled,
                                                    shift_func = self.shift_x,
                                                    shift_speed_denom_threshold = shift_speed_denom_threshold)

    def advection(self, q_scaled: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Compute the advection term = diff_x(q) using the appropriate spatial derivative operator."""
        return self.diff_x(q_scaled, order=1)
    
    def rhs_poly(self, q_scaled: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Evaluate the polynomial part of the RHS, without the shift term."""
        q = self.apply_sqrt_inner_product_mass(q_scaled, action="inverse")
        rhs_q = self.base_fom.rhs(q)  # Assuming time-independent RHS for now
        return self.apply_sqrt_inner_product_mass(rhs_q, action="forward")
    
    def eval_rhs_poly_scaled(self, k: int, args: List[Union[Vector, Matrix]]) -> Union[Vector, Matrix]:
        unscaled = [self.apply_sqrt_inner_product_mass(a, action="inverse") for a in args]  # self=cubic
        res = self.base_fom.poly_operators[k](*unscaled)                                    # base_fom=spectral f
        return self.apply_sqrt_inner_product_mass(res, action="forward")                    # self=cubic
    
class SymmetryReducedScaledReducedOrderModel:
    """Abstract class for general (linear-bilinear) ROMs with symmetry-reducing term."""
    def __init__(self,
                poly_comp: List[int],
                phi_scaled: Matrix,
                tensors: ROMTensorTuple,
                q_template_dxx_scaled: Vector,
                shift_func: Callable[[Union[Vector, Matrix], Union[float, Vector]], Union[Vector, Matrix]],
                shift_speed_denom_threshold: float):
        
        self.poly_comp = poly_comp
        self.phi_scaled = phi_scaled
        self.tensors = tensors
        self.shift_func = shift_func
        self.q_template_dxx_scaled = q_template_dxx_scaled
        self.shift_speed_denom_threshold = shift_speed_denom_threshold
        self._get_indices_rhs()
        self._get_indices_shift_speed_numer()
        
    @classmethod
    def build(cls,
            poly_comp: List[int],
            phi_scaled: Matrix,
            tensors: ROMTensorTuple,
            q_template_dxx_scaled: Vector,
            shift_func: Callable[[Union[Vector, Matrix], Union[float, Vector]], Union[Vector, Matrix]],
            shift_speed_denom_threshold: float):
        return cls(poly_comp, phi_scaled, tensors,
                q_template_dxx_scaled = q_template_dxx_scaled,
                shift_func = shift_func,
                shift_speed_denom_threshold = shift_speed_denom_threshold)
        
    def _get_indices_rhs(self):
        """Generates the subscript indices for the einsum evaluation of the polynomial part of the symmetry-reduced dynamics"""
        ss = []
        for k in self.poly_comp:
            ssk = ascii_lowercase[:k+1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        self.einsum_ss_rhs_poly = tuple(ss)
        self.einsum_str_rhs_poly = [",".join(ss) for ss in self.einsum_ss_rhs_poly]
        
        # Pre-plan einsum contraction paths. Path depends only on shapes, not values
        r = self.phi_scaled.shape[1]
        z_dummy = np.zeros(r)
        self.einsum_path_rhs_poly = []
        for i, k in enumerate(self.poly_comp):
            eq = self.einsum_str_rhs_poly[i]
            path, _ = np.einsum_path(eq, self.tensors[i], *([z_dummy] * k), optimize='greedy')
            self.einsum_path_rhs_poly.append(path)
        
    def _get_indices_shift_speed_numer(self):
        """Generates the subscript indices for the einsum evaluation of the numerator of the shift reconstruction equation, part of the symmetry-reduced dynamics"""
        ss = []
        for k in self.poly_comp:
            ssk = ascii_lowercase[:k]
            ssk = [ssk] + [s for s in ssk]
            ss.append(ssk)

        self.einsum_ss_shift_speed_numer = tuple(ss)
        self.einsum_str_shift_speed_numer = [",".join(ss) for ss in self.einsum_ss_shift_speed_numer]
        # Cache contraction paths (used only by fallback k≥3; k∈{0,1,2} bypass einsum).
        r = self.phi_scaled.shape[1]
        z_dummy = np.zeros(r)
        self.einsum_path_shift_speed_numer = []
        for i, k in enumerate(self.poly_comp):
            tensor = self.tensors[i + len(self.poly_comp)]
            if k == 0:
                self.einsum_path_shift_speed_numer.append(None)
                continue
            path, _ = np.einsum_path(self.einsum_str_shift_speed_numer[i], tensor,
                                     *([z_dummy] * k), optimize='greedy')
            self.einsum_path_shift_speed_numer.append(path)
        
    def full_to_latent(self, q_scaled: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        return self.phi_scaled.T @ q_scaled
    
    def latent_to_full(self, z: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        return self.phi_scaled @ z
        
    def shift_x(self, q_scaled: Union[Vector, Matrix], c: Union[float, Vector]) -> Union[Vector, Matrix]:
        return self.shift_func(q_scaled, c)
    
    def shift_speed_numer(self, z: Vector) -> float:
        """Evaluate the numerator of the shift speed given the current state z: numer = -1 * <f(q_fitted), q_template_dx> = B1(z) + B2(z, z) + ..."""
        numer = 0.0
        for i, k in enumerate(self.poly_comp):
            tensor = self.tensors[i + len(self.poly_comp)]
            numer += float(np.einsum(self.einsum_str_shift_speed_numer[i],
                                      tensor, *([z] * k),
                                      optimize=self.einsum_path_shift_speed_numer[i]))
        return numer
    
    def shift_speed_denom(self, z: Vector) -> float:
        """Raw denominator D = <tensors[-2], z>."""
        return np.dot(self.tensors[-2], z)
    
    def inv_shift_speed_denom(self, z: Vector) -> float:
        """The regularized inverse of shift speed denominator = D / (D^2 + tau^2)."""
        denom = np.dot(self.tensors[-2], z)
        tau = self.shift_speed_denom_threshold
        return denom / (denom**2 + tau**2)
    
    def shift_speed(self, z: Vector) -> float:
        """Return the shift speed = numer * inv_denom, where inv_denom is regularized to avoid instability when the denominator is close to zero."""
        return self.shift_speed_numer(z) * self.inv_shift_speed_denom(z)
    
    def _rhs_poly(self, z: Vector) -> Vector:
        """Evaluate the polynomial part of the RHS, without the shift term."""
        rhs_poly = np.zeros_like(z)
        for i, k in enumerate(self.poly_comp):
            tensor = self.tensors[i]
            rhs_poly += np.einsum(self.einsum_str_rhs_poly[i],
                                  tensor, *([z] * k),
                                  optimize=self.einsum_path_rhs_poly[i])
        return rhs_poly
    
    def _advection(self, z: Vector) -> Vector:
        """Evaluate the advection term = diff_x(q) in the reduced space"""
        return np.dot(self.tensors[-1], z)
    
    def rhs_z(self, z: Vector) -> Vector:
        return self._rhs_poly(z) + self.shift_speed(z) * self._advection(z)
    
    def rhs_zc(self, t: float, zc: Vector) -> Vector:
        """Evaluate the dynamics of state z and shift amount c."""
        z = zc[:-1]
        dcdt = self.shift_speed(z)
        dzdt = self.rhs_z(z)
        return np.hstack([dzdt, dcdt])
    
    def sample_and_compare(self,
        idx_traj: int,
        y0: Vector, # = [z0, c0]
        t_eval: Vector,
        model_list: List[str],
        model_path: Dict[str, str]) -> Dict[str, List[Union[float, Vector, Matrix]]]:
        """Sample the ROM trajectory given initial state z0 and evaluation time points t_eval
        and also load the benchmark trajectories for comparison and plotting
        e.g., benchmark_list = ["fom", "podgal"]
        """
        try:
            sol = solve_ivp(
                fun = self.rhs_zc,
                t_span = (t_eval[0], t_eval[-1]),
                y0 = y0,
                method = 'Radau',
                t_eval = t_eval,
                rtol = 1e-3, # default value
                atol = 1e-6 # default value
            )
        except (ValueError, FloatingPointError, ArithmeticError) as e:
            print(
                f"Traj {idx_traj}: sample_and_compare raised {type(e).__name__}: {e}. "
                f"Filling all {len(t_eval)} points with NaN."
            )
            sol = SimpleNamespace(
                y=np.full((y0.shape[0], len(t_eval)), np.nan),
                t=t_eval.copy(),
                t_events=[],
                status=-1,
                message=str(e),
                )

        # solve_ivp does NOT raise when its internal step-size control gives up: it returns
        # normally with sol.status == -1 and sol.y truncated to the columns it actually reached.
        # Detect that here and right-pad with NaN so downstream code (which assumes len(t_eval)
        # columns) keeps working.
        if getattr(sol, "status", 0) != 0 or sol.y.shape[1] < len(t_eval):
            n_reached = sol.y.shape[1]
            t_reached = float(sol.t[-1]) if n_reached > 0 else float("nan")
            print(
                f"Traj {idx_traj}: solve_ivp status={getattr(sol, 'status', '?')} "
                f"({getattr(sol, 'message', '')}). Reached t={t_reached:.3f} / {t_eval[-1]:.3f} "
                f"({n_reached}/{len(t_eval)} samples). Padding remainder with NaN."
            )
            y_full = np.full((y0.shape[0], len(t_eval)), np.nan)
            if n_reached > 0:
                y_full[:, :n_reached] = sol.y
            sol = SimpleNamespace(
                y=y_full,
                t=t_eval.copy(),
                t_events=[],
                status=getattr(sol, "status", -1),
                message=getattr(sol, "message", ""),
            )

        Q_rom_fitted_scaled = self.latent_to_full(sol.y[:-1])
        c_rom = sol.y[-1]
        inv_dcdt_denom_rom = np.array([self.inv_shift_speed_denom(sol.y[:-1, idx_time]) for idx_time in range(len(t_eval))])
        Q_rom_scaled = self.shift_x(Q_rom_fitted_scaled, c_rom)

        Q_scaled_list = []
        error_list = []
        c_list = []
        inv_dcdt_denom_list = []
        
        fname_traj_scaled = model_path["traj_scaled"]
        fname_error = model_path["error"]
        fname_shift_amount = model_path["shift_amount"]
        fname_inv_shift_speed_denom = model_path["inv_shift_speed_denom"]
                
        # load FOM data
        Q_fom_scaled = np.load(fname_traj_scaled % (model_list[0], idx_traj))
        Q_scaled_list.append(Q_fom_scaled)
        c_list.append(np.load(fname_shift_amount % (model_list[0], idx_traj)))
        inv_dcdt_denom_list.append(np.load(fname_inv_shift_speed_denom % (model_list[0], idx_traj)))
        
        for model in model_list[1:-1]: # load the benchmark trajectories except FOM and the ROM trajectory we just sampled
            Q_scaled = np.load(fname_traj_scaled % (model, idx_traj))
            Q_scaled_list.append(Q_scaled)
            error = np.load(fname_error % (model, idx_traj))
            error_list.append(error)
            c = np.load(fname_shift_amount % (model, idx_traj))
            c_list.append(c)
            inv_dcdt_denom = np.load(fname_inv_shift_speed_denom % (model, idx_traj))
            inv_dcdt_denom_list.append(inv_dcdt_denom)
            
        diff_rom_fom = Q_rom_scaled - Q_fom_scaled
        error_rom = np.sqrt( (np.mean(np.linalg.norm(diff_rom_fom, axis=0) ** 2)) / np.mean(np.linalg.norm(Q_fom_scaled, axis=0) ** 2) ) # this is the relative RMSE of the ROM trajectory compared to the FOM trajectory
        
        Q_scaled_list.append(Q_rom_scaled)
        error_list.append(error_rom)
        c_list.append(c_rom)
        inv_dcdt_denom_list.append(inv_dcdt_denom_rom)

        np.save(fname_traj_scaled % (model_list[-1], idx_traj), Q_rom_scaled)
        np.save(fname_error % (model_list[-1], idx_traj), error_rom)
        np.save(fname_shift_amount % (model_list[-1], idx_traj), c_rom)
        np.save(fname_inv_shift_speed_denom % (model_list[-1], idx_traj), inv_dcdt_denom_rom)
                
        print(f"Traj {idx_traj}: relative RMSE {100*error_rom:.2f}%")
        
        return {"Q_scaled_list": Q_scaled_list, "error_list": error_list, "c_list": c_list, "inv_dcdt_denom_list": inv_dcdt_denom_list}
        
    def solve(self, z0: Vector, c0: float, t_eval: Vector):
        y0 = np.hstack([z0, c0])
        try:
            sol = solve_ivp(
                fun = self.rhs_zc,
                t_span = (t_eval[0], t_eval[-1]),
                y0 = y0,
                method = 'Radau',
                t_eval = t_eval,
                rtol = 1e-3, # default value
                atol = 1e-6 # default value
            )
        except (ValueError, FloatingPointError, ArithmeticError) as e:
            sol = SimpleNamespace(
                y=np.full((y0.shape[0], len(t_eval)), np.nan),
                t=t_eval.copy(),
                t_events=[],
                status=-1,
                message=str(e),
                )
        if getattr(sol, "status", 0) != 0 or sol.y.shape[1] < len(t_eval):
            n_reached = sol.y.shape[1]
            y_full = np.full((y0.shape[0], len(t_eval)), np.nan)
            if n_reached > 0:
                y_full[:, :n_reached] = sol.y
            sol = SimpleNamespace(
                y=y_full,
                t=t_eval.copy(),
                t_events=[],
                status=getattr(sol, "status", -1),
                message=getattr(sol, "message", ""),
            )
        return sol.y