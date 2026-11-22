"""Sparse plastic modes — the plastic-deformation analogue of fracture modes.

Plastic modes minimise the group-sparsity norm of per-tet strain fields:

    min  sum_tet sigma_y(tet) * vol(tet) * || eps_phys(u)|_tet ||_2

subject to mass-orthogonality against previous modes and a normalisation
constraint. ``eps_phys`` is the strain tensor written in physical-Voigt form so
that its Euclidean norm equals the Frobenius norm of the strain tensor:

    eps_phys = [e_xx, e_yy, e_zz, sqrt(2)*e_xy, sqrt(2)*e_yz, sqrt(2)*e_xz]

Search space
------------

The naive formulation (searching u over the full 3V displacement space) has
two failure modes: it favours rigid-body solutions (which have zero strain
and therefore zero objective) and it favours single-vertex "spikes" (which
concentrate strain in one tet, minimising the objective at the cost of any
physical plausibility). Both are eliminated by restricting the search to the
low-frequency *elastic subspace* — the columns of a precomputed ElasticModes
bundle. Rigid-body motions are absent from that subspace by construction, and
every mode in it is a smooth low-frequency deformation, so any linear
combination is smooth too.

Concretely, we substitute u = Phi_e @ alpha (Phi_e is 3V x M_elastic and
M-orthonormal) into the SOCP, giving a reduced problem in R^{M_elastic}:

    min  sum_tet sigma_y * vol * || eps_phys(Phi_e @ alpha)|_tet ||_2
    s.t. alpha_j^T alpha_k = 0 for previous modes j,  alpha^T c_alpha = 1

which we hand to ``conic_solve`` with the reduced group operator
B_reduced = B_group @ Phi_e (shape 6T x M_elastic). Because Phi_e is
M-orthonormal, the reduced mass matrix is the identity — no separate mass
assembly needed in the reduced problem.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye as speye, diags
from scipy.sparse.linalg import eigsh
import igl

from .conic_solve import conic_solve
from .material_field import MaterialField
from .elastic_modes import _tet_shape_gradients, _strain_displacement_B, ElasticModes


PHYS_VOIGT_SCALE = np.array(
    [1.0, 1.0, 1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]
)


def precompute_tet_operators(nodes, elements):
    """Per-tet B (engineering-shear Voigt) and vol; global 12-DOF index array.

    Analogue of the inner loop of ``elastic_modes.assemble_K_M``, but K and M
    are not built here — this helper is used by any code path that only needs
    the strain operator (plastic modes, fracture-mode warping).
    """
    T = elements.shape[0]
    B_per_tet = np.empty((T, 6, 12), dtype=np.float64)
    vol_per_tet = np.empty(T, dtype=np.float64)
    gdof_per_tet = np.empty((T, 12), dtype=np.int64)
    for t in range(T):
        v_idx = elements[t]
        v0, v1, v2, v3 = nodes[v_idx]
        grads, vol = _tet_shape_gradients(v0, v1, v2, v3)
        if vol < 0.0:
            vol = -vol
            grads = -grads
        B_per_tet[t] = _strain_displacement_B(grads)
        vol_per_tet[t] = vol
        gdof_per_tet[t] = np.array(
            [3 * v_idx[i] + d for i in range(4) for d in range(3)],
            dtype=np.int64,
        )
    return B_per_tet, vol_per_tet, gdof_per_tet


def build_plastic_group_operator(B_per_tet, vol_per_tet, gdof_per_tet,
                                 sigma_y_per_tet, num_vertices):
    """Assemble the sparse (6T, 3V) plastic-strain group operator.

    Row block ``t`` (rows 6t..6t+5) holds the physical-Voigt strain operator
    of tet ``t`` scaled by ``sigma_y(t) * vol(t)``. Given a global displacement
    u in R^{3V}, ``(B_group @ u)[6t..6t+5]`` = ``sigma_y*vol * eps_phys(u)|_t``,
    so the (2,1) norm summed over tets is the plastic-dissipation objective.
    """
    T = B_per_tet.shape[0]
    scale = PHYS_VOIGT_SCALE[None, :, None]
    B_phys = B_per_tet * scale
    w = (sigma_y_per_tet * vol_per_tet)[:, None, None]
    B_scaled = w * B_phys

    row_base = 6 * np.arange(T, dtype=np.int64)
    rows = np.broadcast_to(
        row_base[:, None, None] + np.arange(6, dtype=np.int64)[None, :, None],
        (T, 6, 12),
    )
    cols = np.broadcast_to(gdof_per_tet[:, None, :], (T, 6, 12))

    B_group = coo_matrix(
        (B_scaled.ravel(), (rows.ravel(), cols.ravel())),
        shape=(6 * T, 3 * num_vertices),
    ).tocsr()
    return B_group


class PlasticModes:
    """Precomputed sparse plastic-modes subspace inside an elastic subspace.

    Requires:
      - ``materials.sigma_y`` set (per-tet yield stress; weights the objective).
      - ``elastic_modes`` — a computed :class:`ElasticModes` bundle whose
        columns span the search space. Providing more elastic modes gives the
        SOCP a richer feasible set; typically 20-40 modes is plenty.
    """

    def __init__(self, elastic_modes, materials):
        if not isinstance(elastic_modes, ElasticModes):
            raise TypeError(
                "elastic_modes must be an ElasticModes instance "
                f"(got {type(elastic_modes).__name__})"
            )
        if elastic_modes.modes is None:
            raise ValueError(
                "elastic_modes.compute_modes(...) must be called before "
                "constructing PlasticModes."
            )
        if not isinstance(materials, MaterialField):
            raise TypeError(
                f"materials must be a MaterialField (got {type(materials).__name__})"
            )
        if materials.sigma_y is None:
            raise ValueError(
                "PlasticModes requires materials.sigma_y to be set; "
                "author your .npz with --sigma_y-inside / --sigma_y-outside."
            )
        if materials.num_tets != elastic_modes.num_tets:
            raise ValueError(
                f"materials.num_tets = {materials.num_tets} != "
                f"elastic_modes.num_tets = {elastic_modes.num_tets}"
            )

        self.elastic_modes = elastic_modes
        self.materials = materials
        self.nodes = elastic_modes.nodes
        self.elements = elastic_modes.elements
        self.num_vertices = elastic_modes.num_vertices
        self.num_tets = elastic_modes.num_tets
        self.num_elastic_modes = elastic_modes.modes.shape[1]

        self.sigma_y_per_tet = np.asarray(materials.sigma_y, dtype=np.float64).copy()

        self.B_per_tet = None
        self.vol_per_tet = None
        self.gdof_per_tet = None
        self.B_group = None
        self.B_reduced = None

        self.mode_alphas = None
        self.modes = None
        self.mode_energies = None
        self.n_yielded_per_mode = None

    def assemble(self):
        """Precompute per-tet operators and the reduced-basis group operator."""
        (self.B_per_tet, self.vol_per_tet,
         self.gdof_per_tet) = precompute_tet_operators(self.nodes, self.elements)

        self.B_group = build_plastic_group_operator(
            self.B_per_tet, self.vol_per_tet, self.gdof_per_tet,
            self.sigma_y_per_tet, self.num_vertices,
        )

        Phi = self.elastic_modes.modes
        self.B_reduced = self.B_group @ Phi
        return self

    def compute_modes(self, num_modes=8, max_iter=10, tol=1.0e-4,
                      strain_threshold=1.0e-6, verbose=False):
        """Iterate the reduced SOCP to find ``num_modes`` sparse plastic modes.

        The SOCP lives in R^{num_elastic_modes}. Each mode is initialised with
        the k-th standard basis vector (which corresponds to the k-th elastic
        mode as a starting displacement), then re-projected toward the sparse
        minimum while remaining orthogonal to previously-found modes.

        Returns
        -------
        self, with:
          - ``mode_alphas``    (num_elastic_modes, num_modes)  reduced coefficients
          - ``modes``          (3V, num_modes)                 full displacement vectors
          - ``mode_energies``  (num_modes,)                    (2,1) objective per mode
          - ``n_yielded_per_mode`` (num_modes,)                sparsity diagnostic
        """
        if self.B_reduced is None:
            self.assemble()

        if num_modes > self.num_elastic_modes:
            raise ValueError(
                f"num_modes ({num_modes}) cannot exceed num_elastic_modes "
                f"({self.num_elastic_modes}). Rebuild the ElasticModes bundle "
                f"with more --num-modes."
            )

        M_reduced = speye(self.num_elastic_modes, format="csc")

        As = []
        energies = []
        n_yielded = []
        Phi = self.elastic_modes.modes

        for k in range(num_modes):
            c = np.zeros(self.num_elastic_modes, dtype=np.float64)
            c[k] = 1.0

            iter_num = 0
            diff = 1.0
            alpha = c.copy()
            while diff > tol and iter_num < max_iter:
                aprev = alpha
                a_new = conic_solve(coo_matrix(self.B_reduced), M_reduced, As, c, 6)
                nrm = float(np.linalg.norm(a_new))
                if nrm < 1e-14:
                    if verbose:
                        print(f"  mode {k}: degenerate norm at iter {iter_num}")
                    break
                alpha = a_new / nrm
                c = alpha
                diff = float(np.max(np.abs(alpha - aprev)))
                iter_num += 1

            As.append(alpha)

            u_full = Phi @ alpha
            grp_sq = (self.B_group @ u_full) ** 2
            group_norms = np.sqrt(
                np.add.reduceat(grp_sq, np.arange(0, 6 * self.num_tets, 6))
            )
            energy = float(group_norms.sum())
            energies.append(energy)
            threshold = strain_threshold * self.sigma_y_per_tet * self.vol_per_tet
            n_yielded.append(int((group_norms > threshold).sum()))

            if verbose:
                print(
                    f"  mode {k+1:3d}: iters={iter_num:3d}  diff={diff:.4g}  "
                    f"energy={energy:.4g}  yielded_tets={n_yielded[-1]}"
                )

        self.mode_alphas = (
            np.column_stack(As) if As
            else np.zeros((self.num_elastic_modes, 0))
        )
        self.modes = Phi @ self.mode_alphas
        self.mode_energies = np.asarray(energies, dtype=np.float64)
        self.n_yielded_per_mode = np.asarray(n_yielded, dtype=np.int64)
        return self

    def mode_displacement(self, mode_index, amplitude=1.0):
        """Return per-vertex displacement (V, 3) for a single plastic mode."""
        if self.modes is None:
            raise RuntimeError("Call compute_modes() first.")
        return (self.modes[:, mode_index] * amplitude).reshape(self.num_vertices, 3)

    def mode_strain(self, mode_index):
        """Return per-tet physical-Voigt strain (T, 6) for a single mode.

        Uses the *unscaled* physical-Voigt strain (not the sigma_y*vol-weighted
        version stored in B_group), so the user sees pure strain magnitudes.
        """
        if self.modes is None:
            raise RuntimeError("Call compute_modes() first.")
        self._ensure_tet_operators()
        u = self.modes[:, mode_index]
        u_local = u[self.gdof_per_tet]
        eng = np.einsum("tij,tj->ti", self.B_per_tet, u_local)
        return eng * PHYS_VOIGT_SCALE[None, :]

    def mode_strain_magnitude(self, mode_index):
        """Return per-tet ||eps_phys||_2 (T,) for a single mode (Frobenius norm)."""
        eps = self.mode_strain(mode_index)
        return np.linalg.norm(eps, axis=1)

    def to_file(self, path, elastic_modes_path=None):
        """Save the reduced-coefficient modes + metadata to an .npz sidecar.

        We store the full 3V displacement columns (``modes``) so that downstream
        consumers can use them without needing the elastic bundle at load time;
        we also store the reduced coefficients (``mode_alphas``) plus a hint at
        the elastic-modes path if provided.
        """
        if self.modes is None:
            raise RuntimeError("Call compute_modes() first.")
        payload = dict(
            nodes=self.nodes,
            elements=self.elements,
            sigma_y_per_tet=self.sigma_y_per_tet,
            modes=self.modes,
            mode_alphas=self.mode_alphas,
            mode_energies=self.mode_energies,
            n_yielded_per_mode=self.n_yielded_per_mode,
            num_elastic_modes=np.array(self.num_elastic_modes),
        )
        if elastic_modes_path is not None:
            payload["elastic_modes_path"] = np.array(str(elastic_modes_path))
        np.savez(path, **payload)

    @classmethod
    def from_file(cls, path, elastic_modes=None, materials=None):
        """Load a previously-computed PlasticModes bundle.

        For pure read/preview usage the class can be reconstituted without
        elastic_modes: a shallow instance is returned with ``.modes`` filled
        and ``.elastic_modes`` set to None. For anything that needs to run
        additional SOCP iterations, pass a matching ``ElasticModes`` bundle.
        """
        data = np.load(path, allow_pickle=False)
        nodes = data["nodes"]
        elements = data["elements"]
        num_tets = elements.shape[0]
        sigma_y = data["sigma_y_per_tet"]

        if elastic_modes is not None:
            if materials is None:
                materials = MaterialField(
                    num_tets=num_tets,
                    G_c=np.ones(num_tets),
                    sigma_y=sigma_y,
                )
            obj = cls(elastic_modes=elastic_modes, materials=materials)
        else:
            obj = cls.__new__(cls)
            obj.elastic_modes = None
            obj.materials = None
            obj.nodes = nodes
            obj.elements = elements
            obj.num_vertices = nodes.shape[0]
            obj.num_tets = num_tets
            obj.num_elastic_modes = int(data["num_elastic_modes"])
            obj.sigma_y_per_tet = np.asarray(sigma_y, dtype=np.float64).copy()
            obj.B_per_tet = None
            obj.vol_per_tet = None
            obj.gdof_per_tet = None
            obj.B_group = None
            obj.B_reduced = None

        obj.modes = data["modes"]
        obj.mode_alphas = data["mode_alphas"]
        obj.mode_energies = data["mode_energies"]
        obj.n_yielded_per_mode = data["n_yielded_per_mode"]
        return obj

    def _ensure_tet_operators(self):
        if self.B_per_tet is None:
            (self.B_per_tet, self.vol_per_tet,
             self.gdof_per_tet) = precompute_tet_operators(self.nodes, self.elements)

    def _static_elastic_response(self, f):
        """u = K^{-1} f via modal expansion over the elastic subspace.

        u = sum_k (phi_k^T f / omega_k^2) phi_k    (M-orthonormal modes)
        """
        if self.elastic_modes is None:
            raise RuntimeError(
                "project_impact requires the elastic_modes bundle. Reload via "
                "PlasticModes.from_file(path, elastic_modes=<ElasticModes>)."
            )
        Phi = self.elastic_modes.modes
        omega_sq = np.clip(self.elastic_modes.eigenvalues, 1.0e-16, None)
        f_modal = Phi.T @ np.asarray(f, dtype=np.float64).reshape(-1)
        return Phi @ (f_modal / omega_sq)

    def _per_tet_strain(self, u):
        """Per-tet engineering-shear Voigt strain from a flat displacement u."""
        self._ensure_tet_operators()
        u_local = u[self.gdof_per_tet]
        return np.einsum("tij,tj->ti", self.B_per_tet, u_local)

    def _nodal_force(self, contact_point, direction, magnitude, spread_radius):
        contact = np.asarray(contact_point, dtype=np.float64).reshape(3)
        dir_vec = np.asarray(direction, dtype=np.float64).reshape(3)
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1.0e-12:
            raise ValueError("direction must be non-zero.")
        dir_unit = dir_vec / dir_norm
        force = float(magnitude) * dir_unit
        d = np.linalg.norm(self.nodes - contact[None, :], axis=1)
        f = np.zeros(3 * self.num_vertices, dtype=np.float64)
        if spread_radius <= 0.0:
            v = int(np.argmin(d))
            f[3 * v:3 * v + 3] = force
            return f
        sigma = 0.5 * float(spread_radius)
        w = np.exp(-0.5 * (d / sigma) ** 2)
        w[d > spread_radius] = 0.0
        w_sum = w.sum()
        if w_sum <= 0.0:
            v = int(np.argmin(d))
            f[3 * v:3 * v + 3] = force
            return f
        w = w / w_sum
        for v in np.where(w > 0.0)[0]:
            f[3 * v:3 * v + 3] = w[v] * force
        return f

    def reset_runtime(self):
        """Clear the fired-modes history and cumulative rest-pose displacement."""
        self.fired_modes = []
        self.cumulative_displacement = np.zeros((self.num_vertices, 3), dtype=np.float64)

    def _ensure_runtime(self):
        if not hasattr(self, "fired_modes") or self.fired_modes is None:
            self.reset_runtime()

    def _per_tet_sigma_vm(self, eps_e, poisson_ratio):
        """Per-tet von-Mises stress from an engineering-shear Voigt strain array."""
        E = (self.elastic_modes.E_per_tet if self.elastic_modes is not None
             else np.full(self.num_tets, 2.0e11))
        nu = poisson_ratio
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        tr = eps_e[:, 0] + eps_e[:, 1] + eps_e[:, 2]
        sig = np.empty_like(eps_e)
        sig[:, 0] = lam * tr + 2.0 * mu * eps_e[:, 0]
        sig[:, 1] = lam * tr + 2.0 * mu * eps_e[:, 1]
        sig[:, 2] = lam * tr + 2.0 * mu * eps_e[:, 2]
        sig[:, 3] = mu * eps_e[:, 3]
        sig[:, 4] = mu * eps_e[:, 4]
        sig[:, 5] = mu * eps_e[:, 5]
        p = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
        s = sig.copy(); s[:, 0] -= p; s[:, 1] -= p; s[:, 2] -= p
        s_dd = (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
                + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
        return np.sqrt(1.5 * s_dd)

    def apply_impact(self, contact_point, direction, magnitude,
                     spread_radius=0.0, poisson_ratio=0.3,
                     yield_excess_threshold=0.0,
                     mode_amplitude=None):
        """Fire at most one plastic mode in response to an impact.

        Semantics (mirrors the fracture pipeline's discrete-event model):

        * Compute the static elastic response to the impact.
        * Per-tet excess = sigma_vm - sigma_y. If ``max(excess)`` is below
          ``yield_excess_threshold`` the impact is elastic and no mode fires.
        * Otherwise project the elastic displacement onto the plastic-modes
          basis (in the M-inner-product; modes are M-orthonormal). The winner
          is the highest-scoring mode **among those not yet fired**.
        * Fire the winner: add ``sign(alpha) * mode_amplitude * mode_shape`` to
          the cumulative rest-pose displacement, and record the mode index in
          ``self.fired_modes``.
        * Once all modes have fired, further impacts return
          ``reason='all_modes_exhausted'`` — max plastic deformation reached.

        ``mode_amplitude`` defaults to ``0.05 * mesh_diagonal / max|mode_i|`` so
        each mode adds a dent worth ~5% of the mesh bounding box, giving a
        predictable, visible deformation per firing.
        """
        self._ensure_tet_operators()
        self._ensure_runtime()
        num_modes = self.modes.shape[1] if self.modes is not None else 0

        f = self._nodal_force(contact_point, direction, magnitude, spread_radius)
        u_e_flat = self._static_elastic_response(f)
        eps_e = self._per_tet_strain(u_e_flat)
        sigma_vm = self._per_tet_sigma_vm(eps_e, poisson_ratio)

        excess = sigma_vm - self.sigma_y_per_tet
        peak_tet = int(np.argmax(excess))
        excess_max = float(excess[peak_tet])
        sigma_vm_max = float(sigma_vm[peak_tet])
        sigma_y_local = float(self.sigma_y_per_tet[peak_tet])

        result = dict(
            fired=False, reason=None, winner=None, winner_sign=0.0,
            alpha_scores=None, mode_amplitude=None,
            sigma_vm_max=sigma_vm_max, sigma_y_local=sigma_y_local,
            excess_max=excess_max,
            cumulative_max_disp=float(np.abs(self.cumulative_displacement).max()),
            fired_history=list(self.fired_modes),
            modes_remaining=num_modes - len(self.fired_modes),
        )

        if excess_max <= yield_excess_threshold:
            result["reason"] = "below_yield"
            return result

        if len(self.fired_modes) >= num_modes:
            result["reason"] = "all_modes_exhausted"
            return result

        m_diag = self._lumped_mass_diagonal()
        Mu_flat = m_diag * u_e_flat
        alpha = self.modes.T @ Mu_flat
        alpha_masked = alpha.copy()
        for m in self.fired_modes:
            alpha_masked[m] = 0.0

        if np.abs(alpha_masked).max() < 1.0e-12:
            result["reason"] = "no_matching_mode"
            result["alpha_scores"] = alpha
            return result

        winner = int(np.argmax(np.abs(alpha_masked)))
        winner_sign = float(np.sign(alpha_masked[winner]))

        if mode_amplitude is None:
            bbox = self.nodes.max(0) - self.nodes.min(0)
            mesh_diag = float(np.linalg.norm(bbox))
            mode_max = float(np.abs(self.modes[:, winner]).max())
            mode_amplitude = 0.05 * mesh_diag / max(mode_max, 1.0e-12)

        delta_u = winner_sign * mode_amplitude * self.modes[:, winner].reshape(
            self.num_vertices, 3
        )
        self.fired_modes.append(winner)
        self.cumulative_displacement += delta_u

        result.update(
            fired=True, reason="yielded", winner=winner, winner_sign=winner_sign,
            alpha_scores=alpha, mode_amplitude=float(mode_amplitude),
            cumulative_max_disp=float(np.abs(self.cumulative_displacement).max()),
            fired_history=list(self.fired_modes),
            modes_remaining=num_modes - len(self.fired_modes),
        )
        return result

    def _lumped_mass_diagonal(self):
        """Per-DOF lumped mass (3V,) — built lazily from vol_per_tet."""
        self._ensure_tet_operators()
        rho = (self.elastic_modes.rho_per_tet if self.elastic_modes is not None
               else np.full(self.num_tets, 1.0e3))
        m_vertex = np.zeros(self.num_vertices, dtype=np.float64)
        for t in range(self.num_tets):
            m_lump = float(rho[t]) * float(self.vol_per_tet[t]) / 4.0
            for i in range(4):
                m_vertex[int(self.elements[t, i])] += m_lump
        return np.repeat(m_vertex, 3)

    def __repr__(self):
        n_modes = 0 if self.modes is None else self.modes.shape[1]
        sy = self.sigma_y_per_tet
        return (
            f"PlasticModes(num_vertices={self.num_vertices}, "
            f"num_tets={self.num_tets}, num_modes={n_modes}, "
            f"num_elastic_modes={self.num_elastic_modes}, "
            f"sigma_y range=[{sy.min():.4g}, {sy.max():.4g}])"
        )
