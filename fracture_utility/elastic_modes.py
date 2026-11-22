"""Linear-elastic modal analysis on a tet mesh.

Assembles the standard linear-elasticity stiffness matrix K and a lumped mass
matrix M from the tet mesh, honouring per-tet Young's modulus E and density rho
from an optional MaterialField (with fallback defaults). Solves the generalised
eigenvalue problem K phi = omega^2 M phi via shift-invert around zero, drops the
6 near-zero rigid-body modes, and returns a small dense subspace of the
lowest-frequency deformation modes.

The subspace produced here is used for the runtime elastic projection, the
yield check and the plastic modes.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


_RIGID_BODY_COUNT = 6


def _elasticity_matrix(E, nu):
    """6x6 isotropic linear-elasticity C matrix in Voigt notation.

    Strain vector layout: [exx, eyy, ezz, gamma_xy, gamma_yz, gamma_zx]
    where gamma_ij = 2 * epsilon_ij are engineering shear strains.
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[1, 0] = lam
    C[0, 2] = C[2, 0] = lam
    C[1, 2] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _tet_shape_gradients(v0, v1, v2, v3):
    """Constant per-tet shape function gradients (3 per vertex) and signed volume.

    Returns (grads, vol) where grads is (4, 3) with grads[i] = grad N_i,
    and vol is the signed tet volume.
    """
    J = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
    detJ = np.linalg.det(J)
    vol = detJ / 6.0
    Jinv = np.linalg.inv(J)
    ref_grads = np.array([[-1.0, -1.0, -1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
    grads = ref_grads @ Jinv
    return grads, vol


def _strain_displacement_B(grads):
    """6x12 strain-displacement matrix B for a linear tet.

    Column layout: [ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ux3, uy3, uz3].
    Row layout matches Voigt strain: [exx, eyy, ezz, gxy, gyz, gzx].
    """
    B = np.zeros((6, 12))
    for i in range(4):
        gx, gy, gz = grads[i]
        c = 3 * i
        B[0, c + 0] = gx
        B[1, c + 1] = gy
        B[2, c + 2] = gz
        B[3, c + 0] = gy
        B[3, c + 1] = gx
        B[4, c + 1] = gz
        B[4, c + 2] = gy
        B[5, c + 0] = gz
        B[5, c + 2] = gx
    return B


def assemble_K_M(nodes, elements, E_per_tet, rho_per_tet, poisson_ratio):
    """Assemble linear-elasticity K, lumped M, and cache per-tet B / vol / gdof.

    Parameters
    ----------
    nodes : (V, 3)
        Vertex positions.
    elements : (T, 4)
        Tet-vertex indices.
    E_per_tet : (T,)
        Per-tet Young's modulus (Pa).
    rho_per_tet : (T,)
        Per-tet density (kg / m^3).
    poisson_ratio : float
        Global Poisson ratio (same for all tets in this stage).

    Returns
    -------
    K : csr_matrix of shape (3V, 3V)
    M : csr_matrix of shape (3V, 3V)  (diagonal, lumped)
    B_per_tet : (T, 6, 12) strain-displacement matrices (cached for reuse)
    vol_per_tet : (T,) positive tet volumes
    gdof_per_tet : (T, 12) global DOF indices (row-major x,y,z per vertex)
    """
    V = nodes.shape[0]
    T = elements.shape[0]
    dof = 3 * V

    rows = np.empty(T * 12 * 12, dtype=np.int64)
    cols = np.empty(T * 12 * 12, dtype=np.int64)
    vals = np.empty(T * 12 * 12, dtype=np.float64)

    m_diag = np.zeros(dof, dtype=np.float64)

    B_per_tet = np.empty((T, 6, 12), dtype=np.float64)
    vol_per_tet = np.empty(T, dtype=np.float64)
    gdof_per_tet = np.empty((T, 12), dtype=np.int64)

    C_unit = _elasticity_matrix(1.0, poisson_ratio)

    for t in range(T):
        v_idx = elements[t]
        v0, v1, v2, v3 = nodes[v_idx]
        grads, vol = _tet_shape_gradients(v0, v1, v2, v3)
        if vol < 0.0:
            vol = -vol
            grads = -grads
        B = _strain_displacement_B(grads)
        C = C_unit * E_per_tet[t]
        Ke = vol * (B.T @ C @ B)

        gdof = np.empty(12, dtype=np.int64)
        for i in range(4):
            gdof[3 * i:3 * i + 3] = 3 * v_idx[i] + np.arange(3)
        base = t * 144
        for i in range(12):
            rr = gdof[i]
            for j in range(12):
                idx = base + i * 12 + j
                rows[idx] = rr
                cols[idx] = gdof[j]
                vals[idx] = Ke[i, j]

        m_lump = rho_per_tet[t] * vol / 4.0
        for i in range(4):
            base_dof = 3 * v_idx[i]
            m_diag[base_dof + 0] += m_lump
            m_diag[base_dof + 1] += m_lump
            m_diag[base_dof + 2] += m_lump

        B_per_tet[t] = B
        vol_per_tet[t] = vol
        gdof_per_tet[t] = gdof

    K = coo_matrix((vals, (rows, cols)), shape=(dof, dof)).tocsr()
    K = 0.5 * (K + K.T)

    from scipy.sparse import diags
    M = diags(m_diag, format="csr")

    return K, M, B_per_tet, vol_per_tet, gdof_per_tet


class ElasticModes:
    """Precomputed linear-elastic modal subspace for a tet mesh.

    All plastic-pipeline knobs are optional. If ``materials`` is None or has
    no E / rho fields, the class uses the ``default_E`` / ``default_rho``
    scalars for every tet, so the class is usable with any tet mesh even when
    no MaterialField has been authored yet.
    """

    def __init__(self, nodes, elements, materials=None,
                 poisson_ratio=0.3,
                 default_E=1.0e9, default_rho=1.0e3):
        self.nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        self.elements = np.ascontiguousarray(elements, dtype=np.int64)
        self.num_vertices = int(self.nodes.shape[0])
        self.num_tets = int(self.elements.shape[0])
        self.poisson_ratio = float(poisson_ratio)
        self.default_E = float(default_E)
        self.default_rho = float(default_rho)

        if materials is not None and getattr(materials, "E", None) is not None:
            self.E_per_tet = np.asarray(materials.E, dtype=np.float64).copy()
        else:
            self.E_per_tet = np.full(self.num_tets, self.default_E)
        if materials is not None and getattr(materials, "rho", None) is not None:
            self.rho_per_tet = np.asarray(materials.rho, dtype=np.float64).copy()
        else:
            self.rho_per_tet = np.full(self.num_tets, self.default_rho)

        self.K = None
        self.M = None
        self.B_per_tet = None
        self.vol_per_tet = None
        self.gdof_per_tet = None
        self.eigenvalues = None
        self.frequencies_hz = None
        self.modes = None

    def assemble(self):
        """Assemble K, M, and cache per-tet B / vol / DOF indices."""
        (self.K, self.M, self.B_per_tet,
         self.vol_per_tet, self.gdof_per_tet) = assemble_K_M(
            self.nodes, self.elements,
            self.E_per_tet, self.rho_per_tet,
            self.poisson_ratio,
        )
        return self

    def compute_modes(self, num_modes=30, drop_rigid_body=True,
                      shift_sigma=-1.0, tol=0.0, max_iter=None,
                      verbose=False):
        """Solve the generalised eigenvalue problem K phi = omega^2 M phi.

        Uses shift-invert ARPACK around ``shift_sigma`` to get the lowest-frequency
        modes. The default ``shift_sigma=-1.0`` guarantees K - sigma*M = K + M is
        strictly positive definite (K is only positive semi-definite because of
        the 6 rigid-body modes with omega^2 = 0). The 6 rigid-body modes are
        dropped by default.

        Parameters
        ----------
        num_modes : int
            Number of *deformation* modes to keep (after dropping rigid-body).
        drop_rigid_body : bool
            If True, request ``num_modes + 6`` eigenpairs and drop the 6
            smallest by absolute value.
        """
        if self.K is None:
            self.assemble()

        k_req = num_modes + (_RIGID_BODY_COUNT if drop_rigid_body else 0)
        if k_req >= self.K.shape[0]:
            raise ValueError(
                f"Requested {k_req} eigenpairs but the system has only "
                f"{self.K.shape[0]} DOFs; reduce num_modes."
            )

        eigvals, eigvecs = eigsh(
            self.K, k=k_req, M=self.M,
            sigma=shift_sigma, which="LM",
            tol=tol, maxiter=max_iter,
        )

        order = np.argsort(np.abs(eigvals))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if drop_rigid_body:
            eigvals = eigvals[_RIGID_BODY_COUNT:]
            eigvecs = eigvecs[:, _RIGID_BODY_COUNT:]

        eigvals = np.clip(eigvals, 0.0, None)
        self.eigenvalues = eigvals
        self.frequencies_hz = np.sqrt(eigvals) / (2.0 * np.pi)
        self.modes = eigvecs
        if verbose:
            print(f"[ElasticModes] kept {eigvals.size} modes; "
                  f"omega^2 range [{eigvals.min():.4g}, {eigvals.max():.4g}]; "
                  f"freq range [{self.frequencies_hz.min():.4g}, "
                  f"{self.frequencies_hz.max():.4g}] Hz")
        return self

    def mode_displacement(self, mode_index, amplitude=1.0):
        """Return per-vertex displacement (V, 3) for a single mode."""
        if self.modes is None:
            raise RuntimeError("Call compute_modes() first.")
        phi = self.modes[:, mode_index] * amplitude
        return phi.reshape(self.num_vertices, 3)

    def to_file(self, path):
        """Save modes + metadata to an .npz sidecar."""
        if self.modes is None:
            raise RuntimeError("Call compute_modes() first.")
        np.savez(
            path,
            nodes=self.nodes,
            elements=self.elements,
            eigenvalues=self.eigenvalues,
            frequencies_hz=self.frequencies_hz,
            modes=self.modes,
            E_per_tet=self.E_per_tet,
            rho_per_tet=self.rho_per_tet,
            poisson_ratio=np.array(self.poisson_ratio),
        )

    @classmethod
    def from_file(cls, path):
        """Load a previously-computed ElasticModes bundle."""
        data = np.load(path, allow_pickle=False)
        obj = cls(
            nodes=data["nodes"],
            elements=data["elements"],
            poisson_ratio=float(data["poisson_ratio"]),
        )
        obj.E_per_tet = data["E_per_tet"]
        obj.rho_per_tet = data["rho_per_tet"]
        obj.eigenvalues = data["eigenvalues"]
        obj.frequencies_hz = data["frequencies_hz"]
        obj.modes = data["modes"]
        return obj

    def _ensure_tet_operators(self):
        """Lazily assemble per-tet B / vol / DOF caches if missing.

        Needed when the object was loaded via ``from_file`` (which restores
        K/M-derived quantities but not the per-tet operators).
        """
        if self.B_per_tet is None:
            self.assemble()

    def per_tet_strain(self, displacement):
        """Compute per-tet Voigt strain from a per-vertex displacement field.

        Parameters
        ----------
        displacement : (V, 3) or (3V,)
            Per-vertex displacement field in world coordinates.

        Returns
        -------
        strain : (T, 6)
            Voigt strain [exx, eyy, ezz, gxy, gyz, gzx] per tet, using
            engineering shear strains (gamma_ij = 2 * epsilon_ij).
        """
        self._ensure_tet_operators()
        u = np.asarray(displacement, dtype=np.float64).reshape(-1)
        if u.size != 3 * self.num_vertices:
            raise ValueError(
                f"Displacement has {u.size} entries, expected {3 * self.num_vertices}."
            )
        u_local = u[self.gdof_per_tet]
        strain = np.einsum("tij,tj->ti", self.B_per_tet, u_local)
        return strain

    def per_tet_von_mises_strain(self, strain):
        """Convert per-tet Voigt strain (T, 6) to a scalar von-Mises equivalent.

        eps_vm = sqrt( 2/3 * (e_dev : e_dev) ) where e_dev is the deviatoric part.
        Here we work in engineering shear (gamma_ij = 2 e_ij), which is why the
        shear terms are divided by 2 before forming e_dev.
        """
        strain = np.asarray(strain, dtype=np.float64)
        exx, eyy, ezz = strain[:, 0], strain[:, 1], strain[:, 2]
        exy = 0.5 * strain[:, 3]
        eyz = 0.5 * strain[:, 4]
        ezx = 0.5 * strain[:, 5]
        mean = (exx + eyy + ezz) / 3.0
        dxx = exx - mean
        dyy = eyy - mean
        dzz = ezz - mean
        j2 = 0.5 * (dxx * dxx + dyy * dyy + dzz * dzz) + (
            exy * exy + eyz * eyz + ezx * ezx
        )
        return np.sqrt((4.0 / 3.0) * j2)

    def nodal_force_from_impact(self, contact_point, direction, magnitude,
                                spread_radius=0.0):
        """Distribute an impact force to mesh vertices.

        With ``spread_radius == 0``, the entire force lands on the single
        vertex nearest the contact point. With ``spread_radius > 0``, the
        force is spread over vertices within that radius using Gaussian
        weights (sigma = spread_radius / 2) then re-normalised to preserve
        the total force magnitude.

        Returns
        -------
        f : (3V,) global nodal-force vector.
        """
        contact = np.asarray(contact_point, dtype=np.float64).reshape(3)
        dir_vec = np.asarray(direction, dtype=np.float64).reshape(3)
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-12:
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

    def project_impact(self, contact_point, direction, magnitude,
                       num_frames=60, dt=1.0 / 240.0,
                       rayleigh_alpha=0.5, rayleigh_beta=1.0e-5,
                       spread_radius=0.0):
        """Impulse-driven modal free-vibration response.

        Treats the impact as an instantaneous impulse at t = 0 with total
        vector F = magnitude * direction applied at ``contact_point``, then
        integrates each mode independently as a damped harmonic oscillator
        with Rayleigh damping (2 zeta_k omega_k = alpha + beta * omega_k^2).

        Because eigsh returns M-orthonormal modes (phi_i^T M phi_j = delta_ij),
        the modal mass is 1 for every mode, so the initial modal velocity is
        directly q_dot_k(0+) = phi_k^T F.

        Returns
        -------
        displacements : (num_frames, V, 3)
        strains       : (num_frames, T, 6)   Voigt strain per tet per frame
        times         : (num_frames,)        seconds
        """
        if self.modes is None:
            raise RuntimeError("Call compute_modes() first.")

        Phi = self.modes
        num_modes = Phi.shape[1]

        f_global = self.nodal_force_from_impact(
            contact_point, direction, magnitude, spread_radius=spread_radius,
        )
        f_modal = Phi.T @ f_global

        omega_sq = np.clip(self.eigenvalues, 1e-16, None)
        omega = np.sqrt(omega_sq)
        two_zeta_omega = rayleigh_alpha + rayleigh_beta * omega_sq
        zeta = 0.5 * two_zeta_omega / omega
        overdamped = zeta >= 1.0
        omega_d = np.zeros_like(omega)
        omega_d[~overdamped] = omega[~overdamped] * np.sqrt(
            1.0 - zeta[~overdamped] ** 2
        )

        times = np.arange(num_frames, dtype=np.float64) * dt
        q = np.zeros((num_frames, num_modes), dtype=np.float64)

        under = ~overdamped
        if under.any():
            envelope = np.exp(-zeta[under][None, :] * omega[under][None, :] * times[:, None])
            q[:, under] = (
                (f_modal[under] / omega_d[under])[None, :]
                * envelope
                * np.sin(omega_d[under][None, :] * times[:, None])
            )

        over = overdamped
        if over.any():
            zt = zeta[over]
            wt = omega[over]
            root = np.sqrt(np.maximum(zt * zt - 1.0, 0.0))
            r1 = -wt * (zt - root)
            r2 = -wt * (zt + root)
            A = f_modal[over] / (r1 - r2)
            q[:, over] = A[None, :] * (
                np.exp(r1[None, :] * times[:, None])
                - np.exp(r2[None, :] * times[:, None])
            )

        displacements = np.empty((num_frames, self.num_vertices, 3), dtype=np.float64)
        strains = np.empty((num_frames, self.num_tets, 6), dtype=np.float64)
        for i in range(num_frames):
            u_flat = Phi @ q[i]
            displacements[i] = u_flat.reshape(self.num_vertices, 3)
            strains[i] = self.per_tet_strain(u_flat)

        return displacements, strains, times

    def __repr__(self):
        n_modes = 0 if self.modes is None else self.modes.shape[1]
        return (f"ElasticModes(num_vertices={self.num_vertices}, "
                f"num_tets={self.num_tets}, "
                f"poisson_ratio={self.poisson_ratio}, "
                f"num_modes={n_modes})")
