"""Per-tet plastic state and J2 return-mapping (small-strain, perfect plasticity).

Consumes per-tet total strain in Voigt engineering-shear form (matching the output
of ``ElasticModes.per_tet_strain``) and per-tet material parameters (yield stress
and Young's modulus, plus a global Poisson ratio). Updates a running plastic-strain
accumulator via the textbook radial-return algorithm for J2 flow with associative
plasticity and no hardening.

Voigt convention (matches ``elastic_modes.py`` throughout):
    strain: [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]
            where gamma_ij = 2 * eps_ij are engineering shear strains.
    stress: [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_zx]  (no factor of 2)
"""

import numpy as np


def _lame_from_E_nu(E, nu):
    """Isotropic Lame parameters (lambda, mu) from Young's modulus and Poisson ratio."""
    E = np.asarray(E, dtype=np.float64)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def _stress_from_elastic_strain(eps_e, lam, mu):
    """sigma = C(lambda, mu) @ eps_e in Voigt engineering-shear form.

    ``lam`` and ``mu`` are per-tet arrays (T,) or scalars. ``eps_e`` is (T, 6).
    Returns sigma of shape (T, 6).
    """
    tr = eps_e[:, 0] + eps_e[:, 1] + eps_e[:, 2]
    sigma = np.empty_like(eps_e)
    sigma[:, 0] = lam * tr + 2.0 * mu * eps_e[:, 0]
    sigma[:, 1] = lam * tr + 2.0 * mu * eps_e[:, 1]
    sigma[:, 2] = lam * tr + 2.0 * mu * eps_e[:, 2]
    sigma[:, 3] = mu * eps_e[:, 3]
    sigma[:, 4] = mu * eps_e[:, 4]
    sigma[:, 5] = mu * eps_e[:, 5]
    return sigma


def _deviator(sigma):
    """Deviatoric part of a per-tet Voigt stress array (T, 6)."""
    p = (sigma[:, 0] + sigma[:, 1] + sigma[:, 2]) / 3.0
    s = sigma.copy()
    s[:, 0] -= p
    s[:, 1] -= p
    s[:, 2] -= p
    return s, p


def _sigma_vm(s):
    """Von Mises equivalent stress from a per-tet Voigt deviator array (T, 6)."""
    s_dd = (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
            + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    return np.sqrt(1.5 * s_dd)


class PlasticState:
    """Cumulative per-tet plastic state for J2 flow (perfect plasticity).

    Stores plastic strain in Voigt engineering-shear form (matches the strain
    convention used by :class:`ElasticModes`). Because the flow is deviatoric,
    the axial components sum to zero within roundoff.
    """

    def __init__(self, num_tets):
        self.num_tets = int(num_tets)
        self.plastic_strain = np.zeros((self.num_tets, 6), dtype=np.float64)
        self.cumulative_plastic = np.zeros(self.num_tets, dtype=np.float64)

    def reset(self):
        """Clear the accumulator back to zero plastic strain."""
        self.plastic_strain.fill(0.0)
        self.cumulative_plastic.fill(0.0)

    def _broadcast_scalar(self, x, name):
        if np.isscalar(x):
            return np.full(self.num_tets, float(x))
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.shape != (self.num_tets,):
            raise ValueError(
                f"{name} must be scalar or shape ({self.num_tets},), "
                f"got {arr.shape}"
            )
        return arr

    def check_yield(self, total_strain, sigma_y, E, poisson_ratio,
                    include_current_plastic=True):
        """Return per-tet trial-stress von-Mises and yield mask.

        Does NOT update the state; use :meth:`apply_return_mapping` for that.

        Parameters
        ----------
        total_strain : (T, 6)
            Per-tet total strain in Voigt engineering-shear form.
        sigma_y : (T,) or scalar
            Per-tet yield stress (Pa).
        E : (T,) or scalar
            Per-tet Young's modulus (Pa).
        poisson_ratio : float
            Global Poisson ratio.
        include_current_plastic : bool
            If True (default), subtracts the current accumulator to get the
            elastic strain. If False, treats ``total_strain`` as the elastic
            strain directly (useful for one-shot checks).

        Returns
        -------
        sigma_vm_trial : (T,)
        yielded : (T,) bool
        """
        sy = self._broadcast_scalar(sigma_y, "sigma_y")
        Ea = self._broadcast_scalar(E, "E")
        lam, mu = _lame_from_E_nu(Ea, poisson_ratio)
        eps_e = (total_strain - self.plastic_strain
                 if include_current_plastic else total_strain)
        sigma_trial = _stress_from_elastic_strain(eps_e, lam, mu)
        s_trial, _ = _deviator(sigma_trial)
        vm = _sigma_vm(s_trial)
        return vm, vm > sy

    def apply_return_mapping(self, total_strain, sigma_y, E, poisson_ratio):
        """Advance the plastic state one step and return the returned stress.

        Standard radial-return algorithm for J2 with perfect plasticity:

            sigma_trial = C : (eps_total - eps_p^n)
            f = sigma_vm(sigma_trial) - sigma_y
            if f <= 0: elastic step, no update.
            else:
                delta_gamma = f / (3 * mu)
                delta_eps_p = delta_gamma * (3/2) * s_trial / sigma_vm_trial
                                (with a factor of 2 on the shear components
                                 when converting to engineering-shear Voigt)

        Returns
        -------
        sigma : (T, 6)     stress after the return step (Voigt)
        yielded : (T,)     bool mask of tets that yielded this step
        delta_gamma : (T,) plastic multiplier increment per tet
        """
        sy = self._broadcast_scalar(sigma_y, "sigma_y")
        Ea = self._broadcast_scalar(E, "E")
        lam, mu = _lame_from_E_nu(Ea, poisson_ratio)

        eps_e_trial = total_strain - self.plastic_strain
        sigma_trial = _stress_from_elastic_strain(eps_e_trial, lam, mu)
        s_trial, _ = _deviator(sigma_trial)
        vm_trial = _sigma_vm(s_trial)

        f = vm_trial - sy
        yielded = f > 0.0
        delta_gamma = np.zeros(self.num_tets, dtype=np.float64)
        delta_gamma[yielded] = f[yielded] / (3.0 * mu[yielded])

        factor = np.zeros(self.num_tets, dtype=np.float64)
        safe_vm = np.where(vm_trial > 0.0, vm_trial, 1.0)
        factor[yielded] = delta_gamma[yielded] * 1.5 / safe_vm[yielded]

        delta_eps_p = np.empty_like(self.plastic_strain)
        delta_eps_p[:, 0] = factor * s_trial[:, 0]
        delta_eps_p[:, 1] = factor * s_trial[:, 1]
        delta_eps_p[:, 2] = factor * s_trial[:, 2]
        delta_eps_p[:, 3] = factor * 2.0 * s_trial[:, 3]
        delta_eps_p[:, 4] = factor * 2.0 * s_trial[:, 4]
        delta_eps_p[:, 5] = factor * 2.0 * s_trial[:, 5]

        self.plastic_strain += delta_eps_p
        self.cumulative_plastic += delta_gamma

        eps_e_new = total_strain - self.plastic_strain
        sigma_new = _stress_from_elastic_strain(eps_e_new, lam, mu)
        return sigma_new, yielded, delta_gamma

    def deformation_gradient(self):
        """Return per-tet 3x3 plastic deformation-gradient tensors F_p = I + eps_p.

        Small-strain linearisation: F_p ~ I + eps_p_tensor. Intended for a
        fracture-mode pushforward.
        """
        ep = self.plastic_strain
        F = np.zeros((self.num_tets, 3, 3), dtype=np.float64)
        F[:, 0, 0] = 1.0 + ep[:, 0]
        F[:, 1, 1] = 1.0 + ep[:, 1]
        F[:, 2, 2] = 1.0 + ep[:, 2]
        F[:, 0, 1] = F[:, 1, 0] = 0.5 * ep[:, 3]
        F[:, 1, 2] = F[:, 2, 1] = 0.5 * ep[:, 4]
        F[:, 0, 2] = F[:, 2, 0] = 0.5 * ep[:, 5]
        return F

    def to_file(self, path):
        """Save the accumulator to an .npz sidecar."""
        np.savez(
            path,
            plastic_strain=self.plastic_strain,
            cumulative_plastic=self.cumulative_plastic,
        )

    @classmethod
    def from_file(cls, path, num_tets=None):
        """Load a previously-saved PlasticState .npz."""
        data = np.load(path, allow_pickle=False)
        ep = data["plastic_strain"]
        cp = data["cumulative_plastic"]
        T = ep.shape[0]
        if num_tets is not None and T != int(num_tets):
            raise ValueError(
                f"{path}: num_tets mismatch (file has {T}, expected {num_tets})"
            )
        obj = cls(T)
        obj.plastic_strain = ep.astype(np.float64).copy()
        obj.cumulative_plastic = cp.astype(np.float64).copy()
        return obj

    def __repr__(self):
        return (f"PlasticState(num_tets={self.num_tets}, "
                f"cumulative_plastic max={self.cumulative_plastic.max():.4g}, "
                f"|eps_p|_max={np.abs(self.plastic_strain).max():.4g})")
