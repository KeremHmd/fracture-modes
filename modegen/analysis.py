"""Static elastic impact analysis shared by routers and responders (numpy only).

Given a per-vertex force field on a piece's cage, computes the quasi-static
elastic response u = K^{-1} f through the piece's precomputed elastic
subspace, the per-tet strain, and the von-Mises stress, and reports the
peak-excess tet with its material class and thresholds.
"""

import numpy as np

from .interfaces import von_mises_from_strain

_CLASSES = ("rigid", "brittle", "elastic-plastic")


def static_elastic_analysis(piece, force, poisson_ratio=0.3):
    """Return the analysis dict, or ``None`` when the piece cannot be analysed
    (no elastic bundle, or no ``sigma_y`` in its material)."""
    em = getattr(piece, "elastic_modes", None)
    mats = getattr(piece, "materials", None)
    if em is None or em.modes is None or mats is None or mats.sigma_y is None:
        return None
    f = np.asarray(force, dtype=float).reshape(-1)
    Phi = em.modes
    w2 = np.clip(em.eigenvalues, 1.0e-16, None)
    u_flat = Phi @ ((Phi.T @ f) / w2)
    eps = em.per_tet_strain(u_flat)
    sigma_vm = von_mises_from_strain(eps, em.E_per_tet, poisson_ratio)
    sigma_y = np.asarray(mats.sigma_y, dtype=float)
    sigma_c = (np.asarray(mats.sigma_c, dtype=float) if mats.sigma_c is not None
               else np.full_like(sigma_y, np.inf))
    excess_y = sigma_vm - sigma_y
    excess_c = sigma_vm - sigma_c
    peak = int(np.argmax(np.maximum(excess_y, excess_c)))
    mc = ("elastic-plastic" if mats.material_class is None
          else _CLASSES[int(mats.material_class[peak])])
    return dict(
        u_flat=u_flat, strain=eps, sigma_vm=sigma_vm,
        peak_tet=peak, material_class=mc,
        sigma_vm_max=float(sigma_vm[peak]),
        sigma_y_local=float(sigma_y[peak]),
        sigma_c_local=float(sigma_c[peak]),
        excess_y=float(excess_y[peak]),
        excess_c=float(excess_c[peak]),
    )
