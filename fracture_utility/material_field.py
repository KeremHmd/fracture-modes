"""Heterogeneous material specification for fracture modes"""

import numpy as np

_INTERFACE_RULES = ("min", "harmonic", "mean")

MATERIAL_CLASSES = ("rigid", "brittle", "elastic-plastic")
_MATERIAL_CLASS_TO_CODE = {name: code for code, name in enumerate(MATERIAL_CLASSES)}
_CODE_TO_MATERIAL_CLASS = {code: name for name, code in _MATERIAL_CLASS_TO_CODE.items()}


def _encode_material_class(labels, num_tets):
    if labels is None:
        return None
    arr = np.asarray(labels)
    if arr.dtype.kind in ("U", "S", "O"):
        codes = np.empty(arr.shape, dtype=np.int8)
        flat_arr = arr.reshape(-1)
        flat_codes = codes.reshape(-1)
        for i in range(flat_arr.size):
            key = str(flat_arr[i])
            if key not in _MATERIAL_CLASS_TO_CODE:
                raise ValueError(
                    f"Unknown material_class label {key!r}; expected one of "
                    f"{MATERIAL_CLASSES}"
                )
            flat_codes[i] = _MATERIAL_CLASS_TO_CODE[key]
    else:
        codes = arr.astype(np.int8).reshape(-1)
        unknown = np.setdiff1d(codes, list(_CODE_TO_MATERIAL_CLASS.keys()))
        if unknown.size:
            raise ValueError(
                f"Unknown material_class codes {unknown.tolist()}; expected "
                f"codes in {list(_CODE_TO_MATERIAL_CLASS.keys())}"
            )
    codes = codes.reshape(-1)
    if codes.shape != (num_tets,):
        raise ValueError(
            f"material_class must have shape ({num_tets},), got {codes.shape}"
        )
    return codes


def _validate_positive_field(name, arr, num_tets):
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.shape != (num_tets,):
        raise ValueError(f"{name} must have shape ({num_tets},), got {a.shape}")
    if np.any(a <= 0.0):
        raise ValueError(f"{name} values must be strictly positive")
    return a


class MaterialField:
    """Per-tet material specification for the heterogeneous pipeline"""

    def __init__(self, num_tets, G_c=None, rho=None, E=None,
                 sigma_y=None, sigma_c=None, material_class=None,
                 interface_rule="min"):
        self.num_tets = int(num_tets)

        if G_c is None:
            self.G_c = np.ones(self.num_tets, dtype=float)
        else:
            self.G_c = np.asarray(G_c, dtype=float).reshape(-1)
        if self.G_c.shape != (self.num_tets,):
            raise ValueError(
                f"G_c must have shape ({self.num_tets},), got {self.G_c.shape}"
            )
        if np.any(self.G_c <= 0.0):
            raise ValueError("G_c values must be strictly positive")

        if interface_rule not in _INTERFACE_RULES:
            raise ValueError(
                f"interface_rule must be one of {_INTERFACE_RULES}, "
                f"got {interface_rule!r}"
            )
        self.interface_rule = interface_rule

        self.rho = _validate_positive_field("rho", rho, self.num_tets)
        self.E = _validate_positive_field("E", E, self.num_tets)
        self.sigma_y = _validate_positive_field("sigma_y", sigma_y, self.num_tets)
        self.sigma_c = _validate_positive_field("sigma_c", sigma_c, self.num_tets)
        self.material_class = _encode_material_class(material_class, self.num_tets)

    @classmethod
    def homogeneous(cls, num_tets, G_c=1.0):
        """Build a homogeneous field - G_c constant across all tets"""
        return cls(num_tets, G_c=np.full(num_tets, float(G_c)))

    @classmethod
    def from_file(cls, path, num_tets):
        """Load a MaterialField from an .npz sidecar file"""
        data = np.load(path, allow_pickle=False)
        if "G_c" not in data.files:
            raise ValueError(
                f"{path}: missing 'G_c' array. Got keys: {data.files}"
            )
        G_c = data["G_c"]
        interface_rule = (
            str(data["interface_rule"])
            if "interface_rule" in data.files
            else "min"
        )
        rho = data["rho"] if "rho" in data.files else None
        E = data["E"] if "E" in data.files else None
        sigma_y = data["sigma_y"] if "sigma_y" in data.files else None
        sigma_c = data["sigma_c"] if "sigma_c" in data.files else None
        material_class = (
            data["material_class"]
            if "material_class" in data.files else None
        )
        return cls(
            num_tets=num_tets,
            G_c=G_c,
            rho=rho,
            E=E,
            sigma_y=sigma_y,
            sigma_c=sigma_c,
            material_class=material_class,
            interface_rule=interface_rule,
        )

    def to_file(self, path):
        """Save this MaterialField to an .npz sidecar file"""
        kwargs = {
            "G_c": self.G_c,
            "interface_rule": np.array(self.interface_rule),
        }
        if self.rho is not None:
            kwargs["rho"] = self.rho
        if self.E is not None:
            kwargs["E"] = self.E
        if self.sigma_y is not None:
            kwargs["sigma_y"] = self.sigma_y
        if self.sigma_c is not None:
            kwargs["sigma_c"] = self.sigma_c
        if self.material_class is not None:
            kwargs["material_class"] = self.material_class
        np.savez(path, **kwargs)

    def material_class_labels(self):
        """Return per-tet material_class as string labels, or None if unset."""
        if self.material_class is None:
            return None
        return np.array(
            [_CODE_TO_MATERIAL_CLASS[int(c)] for c in self.material_class],
            dtype=object,
        )

    def face_G_c(self, t1, t2):
        """Combine per-tet G_c values for one (t1, t2) pair via interface_rule."""
        g1 = float(self.G_c[t1])
        g2 = float(self.G_c[t2])
        if self.interface_rule == "min":
            return g1 if g1 < g2 else g2
        if self.interface_rule == "harmonic":
            return 2.0 * g1 * g2 / (g1 + g2)
        return 0.5 * (g1 + g2)

    def face_G_c_vectorized(self, t1_arr, t2_arr):
        """Same as face_G_c but vectorized over arrays of tet pairs."""
        g1 = self.G_c[t1_arr]
        g2 = self.G_c[t2_arr]
        if self.interface_rule == "min":
            return np.minimum(g1, g2)
        if self.interface_rule == "harmonic":
            return 2.0 * g1 * g2 / (g1 + g2)
        return 0.5 * (g1 + g2)

    def __repr__(self):
        extras = []
        if self.rho is not None:
            extras.append(f"rho=[{self.rho.min():.4g}, {self.rho.max():.4g}]")
        if self.E is not None:
            extras.append(f"E=[{self.E.min():.4g}, {self.E.max():.4g}]")
        if self.sigma_y is not None:
            extras.append(
                f"sigma_y=[{self.sigma_y.min():.4g}, {self.sigma_y.max():.4g}]"
            )
        if self.sigma_c is not None:
            extras.append(
                f"sigma_c=[{self.sigma_c.min():.4g}, {self.sigma_c.max():.4g}]"
            )
        if self.material_class is not None:
            unique, counts = np.unique(self.material_class, return_counts=True)
            cls_summary = ", ".join(
                f"{_CODE_TO_MATERIAL_CLASS[int(u)]}={int(c)}"
                for u, c in zip(unique, counts)
            )
            extras.append(f"material_class={{{cls_summary}}}")
        extras_str = (", " + ", ".join(extras)) if extras else ""
        return (f"MaterialField(num_tets={self.num_tets}, "
                f"G_c range=[{self.G_c.min():.4g}, {self.G_c.max():.4g}], "
                f"interface_rule={self.interface_rule!r}"
                f"{extras_str})")
