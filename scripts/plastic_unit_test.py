# 1-tet J2 return-mapping unit test. Applies a strain schedule to a single "tet",
# runs the return mapping every step, and prints the stress / plastic-strain response.
#
# Acceptance behaviour (perfect plasticity, uniaxial strain, load-then-unload):
#   - stress rises linearly with total axial strain (elastic ramp)
#   - stress plateaus at the yield stress once sigma_vm reaches sigma_y
#   - unloading is purely elastic (linear slope again)
#   - after full unloading, permanent plastic strain remains
#
# Usage:
#   python scripts\plastic_unit_test.py [--sigma-y V] [--E V] [--poisson-ratio V]
#                                       [--mode uniaxial-strain|pure-shear]
#                                       [--strain-schedule linear:START:END:N |
#                                                          load-unload:PEAK:N]
#                                       [--print-every N] [--plot FILE.png]

import argparse
import os
import sys

import numpy as np

from context import fracture_utility as fracture


parser = argparse.ArgumentParser(
    description="Single-tet J2 return-mapping unit test."
)
parser.add_argument("--sigma-y", type=float, default=2.0e8,
                    help="Yield stress (Pa). Default 2e8 (~mild steel).")
parser.add_argument("--E", type=float, default=2.0e11,
                    help="Young's modulus (Pa). Default 2e11.")
parser.add_argument("--poisson-ratio", type=float, default=0.3)
parser.add_argument("--mode", choices=["uniaxial-strain", "pure-shear"],
                    default="uniaxial-strain",
                    help="Which strain component to drive. "
                         "uniaxial-strain applies eps_xx with lateral constraint. "
                         "pure-shear applies gamma_xy only.")
parser.add_argument("--strain-schedule", default="load-unload:0.003:200",
                    help="Schedule format: linear:START:END:N (linear ramp of N steps), "
                         "or load-unload:PEAK:N (N/2 up 0 -> PEAK, N/2 down PEAK -> 0).")
parser.add_argument("--print-every", type=int, default=10,
                    help="Print a table row every N steps (0 = only summary).")
parser.add_argument("--plot", default=None,
                    help="Save a matplotlib figure to this path (PNG). "
                         "Requires matplotlib.")
args = parser.parse_args()


def parse_schedule(spec):
    parts = spec.split(":")
    kind = parts[0]
    if kind == "linear":
        if len(parts) != 4:
            raise SystemExit("linear expects: linear:START:END:N")
        start = float(parts[1]); end = float(parts[2]); n = int(parts[3])
        return np.linspace(start, end, n)
    if kind == "load-unload":
        if len(parts) != 3:
            raise SystemExit("load-unload expects: load-unload:PEAK:N")
        peak = float(parts[1]); n = int(parts[2])
        n_up = n // 2
        n_dn = n - n_up
        up = np.linspace(0.0, peak, n_up)
        dn = np.linspace(peak, 0.0, n_dn + 1)[1:]
        return np.concatenate([up, dn])
    raise SystemExit(f"Unknown schedule kind: {kind}")


schedule = parse_schedule(args.strain_schedule)
N_steps = schedule.size

mu = args.E / (2.0 * (1.0 + args.poisson_ratio))
lam = args.E * args.poisson_ratio / ((1.0 + args.poisson_ratio) * (1.0 - 2.0 * args.poisson_ratio))
if args.mode == "uniaxial-strain":
    eps_yield_theory = args.sigma_y / (2.0 * mu)
    axial_component = 0
    schedule_label = "eps_xx (uniaxial strain, lateral constrained)"
    stress_label = "sig_xx"
else:
    eps_yield_theory = args.sigma_y / (np.sqrt(3.0) * mu)
    axial_component = 3
    schedule_label = "gamma_xy (pure shear)"
    stress_label = "sig_xy"

print(f"Material: E={args.E:.4g} Pa, nu={args.poisson_ratio}, sigma_y={args.sigma_y:.4g} Pa")
print(f"Derived:  mu={mu:.4g} Pa, lambda={lam:.4g} Pa")
print(f"Mode:     {args.mode}")
print(f"Schedule: {args.strain_schedule}  ({N_steps} steps, peak={schedule.max():.4g})")
print(f"Predicted yield onset at {schedule_label} = {eps_yield_theory:.4g}")
print()

ps = fracture.PlasticState(num_tets=1)

total_strain_hist = np.zeros((N_steps, 6))
stress_hist = np.zeros((N_steps, 6))
sigma_vm_hist = np.zeros(N_steps)
cumulative_plastic_hist = np.zeros(N_steps)
plastic_axial_hist = np.zeros(N_steps)
yielded_hist = np.zeros(N_steps, dtype=bool)

first_yield_step = None
for i, val in enumerate(schedule):
    eps_total = np.zeros((1, 6))
    eps_total[0, axial_component] = val
    total_strain_hist[i] = eps_total[0]
    sigma_new, yielded, dgamma = ps.apply_return_mapping(
        eps_total, args.sigma_y, args.E, args.poisson_ratio,
    )
    stress_hist[i] = sigma_new[0]
    s = sigma_new[0].copy()
    p = (s[0] + s[1] + s[2]) / 3.0
    dev = s.copy(); dev[0] -= p; dev[1] -= p; dev[2] -= p
    sigma_vm_hist[i] = np.sqrt(1.5 * (dev[0]**2 + dev[1]**2 + dev[2]**2
                                      + 2.0 * (dev[3]**2 + dev[4]**2 + dev[5]**2)))
    cumulative_plastic_hist[i] = ps.cumulative_plastic[0]
    plastic_axial_hist[i] = ps.plastic_strain[0, axial_component]
    yielded_hist[i] = yielded[0]
    if first_yield_step is None and yielded[0]:
        first_yield_step = i


if args.print_every > 0:
    print(f"{'step':>5} {'total':>12} {'stress_axial':>14} {'sigma_vm':>12} "
          f"{'cum_ep':>12} {'ep_axial':>12} {'yielded':>8}")
    for i in range(0, N_steps, args.print_every):
        print(f"{i:5d} {total_strain_hist[i, axial_component]:12.4e} "
              f"{stress_hist[i, axial_component]:14.4e} "
              f"{sigma_vm_hist[i]:12.4e} "
              f"{cumulative_plastic_hist[i]:12.4e} "
              f"{plastic_axial_hist[i]:12.4e} "
              f"{'*' if yielded_hist[i] else '':>8}")
    print(f"{N_steps-1:5d} {total_strain_hist[-1, axial_component]:12.4e} "
          f"{stress_hist[-1, axial_component]:14.4e} "
          f"{sigma_vm_hist[-1]:12.4e} "
          f"{cumulative_plastic_hist[-1]:12.4e} "
          f"{plastic_axial_hist[-1]:12.4e} "
          f"{'*' if yielded_hist[-1] else '':>8}")


print("\n--- Summary ---")
if first_yield_step is None:
    print("No step yielded. (Try a larger --strain-schedule peak.)")
else:
    yield_val = total_strain_hist[first_yield_step, axial_component]
    print(f"First yield at step {first_yield_step}: "
          f"{schedule_label} = {yield_val:.4g}   "
          f"(theory: {eps_yield_theory:.4g}, ratio: {yield_val / eps_yield_theory:.4f})")
print(f"Peak sigma_vm reached:     {sigma_vm_hist.max():.4g} Pa  "
      f"(cap: {args.sigma_y:.4g} Pa, over-cap by {sigma_vm_hist.max() - args.sigma_y:.4g})")
print(f"Peak axial stress:         {stress_hist[:, axial_component].max():.4g} Pa")
print(f"Final total strain (axial): {total_strain_hist[-1, axial_component]:.4g}")
print(f"Final stress (axial):       {stress_hist[-1, axial_component]:.4g} Pa")
print(f"Final cumulative plastic:   {cumulative_plastic_hist[-1]:.4g}")
print(f"Final plastic strain (axial component): {plastic_axial_hist[-1]:.4g}")
print(f"Final plastic strain 6-vec: {ps.plastic_strain[0]}")


if args.plot is not None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n[skip] matplotlib not available; cannot write {args.plot}")
        sys.exit(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    total_axial = total_strain_hist[:, axial_component]
    stress_axial = stress_hist[:, axial_component]

    ax = axes[0]
    ax.plot(total_axial, stress_axial, "-o", markersize=2)
    ax.axhline(args.sigma_y, color="k", linestyle="--", alpha=0.3, label="sigma_y")
    ax.axhline(-args.sigma_y, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel(f"total {schedule_label}")
    ax.set_ylabel(f"{stress_label} (Pa)")
    ax.set_title("stress-strain (axial component)")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(total_axial, sigma_vm_hist, "-o", markersize=2)
    ax.axhline(args.sigma_y, color="k", linestyle="--", alpha=0.3, label="sigma_y")
    ax.set_xlabel(f"total {schedule_label}")
    ax.set_ylabel("sigma_vm (Pa)")
    ax.set_title("von Mises stress vs total strain")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    ax.plot(np.arange(N_steps), cumulative_plastic_hist, "-", label="cumulative plastic")
    ax.plot(np.arange(N_steps), plastic_axial_hist, "-", label="eps_p axial")
    ax.plot(np.arange(N_steps), total_axial, "--", alpha=0.4, label="total axial strain")
    ax.set_xlabel("step")
    ax.set_ylabel("strain")
    ax.set_title("plastic strain accumulation")
    ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    fig.savefig(args.plot, dpi=120)
    print(f"\nWrote plot: {args.plot}")
