"""
Wellbore Model Sensitivity: Heat-Loss Coefficient (U) and Roughness (epsilon)
==============================================================================

Generates a 2-panel figure for Section 2.2 (Methodology) demonstrating
that epsilon primarily controls deliverability (hydraulic, Eq. 4) while
U primarily controls wellhead temperature (thermal, Eq. 5). Uses a deep
scenario (35 MPa, 450 C, kxb = 1000 md*m, ~4 km) where both parameters
have their largest impact.

Panel order matches the equation order in Section 2.2:
  a) Casing roughness -> pressure equation (Eq. 4)
  b) Heat-loss coefficient -> energy equation (Eq. 5)

Results are cached via pickle to avoid rerunning. Delete the .pkl file
to force a recomputation.

Author: Samuel W. Scott
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import os
import CoolProp.CoolProp as CP

# Suppress runtime warnings during solver exploration
warnings.filterwarnings('ignore', category=RuntimeWarning)

from reservoir import (
    coupled_model, evaluate_at_flow_rates,
    rock_temperature_linear, depth_for_pressure,
    DEFAULT_RESERVOIR_PARAMS, DEFAULT_WELL_PARAMS,
)
from power_cycle import power_cycle_analysis

# ============================================================================
# SCENARIO PARAMETERS
# ============================================================================

# Deep scenario: most sensitive to U and epsilon
P_RES = 35.0       # MPa
T_RES = 450.0      # C
KXB = 1000.0       # md*m
WHP_FIXED = 10.0   # MPa (default parametric WHP)

# Parameter sweep values
U_VALUES = [1.0, 2.5, 5.0]         # W/m/K
U_LABELS = ['U = 1.0', 'U = 2.5 (default)', 'U = 5.0']
U_STYLES = ['--', '-', ':']
U_COLORS = ['#2166AC', '#333333', '#B2182B']  # blue, black, red

EPS_VALUES = [0.01e-3, 0.046e-3, 0.3e-3]    # m
EPS_LABELS = [r'$\epsilon$ = 0.01 mm', r'$\epsilon$ = 0.046 mm (default)',
              r'$\epsilon$ = 0.3 mm']
EPS_STYLES = ['--', '-', ':']
EPS_COLORS = ['#2166AC', '#333333', '#B2182B']

# Flow rate sweep
FLOW_MIN = 2.0
FLOW_MAX = 90.0
FLOW_STEP = 2.0

# Cache file
CACHE_FILE = 'sensitivity_U_epsilon_cache.pkl'

# ============================================================================
# FORMATTING
# ============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.0,
    'figure.dpi': 150,
})


# ============================================================================
# COMPUTATION
# ============================================================================

def run_sweep(P_res, T_res, kxb, U, epsilon):
    """
    Sweep mass flow rates through the coupled model for given U, epsilon.

    Returns arrays of (flow, whp, wht, h_surface) for all successful runs,
    sorted by descending flow rate.
    """
    depth = depth_for_pressure(P_res)

    well_params = dict(DEFAULT_WELL_PARAMS)
    well_params['depth_m'] = int(depth)
    well_params['heat_loss_factor'] = U
    well_params['roughness_m'] = epsilon

    reservoir_params = dict(DEFAULT_RESERVOIR_PARAMS)
    reservoir_params['transmissivity_md_m'] = kxb

    rock_temps = rock_temperature_linear(well_params, T_res)

    flows = np.arange(FLOW_MIN, FLOW_MAX + FLOW_STEP, FLOW_STEP)

    results = evaluate_at_flow_rates(
        flows, P_res, T_res, rock_temps, reservoir_params, well_params)

    ok = results['success']
    flow = results['flow_kg_s'][ok]
    whp = results['whp_MPa'][ok]
    wht = results['T_surface_C'][ok]
    h_surface = results['h_surface_MJkg'][ok]

    # Sort by descending flow rate — this gives monotonically
    # increasing WHP along the main operating branch.
    order = np.argsort(-flow)

    return {
        'flow': flow[order],
        'whp': whp[order],
        'wht': wht[order],
        'h_surface': h_surface[order],
    }


def compute_all():
    """Run all sweeps and return (u_results, eps_results) dicts."""
    print(f"Scenario: P_res = {P_RES} MPa, T_res = {T_RES} C, "
          f"kxb = {KXB} md*m")
    print(f"Depth = {depth_for_pressure(P_RES):.0f} m")
    print()

    # U sensitivity (fix epsilon at default)
    print("=== U sensitivity (epsilon = 0.046 mm) ===")
    u_results = {}
    for U in U_VALUES:
        print(f"  Running U = {U} W/m/K ...")
        u_results[U] = run_sweep(P_RES, T_RES, KXB, U, 0.046e-3)
        n = len(u_results[U]['flow'])
        if n > 0:
            print(f"    {n} valid points, WHP {u_results[U]['whp'].min():.1f}-"
                  f"{u_results[U]['whp'].max():.1f} MPa, "
                  f"WHT {u_results[U]['wht'].min():.0f}-"
                  f"{u_results[U]['wht'].max():.0f} C")
        else:
            print(f"    WARNING: no valid points returned")

    # Roughness sensitivity (fix U at default)
    print("\n=== Roughness sensitivity (U = 2.5 W/m/K) ===")
    eps_results = {}
    for eps in EPS_VALUES:
        print(f"  Running epsilon = {eps*1e3:.3f} mm ...")
        eps_results[eps] = run_sweep(P_RES, T_RES, KXB, 2.5, eps)
        n = len(eps_results[eps]['flow'])
        if n > 0:
            print(f"    {n} valid points, WHP {eps_results[eps]['whp'].min():.1f}-"
                  f"{eps_results[eps]['whp'].max():.1f} MPa, "
                  f"flow {eps_results[eps]['flow'].min():.1f}-"
                  f"{eps_results[eps]['flow'].max():.1f} kg/s")
        else:
            print(f"    WARNING: no valid points returned")

    return u_results, eps_results


def _sort_by_flow(results_dict):
    """Re-sort all result arrays by descending flow rate."""
    for key in results_dict:
        r = results_dict[key]
        if len(r['flow']) > 0:
            order = np.argsort(-r['flow'])
            for field in ['flow', 'whp', 'wht', 'h_surface']:
                r[field] = r[field][order]


def load_or_compute():
    """Load cached results or compute and cache."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached results from {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f"  U cases: {list(data['u_results'].keys())}")
        print(f"  eps cases: {[f'{k*1e3:.3f} mm' for k in data['eps_results'].keys()]}")
        u_results = data['u_results']
        eps_results = data['eps_results']
        _sort_by_flow(u_results)
        _sort_by_flow(eps_results)
        return u_results, eps_results
    else:
        u_results, eps_results = compute_all()
        print(f"\nCaching results to {CACHE_FILE}")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'u_results': u_results, 'eps_results': eps_results,
                         'P_RES': P_RES, 'T_RES': T_RES, 'KXB': KXB}, f)
        return u_results, eps_results


# ============================================================================
# PLOTTING
# ============================================================================

def saturation_temperature_curve(P_min_MPa=1.0, P_max_MPa=22.0, n=200):
    """Compute T_sat(P) for plotting the boiling curve."""
    P_vals = np.linspace(P_min_MPa, P_max_MPa, n)
    T_vals = np.full_like(P_vals, np.nan)
    for i, P in enumerate(P_vals):
        try:
            T_vals[i] = CP.PropsSI('T', 'P', P * 1e6, 'Q', 0, 'Water') - 273.15
        except Exception:
            pass
    return P_vals, T_vals


def make_figure(u_results, eps_results):
    """Generate the 2-panel sensitivity figure."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------------------------------------------------
    # Panel (a): mass flow vs WHP for different epsilon (Eq. 4)
    # ------------------------------------------------------------------
    for i, eps in enumerate(EPS_VALUES):
        r = eps_results[eps]
        if len(r['flow']) == 0:
            continue
        ax1.plot(r['whp'], r['flow'],
                 linestyle=EPS_STYLES[i], color=EPS_COLORS[i],
                 label=EPS_LABELS[i])

    ax1.set_xlabel('Wellhead pressure (MPa)')
    ax1.set_ylabel('Mass flow rate (kg/s)')
    ax1.legend(loc='lower left', frameon=True, fancybox=False,
               edgecolor='gray')
    ax1.grid(True, alpha=0.3)

    # Panel label
    ax1.text(0.02, 0.98, 'a)', transform=ax1.transAxes,
             fontsize=18, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='none', alpha=0.8))

    # Tight axis limits
    all_whp_e = np.concatenate([r['whp'] for r in eps_results.values() if len(r['whp']) > 0])
    all_flow_e = np.concatenate([r['flow'] for r in eps_results.values() if len(r['flow']) > 0])
    ax1.set_xlim(np.floor(all_whp_e.min()) - 1, np.ceil(all_whp_e.max()) + 1)
    ax1.set_ylim(0, np.ceil(all_flow_e.max() / 10) * 10 + 5)

    # ------------------------------------------------------------------
    # Panel (b): WHT vs WHP for different U (Eq. 5)
    # ------------------------------------------------------------------
    for i, U in enumerate(U_VALUES):
        r = u_results[U]
        if len(r['flow']) == 0:
            continue
        ax2.plot(r['whp'], r['wht'],
                 linestyle=U_STYLES[i], color=U_COLORS[i],
                 label=U_LABELS[i])

    # Saturation curve
    P_sat, T_sat = saturation_temperature_curve()
    ax2.plot(P_sat, T_sat, color='gray', ls='--', lw=1.0,
             label=r'T$_{\mathrm{sat}}$(P)')

    ax2.set_xlabel('Wellhead pressure (MPa)')
    ax2.set_ylabel('Wellhead temperature (\u00B0C)')
    ax2.legend(loc='lower right', frameon=True, fancybox=False,
               edgecolor='gray')
    ax2.grid(True, alpha=0.3)

    # Panel label
    ax2.text(0.02, 0.98, 'b)', transform=ax2.transAxes,
             fontsize=18, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='none', alpha=0.8))

    # Tight axis limits
    all_whp_u = np.concatenate([r['whp'] for r in u_results.values() if len(r['whp']) > 0])
    all_wht_u = np.concatenate([r['wht'] for r in u_results.values() if len(r['wht']) > 0])
    ax2.set_xlim(np.floor(all_whp_u.min()) - 1, np.ceil(all_whp_u.max()) + 1)
    ax2.set_ylim(np.floor(all_wht_u.min() / 10) * 10 - 10,
                 np.ceil(all_wht_u.max() / 10) * 10 + 10)

    fig.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    u_results, eps_results = load_or_compute()

    # Print summary at WHP closest to 10 MPa
    print("\n=== Summary at WHP closest to 10 MPa ===")
    for U in U_VALUES:
        r = u_results[U]
        if len(r['flow']) > 0:
            idx = np.argmin(np.abs(r['whp'] - WHP_FIXED))
            print(f"  U={U}: flow={r['flow'][idx]:.1f} kg/s, "
                  f"WHP={r['whp'][idx]:.1f} MPa, WHT={r['wht'][idx]:.0f} C")

    for eps in EPS_VALUES:
        r = eps_results[eps]
        if len(r['flow']) > 0:
            idx = np.argmin(np.abs(r['whp'] - WHP_FIXED))
            print(f"  eps={eps*1e3:.3f} mm: flow={r['flow'][idx]:.1f} kg/s, "
                  f"WHP={r['whp'][idx]:.1f} MPa, WHT={r['wht'][idx]:.0f} C")

    # Generate figure
    fig = make_figure(u_results, eps_results)

    # Save
    fig.savefig('sensitivity_U_epsilon.png', dpi=300, bbox_inches='tight')
    fig.savefig('sensitivity_U_epsilon.pdf', bbox_inches='tight')
    print(f"\nFigure saved to sensitivity_U_epsilon.png/pdf")
    plt.show()


if __name__ == '__main__':
    main()
