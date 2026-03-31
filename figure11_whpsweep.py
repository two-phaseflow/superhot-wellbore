"""
WHP Sweep Analysis (Figure 11)
===============================

Sweeps wellhead pressure for selected high-P scenarios to show:
  (a) Surface enthalpy vs WHP with h_sat,vapor reference
  (b) Power output vs WHP with binary/flash markers

Intended for Section 4.3 (Design Implications).

Uses the dict-passing API:
    solve_flow_for_whp() -> coupled_model dict -> power_cycle_analysis()

Author: Samuel W. Scott
"""

import numpy as np
import os
import pickle
import warnings
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

from reservoir import (
    solve_flow_for_whp,
    rock_temperature_linear,
    depth_for_pressure,
)
from power_cycle import power_cycle_analysis

# ============================================================================
# PARAMETERS
# ============================================================================

FIXED_KXB = 1000.0     # md*m
T_SURFACE = 10.0        # C

# Scenarios: (P_res MPa, T_res C)
SCENARIOS = [
    (25, 450),
    (30, 450),
    (35, 450),
    (40, 450),
    (35, 500),
]

# WHP sweep range
WHP_RANGE = np.arange(2, 23, 1)  # 2 to 22 MPa in 1 MPa steps

WELL_PARAMS_TEMPLATE = {
    'diameter_m': 0.217,
    'delta_z_m': 10,
}

CACHE_FILE = 'whp_sweep_cache.pkl'

# ============================================================================
# FORMATTING
# ============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
})

# ============================================================================
# COMPUTATION
# ============================================================================

def run_whp_sweep(P_res, T_res, whp_range):
    """Sweep WHP for a single (P_res, T_res) scenario."""

    depth = int(depth_for_pressure(P_res))

    well_params = dict(WELL_PARAMS_TEMPLATE)
    well_params['depth_m'] = depth

    reservoir_params = {
        'transmissivity_md_m': FIXED_KXB,
        'drainage_radius_m': 500.0,
    }

    rock_temps = rock_temperature_linear(well_params, T_res, T_SURFACE)

    results = {
        'whp_MPa': [],
        'mass_flow_kg_s': [],
        'surface_enthalpy_MJ_kg': [],
        'surface_temp_C': [],
        'power_MW': [],
        'cycle_type': [],
    }

    previous_solution = None

    for whp in whp_range:
        print(f"  P_res={P_res}, T_res={T_res}, WHP={whp:.0f} MPa: ",
              end='')

        # Skip if WHP >= reservoir pressure
        if whp >= P_res - 1:
            print("WHP too close to P_res")
            continue

        try:
            cm = solve_flow_for_whp(
                target_whp_MPa=whp,
                P_reservoir_MPa=P_res,
                T_reservoir_C=T_res,
                rock_temperatures=rock_temps,
                reservoir_params=reservoir_params,
                well_params=well_params,
                previous_solution=previous_solution,
                verbose=True,
            )

            if not cm['success']:
                print("solver failed")
                continue

            if not cm.get('converged', False):
                print("did not converge")
                continue

            flow = cm['mass_flow_kgs']
            if np.isnan(flow) or flow <= 0:
                print("no flow")
                continue

            # Power calculation
            pc = power_cycle_analysis(cm)

            if pc['success'] and not np.isnan(pc['power_MWe']):
                power = pc['power_MWe']
                cycle = pc['cycle']
            else:
                power = np.nan
                cycle = 'unknown'

            results['whp_MPa'].append(cm['whp_MPa'])
            results['mass_flow_kg_s'].append(flow)
            results['surface_enthalpy_MJ_kg'].append(cm['h_surface_MJkg'])
            results['surface_temp_C'].append(cm['T_surface_C'])
            results['power_MW'].append(power)
            results['cycle_type'].append(cycle)

            previous_solution = {'whp': cm['whp_MPa'], 'flow': flow}

            print(f"q={flow:.1f} kg/s, h={cm['h_surface_MJkg']:.3f} MJ/kg, "
                  f"T={cm['T_surface_C']:.1f} C, P={power:.1f} MW, "
                  f"{cycle}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    for key in results:
        results[key] = np.array(results[key])

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(all_results, save_path='figure11_whp_sweep.png'):
    """Two-panel figure:
    (a) Surface enthalpy vs WHP with h_sat,vapor reference
    (b) Power output vs WHP with binary/flash markers
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color/style by scenario
    styles = {
    (25, 450): {'color': '#000000', 'ls': '-',  'marker': 'o',
                 'label': '25 MPa, 450 \u00B0C'},
    (30, 450): {'color': '#2ca02c', 'ls': '-',  'marker': 's',
                 'label': '30 MPa, 450 \u00B0C'},
    (35, 450): {'color': '#d62728', 'ls': '-',  'marker': '^',
                 'label': '35 MPa, 450 \u00B0C'},
    (40, 450): {'color': '#ff7f0e', 'ls': '-',  'marker': 'D',
                 'label': '40 MPa, 450 \u00B0C'},
    (35, 500): {'color': '#d62728', 'ls': '--', 'marker': '^',
                 'label': '35 MPa, 500 \u00B0C'},
    }

    # ------------------------------------------------------------------
    # Panel (a): Surface enthalpy vs WHP
    # ------------------------------------------------------------------

    # h_sat,vapor(WHP) reference curve
    whp_fine = np.linspace(2, 22, 200)
    h_sat_vap_ref = np.array([
        CP.PropsSI('H', 'P', p * 1e6, 'Q', 1, 'Water') / 1e6
        for p in whp_fine])
    ax1.plot(whp_fine, h_sat_vap_ref, 'k:', linewidth=2)

    # Set limits before computing text rotation
    ax1.set_xlim(2, 22)
    ax1.set_ylim(2.4, 3.2)
    fig.canvas.draw()

    # Rotated h_sat,vapor label
    idx = np.argmin(np.abs(whp_fine - 12))
    disp_p1 = ax1.transData.transform(
        (whp_fine[idx - 5], h_sat_vap_ref[idx - 5]))
    disp_p2 = ax1.transData.transform(
        (whp_fine[idx + 5], h_sat_vap_ref[idx + 5]))
    angle = np.degrees(np.arctan2(
        disp_p2[1] - disp_p1[1], disp_p2[0] - disp_p1[0]))
    ax1.text(whp_fine[idx], h_sat_vap_ref[idx]+0.01,
             '$h_\\mathrm{\\ sat,vapor}$',
             fontsize=14, ha='center', va='top', rotation=angle,
             rotation_mode='anchor', fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='none', alpha=1))

    # Shade two-phase region
    ax1.fill_between(whp_fine, 0, h_sat_vap_ref,
                     color='lightblue', alpha=0.2)
    ax1.text(8, 2.65, 'Two-phase\nat wellhead', fontsize=14,
             color='steelblue', ha='center')

    for (P_res, T_res), results in sorted(all_results.items()):
        if len(results['whp_MPa']) == 0:
            continue
        s = styles[(P_res, T_res)]
        ax1.plot(results['whp_MPa'], results['surface_enthalpy_MJ_kg'],
                 linestyle=s['ls'], color=s['color'], marker=s['marker'],
                 markersize=5, linewidth=1.5, label=s['label'])

    ax1.set_xlabel('Wellhead pressure (MPa)', fontsize=18)
    ax1.set_ylabel('Wellhead enthalpy (MJ/kg)', fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.legend(fontsize=12, loc='upper right', frameon=True,
               fancybox=False, edgecolor='gray',
               title='P$_{\\mathrm{res}}$, T$_{\\mathrm{res}}$',
               title_fontsize=12)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax1.text(0.02, 0.98, 'a)', transform=ax1.transAxes, fontsize=18,
             fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                       edgecolor='none', alpha=0.8))

    # ------------------------------------------------------------------
    # Panel (b): Power output vs WHP
    # ------------------------------------------------------------------

    for (P_res, T_res), results in sorted(all_results.items()):
        if len(results['whp_MPa']) == 0:
            continue
        s = styles[(P_res, T_res)]
        whp = results['whp_MPa']
        pwr = results['power_MW']
        cycle = np.array(results['cycle_type'])

        # Line
        ax2.plot(whp, pwr, linestyle=s['ls'], color=s['color'],
                 linewidth=1.5, label='_nolegend_')

        # Binary (filled) vs flash (open) markers
        is_binary = cycle == 'binary'
        is_flash = cycle == 'flash'

        if np.any(is_binary):
                ax2.plot(whp[is_binary], pwr[is_binary], s['marker'],
                         color=s['color'], markersize=5,
                         markerfacecolor=s['color'])
        if np.any(is_flash):
            ax2.plot(whp[is_flash], pwr[is_flash], s['marker'],
                     color=s['color'], markersize=5,
                     markerfacecolor='white',
                     markeredgecolor=s['color'],
                     markeredgewidth=1.5)

    ax2.set_xlabel('Wellhead pressure (MPa)', fontsize=18)
    ax2.set_ylabel('Power output (MWe)', fontsize=18)
    ax2.set_xlim(2, 22)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax2.text(0.02, 0.98, 'b)', transform=ax2.transAxes, fontsize=18,
             fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                       edgecolor='none', alpha=0.8))

    # Main legend
    from matplotlib.lines import Line2D
    legend_handles = []
    for (P_res, T_res) in sorted(all_results.keys()):
        s = styles[(P_res, T_res)]
        legend_handles.append(
            Line2D([0], [0], color=s['color'], linestyle=s['ls'],
                   marker=s['marker'], markersize=6,
                   markerfacecolor=s['color'], linewidth=1.5,
                   label=s['label']))
    
    leg1 = ax2.legend(handles=legend_handles, fontsize=12,
                      loc='lower left', frameon=True,
                      fancybox=False, edgecolor='gray',
                      title='P$_{\\mathrm{res}}$, T$_{\\mathrm{res}}$',
                      title_fontsize=12)
    ax2.add_artist(leg1)

    # Binary/flash legend
    from matplotlib.lines import Line2D
    binary_marker = Line2D([0], [0], marker='o', color='black',
                           markerfacecolor='black', markersize=6,
                           linestyle='None', label='Binary cycle')
    flash_marker = Line2D([0], [0], marker='o', color='black',
                          markerfacecolor='white',
                          markeredgecolor='black',
                          markeredgewidth=1.5, markersize=6,
                          linestyle='None', label='Flash cycle')
    ax2.legend(handles=[binary_marker, flash_marker],
               fontsize=12, loc='upper right', frameon=True,
               fancybox=False, edgecolor='gray')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    if os.path.exists(CACHE_FILE):
        print(f"Loading cached results from {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            all_results = pickle.load(f)
    else:
        all_results = {}

    runs_needed = [(P, T) for P, T in SCENARIOS
                   if (P, T) not in all_results]

    if runs_needed:
        print(f"Running {len(runs_needed)} scenarios...")
        for P_res, T_res in runs_needed:
            print(f"\n{'=' * 60}")
            print(f"P_res = {P_res} MPa, T_res = {T_res} C, "
                  f"kxb = {FIXED_KXB} md*m, "
                  f"depth = {int(depth_for_pressure(P_res))} m")
            print('=' * 60)
            results = run_whp_sweep(P_res, T_res, WHP_RANGE)
            all_results[(P_res, T_res)] = results

        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nCached results to {CACHE_FILE}")
    else:
        print("All results cached.")

    fig = plot_results(all_results)
    plt.show()

    print(f"\nTo rerun, delete: {CACHE_FILE}")