"""
Wellhead Velocity and Mass Flow Rate vs Reservoir Pressure (Figure 12)
=======================================================================

Compares two casing diameters (9 5/8" and 13 3/8") showing:
  - Mass flow rate (dashed lines) vs reservoir pressure
  - Wellhead velocity (solid lines with markers) vs reservoir pressure
  - 50 m/s erosional velocity reference

Fixed transmissivity = 2000 md*m, three reservoir temperatures.

Uses the dict-passing API:
    solve_flow_for_whp() -> coupled_model dict

Author: Samuel W. Scott
"""

import numpy as np
import os
import pickle
import warnings
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from matplotlib.lines import Line2D

from reservoir import (
    solve_flow_for_whp,
    rock_temperature_linear,
    depth_for_pressure,
)

# ============================================================================
# PARAMETERS
# ============================================================================

FIXED_WHP = 10.0          # MPa
FIXED_KXB = 2000.0        # md*m
T_SURFACE = 10.0           # C
V_CRIT = 50                # m/s erosional velocity limit

RESERVOIR_TEMPS = [450, 500, 550]  # C

DIAMETERS = {
    '9 5/8"': 0.217,     # m ID (Ingason et al., 2014)
    '13 3/8"': 0.315,    # m ID
}

PRESSURE_RANGE = np.linspace(15, 45, 31)

WELL_PARAMS_TEMPLATE = {
    'delta_z_m': 10,
}

CACHE_FILE = 'velocity_diameter_cache.pkl'

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

def run_analysis(pressure_range, diameter_m, reservoir_temp):
    """Compute mass flow and wellhead velocity vs reservoir pressure."""

    area = np.pi * (diameter_m / 2) ** 2

    results = {
        'pressure_MPa': [],
        'mass_flow_kg_s': [],
        'velocity_m_s': [],
        'surface_enthalpy_MJ_kg': [],
        'surface_temp_C': [],
    }

    previous_solution = None

    for p_res in pressure_range:
        depth = int(depth_for_pressure(p_res))

        well_params = dict(WELL_PARAMS_TEMPLATE)
        well_params['depth_m'] = depth
        well_params['diameter_m'] = diameter_m

        reservoir_params = {
            'transmissivity_md_m': FIXED_KXB,
            'drainage_radius_m': 500.0,
        }

        rock_temps = rock_temperature_linear(
            well_params, reservoir_temp, T_SURFACE)

        print(f"  D={diameter_m:.3f}, T={reservoir_temp}, "
              f"P={p_res:.0f} MPa: ", end='')

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                cm = solve_flow_for_whp(
                    target_whp_MPa=FIXED_WHP,
                    P_reservoir_MPa=p_res,
                    T_reservoir_C=reservoir_temp,
                    rock_temperatures=rock_temps,
                    reservoir_params=reservoir_params,
                    well_params=well_params,
                    previous_solution=previous_solution,
                    verbose=False,
                )

            if not cm['success']:
                print("solver failed")
                continue

            if not cm.get('converged', False):
                # Accept wellbore-limited solutions too
                if np.isnan(cm.get('whp_MPa', np.nan)):
                    print("no valid solution")
                    continue

            flow = cm['mass_flow_kgs']
            if np.isnan(flow) or flow <= 0:
                print("no flow")
                continue

            # Compute wellhead density and velocity from (P, h)
            P_whp_Pa = cm['whp_MPa'] * 1e6
            h_J_kg = cm['h_surface_MJkg'] * 1e6
            rho_wh = CP.PropsSI('D', 'P', P_whp_Pa, 'H', h_J_kg,
                                'Water')
            velocity = flow / (rho_wh * area)

            results['pressure_MPa'].append(p_res)
            results['mass_flow_kg_s'].append(flow)
            results['velocity_m_s'].append(velocity)
            results['surface_enthalpy_MJ_kg'].append(cm['h_surface_MJkg'])
            results['surface_temp_C'].append(cm['T_surface_C'])

            previous_solution = {'whp': cm['whp_MPa'], 'flow': flow}

            print(f"q={flow:.1f} kg/s, v={velocity:.1f} m/s, "
                  f"rho={rho_wh:.1f} kg/m3")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    for key in results:
        results[key] = np.array(results[key])

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(all_results,
                 save_path='figure12_velocity_diameter.png'):
    """Two-panel figure: (a) 9 5/8" casing, (b) 13 3/8" casing.
    Each panel: mass flow (dashed, left y) and velocity (solid+markers,
    right y)."""

    colors = {450: '#000000', 500: '#E41A1C', 550: '#377EB8'}
    markers = {450: 'o', 500: 's', 550: '^'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (casing_label, diameter_m) in zip(
            [ax1, ax2], DIAMETERS.items()):

        ax_vel = ax.twinx()

        for T_res in RESERVOIR_TEMPS:
            key = (diameter_m, T_res)
            if key not in all_results:
                continue
            res = all_results[key]
            if len(res['pressure_MPa']) == 0:
                continue

            c = colors[T_res]
            p = res['pressure_MPa']

            # Mass flow - dashed lines
            ax.plot(p, res['mass_flow_kg_s'], '--', color=c,
                    marker=markers[T_res], linewidth=1.5, markersize=4,
                    markevery=3)

            # Velocity - solid lines with markers
            ax_vel.plot(p, res['velocity_m_s'], '-o', color=c,
                        marker=markers[T_res], linewidth=1.5, markersize=5,
                        label=f'{T_res} \u00B0C')

        # Erosional velocity threshold
        ax_vel.axhline(V_CRIT, color='gray', linestyle=':', linewidth=2)
        ax_vel.text(16, V_CRIT + 1, '50 m/s', fontsize=14,
                    va='bottom', color='gray')

        # Formatting
        ax.set_xlabel('Reservoir pressure (MPa)', fontsize=18)
        ax.set_xlim(14, 46)
        ax.set_ylim(bottom=0)
        ax_vel.set_ylim(bottom=0)
        ax.tick_params(axis='both', labelsize=16)
        ax_vel.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)

    # Y-axis labels
    ax1.set_ylabel('Mass flow rate (kg/s)', fontsize=18)
    ax2.set_ylabel('Mass flow rate (kg/s)', fontsize=18)
    fig.axes[2].set_ylabel('Wellhead velocity (m/s)', fontsize=18)
    fig.axes[3].set_ylabel('Wellhead velocity (m/s)', fontsize=18)

    # Panel labels with casing info
    ax1.text(0.03, 0.97,
             f'a) 9 5/8" casing, ID = 0.217 m\n'
             f'    k x b = {int(FIXED_KXB)} md\u00b7m, '
             f'WHP = {FIXED_WHP:.0f} MPa',
             transform=ax1.transAxes, fontsize=14, va='top',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                       edgecolor='none', alpha=0.8))
    ax2.text(0.03, 0.97,
             f'b) 13 3/8" casing, ID = 0.315 m\n'
             f'    k x b = {int(FIXED_KXB)} md\u00b7m, '
             f'WHP = {FIXED_WHP:.0f} MPa',
             transform=ax2.transAxes, fontsize=14, va='top',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                       edgecolor='none', alpha=0.8))

    # Legend on each panel
    dummy_mass = Line2D([0], [0], color='gray', linewidth=1.5,
                    linestyle='--', marker='o', markersize=4,
                    label='Mass flow')
    dummy_vel = Line2D([0], [0], color='gray', linewidth=2,
                       linestyle='-', marker='o', markersize=5,
                       label='Velocity')

    for i, ax_panel in enumerate([ax1, ax2]):
        ax_twin = fig.axes[2 + i]
        lines, labels = ax_twin.get_legend_handles_labels()
        ax_panel.legend(
            handles=lines + [dummy_mass, dummy_vel],
            fontsize=12, loc='lower right', frameon=True,
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

    runs_needed = []
    for diam_label, diam_m in DIAMETERS.items():
        for T_res in RESERVOIR_TEMPS:
            key = (diam_m, T_res)
            if key not in all_results:
                runs_needed.append((diam_label, diam_m, T_res))

    if runs_needed:
        print(f"Running {len(runs_needed)} analyses...")
        for diam_label, diam_m, T_res in runs_needed:
            print(f"\n{'=' * 60}")
            print(f"{diam_label} (D={diam_m} m), T_res = {T_res} C, "
                  f"kxb = {FIXED_KXB} md*m")
            print('=' * 60)
            results = run_analysis(PRESSURE_RANGE, diam_m, T_res)
            all_results[(diam_m, T_res)] = results

        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nCached results to {CACHE_FILE}")
    else:
        print("All results cached.")

    fig = plot_results(all_results)
    plt.show()

    print(f"\nTo rerun, delete: {CACHE_FILE}")