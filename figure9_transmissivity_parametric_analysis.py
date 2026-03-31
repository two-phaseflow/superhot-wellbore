# -*- coding: utf-8 -*-
"""
Transmissivity Parametric Analysis (Figure 9)
===============================================

Analyzes mass flow rate, surface temperature, surface enthalpy,
and power output as a function of reservoir transmissivity at:

    Reservoir temperatures : 450 C (solid), 500 C (dashed)
    Reservoir pressures    : 20, 30, 40 MPa
    Wellhead pressure      : 10 MPa
    Transmissivity range   : 10 - 4000 md.m

Results show 6 curves per panel (3 pressures x 2 temperatures).

Uses the dict-passing API:
    solve_flow_for_whp() -> coupled_model dict -> power_cycle_analysis()
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import CoolProp.CoolProp as CP

from reservoir import (
    solve_flow_for_whp,
    rock_temperature_linear,
    depth_for_pressure,
)
from power_cycle import power_cycle_analysis


# ====================================================================
# FIXED PARAMETERS
# ====================================================================

RESERVOIR_PRESSURES = [20, 30, 40]          # MPa
RESERVOIR_TEMPERATURES = [500]     # C
FIXED_WHP = 10                            # MPa
T_SURFACE = 10                              # C

TRANSMISSIVITY_VALUES = np.array([100, 200, 500, 1000, 2000, 4000])

WELL_PARAMS_TEMPLATE = {
    'diameter_m': 0.217,
    'delta_z_m': 10,
}

CACHE_FILE = 'transmissivity_parametric_cache.pkl'


# ====================================================================
# ANALYSIS
# ====================================================================

def run_transmissivity_parametric(transmissivity_range,
                                  reservoir_pressure, reservoir_temp):
    """
    Run parametric analysis: vary transmissivity at fixed P_res, T_res.

    Parameters
    ----------
    transmissivity_range : array-like
        Transmissivity values to evaluate [md.m].
    reservoir_pressure : float
        Fixed reservoir pressure [MPa].
    reservoir_temp : float
        Fixed reservoir temperature [C].

    Returns
    -------
    dict of arrays.
    """
    depth = int(depth_for_pressure(reservoir_pressure))

    well_params = dict(WELL_PARAMS_TEMPLATE)
    well_params['depth_m'] = depth

    # Rock temperature profile (fixed for this P, T combination)
    rock_temps = rock_temperature_linear(
        well_params, reservoir_temp, T_surface_C=T_SURFACE)

    print('=' * 70)
    print(f'ANALYSIS FOR P_RES = {reservoir_pressure} MPa, '
          f'T_RES = {int(reservoir_temp)} C (depth = {depth} m)')
    print('=' * 70)

    results = {
        'transmissivity_md_m': [],
        'mass_flow_kg_s': [],
        'surface_temp_C': [],
        'surface_enthalpy_MJ_kg': [],
        'power_output_MW': [],
        'choked': [],
        'whp_achieved_MPa': [],
        'cycle_type': [],
    }

    previous_solution = None

    for kxb in transmissivity_range:

        reservoir_params = {
            'transmissivity_md_m': float(kxb),
            'drainage_radius_m': 500.0,
        }

        print(f'kxb={kxb} md.m: ', end='')

        try:
            cm = solve_flow_for_whp(
                target_whp_MPa=FIXED_WHP,
                P_reservoir_MPa=reservoir_pressure,
                T_reservoir_C=reservoir_temp,
                rock_temperatures=rock_temps,
                reservoir_params=reservoir_params,
                well_params=well_params,
                previous_solution=previous_solution,
                min_flow_kg_s=1,
                verbose=True
            )

            if not cm['success']:
                print('x Flow solver failed')
                previous_solution = None
                continue

            # Accept converged solutions AND feasibility-limited
            # solutions where the solver found the max feasible flow
            # but couldn't reach the target WHP.
            if not cm.get('converged', False) and np.isnan(cm.get('whp_MPa', np.nan)):
                print('x No valid solution found')
                previous_solution = None
                continue

            flow = cm['mass_flow_kgs']
            if np.isnan(flow) or flow <= 0:
                print('x Invalid flow rate')
                previous_solution = None
                continue

            is_choked = cm.get('choked', False)

            # Wellbore-limited: WHP exceeds target (choked or friction)
            is_wb_limited = (cm['whp_MPa'] > FIXED_WHP + 0.5)

            pc = power_cycle_analysis(cm)

            if pc['success'] and not np.isnan(pc['power_MWe']):
                power = pc['power_MWe']
                cycle = pc['cycle']
            else:
                power = np.nan
                cycle = 'unknown'
                print('(power calc failed) ', end='')

            # Store results regardless of power success
            results['transmissivity_md_m'].append(kxb)
            results['mass_flow_kg_s'].append(flow)
            results['surface_temp_C'].append(cm['T_surface_C'])
            results['surface_enthalpy_MJ_kg'].append(cm['h_surface_MJkg'])
            results['power_output_MW'].append(power)
            results['choked'].append(is_wb_limited)
            results['whp_achieved_MPa'].append(cm['whp_MPa'])
            results['cycle_type'].append(cycle)

            # Warm start for next kxb step
            previous_solution = {'whp': cm['whp_MPa'], 'flow': flow}

            if is_wb_limited:
                tag = f' [WB-LIMITED, WHP={cm["whp_MPa"]:.1f}]'
            else:
                tag = ''
            print(f'Flow={flow:.1f} kg/s, '
                  f'T={cm["T_surface_C"]:.1f} C, '
                  f'h={cm["h_surface_MJkg"]:.2f} MJ/kg, '
                  f'P={pc["power_MWe"]:.1f} MW, '
                  f'{pc["cycle"]}, '
                  f'WHP={cm["whp_MPa"]:.1f} MPa{tag}')

        except Exception as e:
            print(f'x ERROR: {e}')
            previous_solution = None
            continue

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    print(f'Completed: {len(results["transmissivity_md_m"])} points\n')
    return results


# ====================================================================
# PLOTTING
# ====================================================================

def plot_results(all_results, save_path='figure9_transmissivity_parametric_analysis.png'):
    """
    Create 2x2 plot (Figure 9) with 3 curves per panel (one per pressure).

    Panels a, b, c: filled markers = converged at target WHP,
                     open markers = wellbore-limited (WHP > target).
    Panel d: filled markers = binary cycle, open markers = flash cycle.
    """
    # Saturation properties at WHP
    P_whp_Pa = FIXED_WHP * 1e6
    T_sat_C = CP.PropsSI('T', 'P', P_whp_Pa, 'Q', 0, 'Water') - 273.15
    h_sat_vapor_MJ_kg = CP.PropsSI('H', 'P', P_whp_Pa, 'Q', 1,
                                    'Water') / 1e6

    # Font and style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 11,
    })

    colors = {20: '#000000', 30: '#E41A1C', 40: '#377EB8'}
    linestyles = {20: '-', 30: '--', 40:':'}


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for (pressure, temp), res in sorted(all_results.items()):
        if len(res['transmissivity_md_m']) == 0:
            continue

        kxb = res['transmissivity_md_m']
        color = colors[pressure]
        ls = linestyles[pressure]

        label = f'{pressure} MPa, {int(temp)} \u00b0C'

        # Wellbore-limited mask
        choked = np.asarray(res.get('choked',
                            np.zeros(len(kxb), dtype=bool)), dtype=bool)
        converged = ~choked

        # Binary vs flash mask
        cycle = np.array(res.get('cycle_type',
                         ['binary'] * len(kxb)))
        is_binary = cycle == 'binary'
        is_flash = ~is_binary

        panel_data = [
            res['mass_flow_kg_s'],
            res['surface_temp_C'],
            res['surface_enthalpy_MJ_kg'],
            res['power_output_MW'],
        ]

        for j, ydata in enumerate(panel_data):
            # Continuous line through all points
            axes[j].semilogx(kxb, ydata, linestyle=ls,
                             color=color, linewidth=2.5, label=label)

            if j == 3:
                # Panel d: binary (filled) vs flash (open) only
                if np.any(is_binary):
                    axes[j].semilogx(
                        kxb[is_binary], ydata[is_binary], 's',
                        color=color, markersize=6,
                        markerfacecolor=color, markeredgewidth=1.5)
                if np.any(is_flash):
                    axes[j].semilogx(
                        kxb[is_flash], ydata[is_flash], 's',
                        color=color, markersize=6,
                        markerfacecolor='white',
                        markeredgecolor=color, markeredgewidth=1.5)
            else:
                # Panels a, b, c: converged (filled) vs WB-limited (open)
                if np.any(converged):
                    axes[j].semilogx(
                        kxb[converged], ydata[converged], 's',
                        color=color, markersize=6,
                        markerfacecolor=color, markeredgewidth=1.5)
                if np.any(choked):
                    axes[j].semilogx(
                        kxb[choked], ydata[choked], 's',
                        color=color, markersize=6,
                        markerfacecolor='white',
                        markeredgecolor=color, markeredgewidth=1.5)

    # Legend entries
    from matplotlib.lines import Line2D
    binary_marker = Line2D([0], [0], marker='s', color='black',
                           markerfacecolor='black', markersize=6,
                           linestyle='None', label='Binary cycle')
    flash_marker = Line2D([0], [0], marker='s', color='black',
                          markerfacecolor='white',
                          markeredgecolor='black',
                          markeredgewidth=1.5, markersize=6,
                          linestyle='None', label='Flash cycle')

    has_choked = any(
        np.any(np.asarray(res.get('choked', []), dtype=bool))
        for res in all_results.values())
    if has_choked:
        wb_marker = Line2D([0], [0], marker='s', color='gray',
                           markerfacecolor='white',
                           markeredgecolor='gray', markersize=6,
                           markeredgewidth=1.5, linestyle='None',
                           label='WHP > target')
    else:
        wb_marker = None

    # Saturation line on panel b
    axes[1].axhline(y=T_sat_C, color='gray', linestyle='--',
                    linewidth=1.5, zorder=1)
    axes[1].text(0.55, T_sat_C,
                 f'T$_{{\\mathrm{{sat}}}}$ @ {FIXED_WHP:.0f} MPa',
                 transform=axes[1].get_yaxis_transform(),
                 fontsize=16, va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor='none', alpha=0.9))

    # Saturated vapor enthalpy line on panel c
    axes[2].axhline(y=h_sat_vapor_MJ_kg, color='gray', linestyle='--',
                    linewidth=1.5, zorder=1)
    axes[2].text(0.25, h_sat_vapor_MJ_kg,
                 f'h$_{{\\mathrm{{sat,vapor}}}}$ @ {FIXED_WHP:.0f} MPa',
                 transform=axes[2].get_yaxis_transform(),
                 fontsize=16, va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor='none', alpha=0.9))

    # Axis labels and formatting
    panel_labels = ['a)', 'b)', 'c)', 'd)']
    ylabels = ['Mass flow rate (kg/s)',
               'Wellhead temperature (\u00b0C)',
               'Surface enthalpy (MJ/kg)',
               'Power output (MWe)']

    legend_kwargs = dict(
        fontsize=14, frameon=True, fancybox=False, edgecolor='gray',
        handlelength=3.0, labelspacing=0.4,
    )

    for j, ax in enumerate(axes):
        ax.set_xlabel('Transmissivity (md\u00b7m)', fontsize=18)
        ax.set_ylabel(ylabels[j], fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8,
                which='both')
        ax.text(0.02, 0.98, panel_labels[j],
                transform=ax.transAxes, fontsize=18,
                fontweight='bold', va='top', ha='left')

        if j == 0:
            # Panels a, b, c: main legend upper-left below label,
            # WHP > target legend lower-right
            leg1 = ax.legend(**legend_kwargs, loc='upper left',
                             bbox_to_anchor=(0.0, 0.9))
            if wb_marker is not None:
                ax.add_artist(leg1)
                ax.legend(handles=[wb_marker], fontsize=12,
                          loc='lower right', frameon=True,
                          fancybox=False, edgecolor='gray',
                          bbox_to_anchor=(0.99, 0.08))
        elif j == 1:
            #Panels a, b, c: main legend upper-left below label,
            # WHP > target legend lower-right
            leg1 = ax.legend(**legend_kwargs, loc='upper left',
                             bbox_to_anchor=(0.0, 0.9))
            if wb_marker is not None:
                ax.add_artist(leg1)
                ax.legend(handles=[wb_marker], fontsize=12,
                          loc='lower right', frameon=True,
                          fancybox=False, edgecolor='gray',
                          bbox_to_anchor=(0.99, 0.08))
        elif j == 2:
            #Panels a, b, c: main legend upper-left below label,
            # WHP > target legend lower-right
            leg1 = ax.legend(**legend_kwargs, loc='upper left',
                             bbox_to_anchor=(0.0, 0.9))
            if wb_marker is not None:
                ax.add_artist(leg1)
                ax.legend(handles=[wb_marker], fontsize=12,
                          loc='lower right', frameon=True,
                          fancybox=False, edgecolor='gray',
                          bbox_to_anchor=(0.99, 0.12))
        elif j == 3:
            # Panel d: main legend upper left, binary/flash lower right
            leg1 = ax.legend(**legend_kwargs, loc='upper left',
                             bbox_to_anchor=(0.0, 0.9))
            ax.add_artist(leg1)
            ax.legend(handles=[binary_marker, flash_marker],
                      fontsize=12, loc='lower right', frameon=True,
                      fancybox=False, edgecolor='gray',
                      bbox_to_anchor=(0.99, 0.08))
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {save_path}')

    return fig, axes

# ====================================================================
# MAIN
# ====================================================================

if __name__ == '__main__':

    print(f'\nDual Temperature Analysis Conditions:')
    print(f'  Reservoir Temperatures: {RESERVOIR_TEMPERATURES} C')
    print(f'  Reservoir Pressures: {RESERVOIR_PRESSURES} MPa')
    print(f'  Target WHP: {FIXED_WHP} MPa')
    print(f'  Transmissivity Range: '
          f'{TRANSMISSIVITY_VALUES[0]}-{TRANSMISSIVITY_VALUES[-1]} md.m')
    for P in RESERVOIR_PRESSURES:
        print(f'  {P} MPa -> {int(depth_for_pressure(P))} m depth')
    print()

    # Load or run
    if os.path.exists(CACHE_FILE):
        print(f'Loading cached results from {CACHE_FILE}')
        with open(CACHE_FILE, 'rb') as f:
            all_results = pickle.load(f)
        print(f'Loaded {len(all_results)} cached results')
    else:
        print('No cache found. Will run all analyses.')
        all_results = {}

    combos_to_run = [(P, T) for P in RESERVOIR_PRESSURES
                     for T in RESERVOIR_TEMPERATURES
                     if (P, T) not in all_results]

    if combos_to_run:
        print(f'\nRunning {len(combos_to_run)} analyses...')
        for P_res, T_res in combos_to_run:
            all_results[(P_res, T_res)] = run_transmissivity_parametric(
                TRANSMISSIVITY_VALUES, P_res, T_res)

        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(all_results, f)
        print(f'\nSaved results to {CACHE_FILE}')
    else:
        print('\nAll combinations cached.')

    # Plot
    if all_results:
        fig, axes = plot_results(all_results)

        # Summary
        print('\nSUMMARY')
        print('=' * 70)
        for (pressure, temp), res in sorted(all_results.items()):
            if len(res['transmissivity_md_m']) > 0:
                print(f'\nP_res = {pressure} MPa, T_res = {int(temp)} C:')
                print(f'  Mass flow: {res["mass_flow_kg_s"].min():.1f}'
                      f' - {res["mass_flow_kg_s"].max():.1f} kg/s')
                print(f'  T_surface: {res["surface_temp_C"].min():.1f}'
                      f' - {res["surface_temp_C"].max():.1f} C')
                print(f'  Power:     {res["power_output_MW"].min():.1f}'
                      f' - {res["power_output_MW"].max():.1f} MW')
                n_wb_limited = np.sum(np.asarray(
                    res.get('choked', []), dtype=bool))
                if n_wb_limited > 0:
                    print(f'  WB-limited:    {n_wb_limited} of '
                          f'{len(res["transmissivity_md_m"])} points')
        print('=' * 70)
    else:
        print('\nNo valid results.')

    print(f'\nTo rerun, delete: {CACHE_FILE}')