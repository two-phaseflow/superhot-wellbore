# -*- coding: utf-8 -*-
"""
Temperature Parametric Analysis (Figure 8)
============================================

Analyzes mass flow rate, surface temperature, surface enthalpy,
and power output as a function of reservoir temperature at fixed
conditions:

    Reservoir pressures : 20, 30, 40 MPa
    Wellhead pressure   : 10 MPa
    Transmissivity      : 1000 md.m

Well depth is fixed per pressure curve via depth_for_pressure():
    20 MPa -> 2500 m
    30 MPa -> 3500 m
    40 MPa -> 4500 m

Results are cached to avoid rerunning computationally intensive
simulations. Delete the cache file to force a fresh run.

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

RESERVOIR_PRESSURES = [20,30,40]  # MPa
FIXED_WHP = 10                   # MPa
FIXED_KXB = 1000                   # md.m
T_SURFACE = 10                     # C

WELL_PARAMS_TEMPLATE = {
    'diameter_m': 0.217,
    'delta_z_m': 10,
}

RESERVOIR_PARAMS = {
    'transmissivity_md_m': FIXED_KXB,
    'drainage_radius_m': 500.0,
}

CACHE_FILE = 'temperature_parametric_cache.pkl'


# ====================================================================
# ANALYSIS
# ====================================================================

def run_temperature_parametric(temperature_range, reservoir_pressure):
    """
    Run parametric analysis: vary reservoir temperature at fixed P_res.

    For each temperature, solve_flow_for_whp finds the mass flow rate
    that achieves FIXED_WHP, returning the full coupled_model dict.
    That dict is passed directly to power_cycle_analysis.

    Parameters
    ----------
    temperature_range : array-like
        Reservoir temperatures to evaluate [C].
    reservoir_pressure : float
        Fixed reservoir pressure [MPa].

    Returns
    -------
    dict of arrays: reservoir_temp_C, mass_flow_kg_s, surface_temp_C,
    surface_enthalpy_MJ_kg, power_output_MW, choked, cycle_type,
    whp_achieved_MPa.
    """
    # Depth fixed for this pressure curve
    depth = int(depth_for_pressure(reservoir_pressure))

    well_params = dict(WELL_PARAMS_TEMPLATE)
    well_params['depth_m'] = depth

    print('=' * 70)
    print(f'ANALYSIS FOR P_RES = {reservoir_pressure} MPa '
          f'(depth = {depth} m)')
    print('=' * 70)

    results = {
        'reservoir_temp_C': [],
        'mass_flow_kg_s': [],
        'surface_temp_C': [],
        'surface_enthalpy_MJ_kg': [],
        'power_output_MW': [],
        'choked': [],
        'whp_achieved_MPa': [],
        'cycle_type': [],
    }

    previous_solution = None

    for T_res in temperature_range:

        # Rock temperature profile anchored to this T_res
        rock_temps = rock_temperature_linear(
            well_params, T_res, T_surface_C=T_SURFACE)

        print(f'T_res={T_res:.0f} C: ', end='')

        try:
            cm = solve_flow_for_whp(
                target_whp_MPa=FIXED_WHP,
                P_reservoir_MPa=reservoir_pressure,
                T_reservoir_C=T_res,
                rock_temperatures=rock_temps,
                reservoir_params=RESERVOIR_PARAMS,
                well_params=well_params,
                previous_solution=previous_solution,
                min_flow_kg_s = 10,
                verbose=True
            )

            if not cm['success'] or not cm.get('converged', False):
                print('x Flow solver failed')
                continue

            flow = cm['mass_flow_kgs']
            if np.isnan(flow) or flow <= 0:
                print('x Invalid flow rate')
                continue

            is_choked = cm.get('choked', False)
            is_wb_limited = (cm['whp_MPa'] > FIXED_WHP + 0.5)

            pc = power_cycle_analysis(cm)

            if not pc['success'] or np.isnan(pc['power_MWe']):
                print('x Power calculation failed')
                continue

            # Store results
            results['reservoir_temp_C'].append(T_res)
            results['mass_flow_kg_s'].append(flow)
            results['surface_temp_C'].append(cm['T_surface_C'])
            results['surface_enthalpy_MJ_kg'].append(cm['h_surface_MJkg'])
            results['power_output_MW'].append(pc['power_MWe'])
            results['choked'].append(is_wb_limited)
            results['whp_achieved_MPa'].append(cm['whp_MPa'])
            results['cycle_type'].append(pc['cycle'])

            # Warm start for next temperature step
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
            continue

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    print(f'Completed: {len(results["reservoir_temp_C"])} points\n')
    return results


# ====================================================================
# PLOTTING
# ====================================================================

def plot_results(all_results, save_path='figure8_temperature_parametric.png'):
    """
    Create 2x2 plot (Figure 8) with multiple pressure curves.

    Parameters
    ----------
    all_results : dict
        Keys are reservoir pressures (MPa), values are result dicts.
    save_path : str
        Output filename.
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
    })

    colors = ['#000000', '#E41A1C', '#377EB8']  # black, red, blue
    linestyles = ['-', '--', ':']               # solid, dashed, dotted

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (pressure, res) in enumerate(sorted(all_results.items())):
        if len(res['reservoir_temp_C']) == 0:
            continue

        t = res['reservoir_temp_C']
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        label = f'{pressure} MPa'

        # Wellbore-limited mask
        choked = np.asarray(res.get('choked',
                            np.zeros(len(t), dtype=bool)), dtype=bool)

        # Binary vs flash mask
        cycle = np.array(res.get('cycle_type',
                         ['binary'] * len(t)))
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
            axes[j].plot(t, ydata, '-', color=color, linestyle=ls,
                         linewidth=2.5, label=label)

            if j == 3:
                # Panel d: filled = binary, open = flash
                if np.any(is_binary):
                    axes[j].plot(t[is_binary], ydata[is_binary], 'o',
                                 color=color, markersize=5,
                                 markerfacecolor=color)
                if np.any(is_flash):
                    axes[j].plot(t[is_flash], ydata[is_flash], 'o',
                                 color=color, markersize=5,
                                 markerfacecolor='white',
                                 markeredgecolor=color,
                                 markeredgewidth=1.5)


            # Panel a only: choked overlay (open squares)
            if j == 0 and np.any(choked):
                axes[j].plot(t[choked], ydata[choked], 's',
                             color=color, markersize=8,
                             markerfacecolor='none',
                             markeredgewidth=1.5)

    # Binary/flash legend entries for panel d
    from matplotlib.lines import Line2D
    binary_marker = Line2D([0], [0], marker='o', color='black',
                           markerfacecolor='black', markersize=6,
                           linestyle='None', label='Binary cycle')
    flash_marker = Line2D([0], [0], marker='o', color='black',
                          markerfacecolor='white',
                          markeredgecolor='black',
                          markeredgewidth=1.5, markersize=6,
                          linestyle='None', label='Flash cycle')

    # Wellbore-limited legend entry for panel a
    has_choked = any(
        np.any(np.asarray(res.get('choked', []), dtype=bool))
        for res in all_results.values())
    if has_choked:
        choked_marker = Line2D([0], [0], marker='s', color='gray',
                               markerfacecolor='none', markersize=8,
                               markeredgewidth=1.5, linestyle='None',
                               label='Wellbore-limited')
    else:
        choked_marker = None

    # Saturation line on panel b (temperature)
    axes[1].axhline(y=T_sat_C, color='gray', linestyle='--',
                    linewidth=1.5, zorder=1)
    axes[1].text(0.65, T_sat_C,
                 f'  $\\mathrm{{T_{{sat}}}}$ @ {FIXED_WHP} MPa',
                 transform=axes[1].get_yaxis_transform(),
                 fontsize=16, va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor='none', alpha=0.9))

    # Saturated vapor enthalpy line on panel c
    axes[2].axhline(y=h_sat_vapor_MJ_kg, color='gray', linestyle='--',
                    linewidth=1.5, zorder=1)
    axes[2].text(0.55, h_sat_vapor_MJ_kg,
                 f'  $\\mathrm{{h_{{sat,vapor}}}}$ @ {FIXED_WHP} MPa',
                 transform=axes[2].get_yaxis_transform(),
                 fontsize=16, va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor='none', alpha=0.9))

    # Axis labels and formatting
    panel_labels = ['a)', 'b)', 'c)', 'd)']
    ylabels = ['Mass flow rate (kg/s)',
               'Wellhead temperature (\u00b0C)',
               'Wellhead enthalpy (MJ/kg)',
               'Power output (MWe)']
    legend_locs = ['upper right', 'center left', 'lower right', 'lower right']

    for j, ax in enumerate(axes):
        ax.set_xlabel('Reservoir temperature (\u00b0C)', fontsize=18)
        ax.set_ylabel(ylabels[j], fontsize=18)
        ax.set_xlim(375,600)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
        ax.text(0.02, 0.98, panel_labels[j],
                transform=ax.transAxes, fontsize=18, fontweight='bold',
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='white', alpha=0.9))

        if j == 0:
            # Panel a: main legend + choked entry
            leg1 = ax.legend(fontsize=18, loc=legend_locs[j],
                             frameon=True,
                             title='P$_{\\mathrm{res}}$',
                             title_fontsize=18)
            if choked_marker is not None:
                ax.add_artist(leg1)
                ax.legend(handles=[choked_marker], fontsize=14,
                          loc='upper right', frameon=True)
        elif j == 3:
            # Panel d: main legend + binary/flash legend
            leg1 = ax.legend(fontsize=18, loc=legend_locs[j],
                             frameon=True,
                             title='P$_{\\mathrm{res}}$',
                             title_fontsize=18,
                             bbox_to_anchor=(1.0, 0.05))
            ax.add_artist(leg1)
            ax.legend(handles=[binary_marker, flash_marker],
                      fontsize=14, loc='upper left', frameon=True,
                      bbox_to_anchor=(0.0, 0.88))
        else:
            ax.legend(fontsize=18, loc=legend_locs[j], frameon=True,
                      title='P$_{\\mathrm{res}}$',
                      title_fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {save_path}')

    return fig, axes


# ====================================================================
# MAIN
# ====================================================================

if __name__ == '__main__':

    print(f'\nFixed Analysis Conditions:')
    print(f'  Reservoir Pressures: {RESERVOIR_PRESSURES} MPa')
    print(f'  Target WHP: {FIXED_WHP} MPa')
    print(f'  Transmissivity: {FIXED_KXB} md.m')
    for P in RESERVOIR_PRESSURES:
        print(f'  {P} MPa -> {int(depth_for_pressure(P))} m depth')
    print()

    temperature_range = np.linspace(375, 600, 21)  # 10 C spacing

    # Load or run
    if os.path.exists(CACHE_FILE):
        print(f'Loading cached results from {CACHE_FILE}')
        with open(CACHE_FILE, 'rb') as f:
            all_results = pickle.load(f)

        pressures_to_run = [P for P in RESERVOIR_PRESSURES
                            if P not in all_results
                            or 'power_output_MW' not in all_results[P]
                            or len(all_results[P].get('power_output_MW',
                                                       [])) == 0]
    else:
        print('No cache found. Will run all analyses.')
        all_results = {}
        pressures_to_run = RESERVOIR_PRESSURES

    if pressures_to_run:
        print(f'\nRunning analyses for pressures: {pressures_to_run}')
        for P_res in pressures_to_run:
            all_results[P_res] = run_temperature_parametric(
                temperature_range, P_res)

        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(all_results, f)
        print(f'\nSaved results to {CACHE_FILE}')
    else:
        print('\nAll pressures cached. Using cached results.')

    # Plot
    if all_results:
        fig, axes = plot_results(all_results)

        # Summary
        print('\nSUMMARY')
        print('=' * 70)
        for pressure, res in sorted(all_results.items()):
            if len(res['reservoir_temp_C']) > 0:
                print(f'\nP_res = {pressure} MPa:')
                print(f'  Temperature: {res["reservoir_temp_C"].min():.0f}'
                      f' - {res["reservoir_temp_C"].max():.0f} C')
                print(f'  Mass flow:   {res["mass_flow_kg_s"].min():.1f}'
                      f' - {res["mass_flow_kg_s"].max():.1f} kg/s')
                print(f'  T_surface:   {res["surface_temp_C"].min():.1f}'
                      f' - {res["surface_temp_C"].max():.1f} C')
                print(f'  h_surface:   {res["surface_enthalpy_MJ_kg"].min():.2f}'
                      f' - {res["surface_enthalpy_MJ_kg"].max():.2f} MJ/kg')
                print(f'  Power:       {res["power_output_MW"].min():.1f}'
                      f' - {res["power_output_MW"].max():.1f} MW')
                n_wb_limited = np.sum(np.asarray(
                    res.get('choked', []), dtype=bool))
                if n_wb_limited > 0:
                    print(f'  WB-limited:      {n_wb_limited} of '
                          f'{len(res["reservoir_temp_C"])} points')
        print('=' * 70)
    else:
        print('\nNo valid results.')

    print(f'\nTo rerun, delete: {CACHE_FILE}')