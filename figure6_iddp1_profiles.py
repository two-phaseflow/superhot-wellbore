# -*- coding: utf-8 -*-
"""
IDDP-1 Downhole Profiles: Figure 6 from Scott (2026)
=====================================================

Simulates downhole pressure, temperature, and specific enthalpy
profiles for two flow regimes at IDDP-1:
    - Low flow:  ~7 kg/s  (WHP ~ 14 MPa)
    - High flow: WHP = 4 MPa (flow determined by solver)

Usage:
    python figure6_iddp1_profiles.py
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from reservoir import (
    coupled_model,
    solve_flow_for_whp,
    rock_temperature_boiling,
    DEFAULT_WELL_PARAMS,
)

# ====================================================================
# IDDP-1 WELL AND RESERVOIR PARAMETERS
# ====================================================================
WELL_PARAMS = dict(DEFAULT_WELL_PARAMS)
WELL_PARAMS['depth_m'] = 2100
WELL_PARAMS['diameter_m'] = 0.217

# Reservoir: calibrated values from calibrate_iddp1.py
T_RESERVOIR_C = 500.0
P_RESERVOIR_MPa = 16.11
RESERVOIR_PARAMS = {'transmissivity_md_m': 5318.0}

# Rock temperature: boiling-point-for-depth (Krafla)
ROCK_TEMPS = rock_temperature_boiling(WELL_PARAMS)


# ====================================================================
# SIMULATE TWO FLOW REGIMES
# ====================================================================

# --- Low flow: 7 kg/s ---
print('Low-flow case: 7 kg/s ...')
result_low = coupled_model(
    7.0, P_RESERVOIR_MPa, T_RESERVOIR_C,
    ROCK_TEMPS, RESERVOIR_PARAMS, WELL_PARAMS)

if result_low['success']:
    print(f'  WHP = {result_low["whp_MPa"]:.2f} MPa, '
          f'T_surface = {result_low["T_surface_C"]:.1f} C, '
          f'P_bh = {result_low["P_bh_MPa"]:.2f} MPa')
else:
    raise RuntimeError('Low-flow simulation failed')

# --- High flow: target WHP = 4 MPa ---
print('\nHigh-flow case: solving for WHP = 4 MPa ...')
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    sol = solve_flow_for_whp(
        4.0, P_RESERVOIR_MPa, T_RESERVOIR_C,
        ROCK_TEMPS, RESERVOIR_PARAMS, WELL_PARAMS,
        verbose=True)

if sol['converged']:
    flow_high = sol['flow_rate_kg_s']
    print(f'  Flow = {flow_high:.1f} kg/s, WHP = {sol["whp_MPa"]:.2f} MPa')
else:
    flow_high = 48.0
    print(f'  Solver did not converge, using {flow_high} kg/s')

# Run coupled_model at the solved flow rate to get profiles
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    result_high = coupled_model(
        flow_high, P_RESERVOIR_MPa, T_RESERVOIR_C,
        ROCK_TEMPS, RESERVOIR_PARAMS, WELL_PARAMS)

if result_high['success']:
    print(f'  WHP = {result_high["whp_MPa"]:.2f} MPa, '
          f'T_surface = {result_high["T_surface_C"]:.1f} C, '
          f'P_bh = {result_high["P_bh_MPa"]:.2f} MPa')
else:
    raise RuntimeError('High-flow simulation failed')


# ====================================================================
# EXTRACT PROFILES
# ====================================================================
# profiles tuple: (temp, press, enthalpy, velocity, saturation, density, choked)
# Each profile is a list of (depth_m, value) tuples.

prof_low = result_low['profiles']
prof_high = result_high['profiles']

# Low flow
d_low = np.array([p[0] for p in prof_low[0]]) / 1000.0   # m -> km
T_low = np.array([p[1] for p in prof_low[0]])              # C
P_low = np.array([p[1] for p in prof_low[1]])              # MPa
h_low = np.array([p[1] for p in prof_low[2]])              # MJ/kg

# High flow
d_high = np.array([p[0] for p in prof_high[0]]) / 1000.0
T_high = np.array([p[1] for p in prof_high[0]])
P_high = np.array([p[1] for p in prof_high[1]])
h_high = np.array([p[1] for p in prof_high[2]])

# Reservoir enthalpy for reference line
import CoolProp.CoolProp as CP
h_res_MJkg = CP.PropsSI('H', 'T', T_RESERVOIR_C + 273.15,
                          'P', P_RESERVOIR_MPa * 1e6, 'Water') / 1e6

print(f'\nReservoir reference: P = {P_RESERVOIR_MPa} MPa, '
      f'T = {T_RESERVOIR_C} C, h = {h_res_MJkg:.3f} MJ/kg')


# ====================================================================
# FIGURE 6
# ====================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
})

# Flow label for legend
flow_high_str = (f'{flow_high:.0f}'
                 if abs(flow_high - round(flow_high)) < 0.1
                 else f'{flow_high:.1f}')

fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])   # (a) Pressure
ax2 = fig.add_subplot(gs[0, 1])   # (b) Temperature
ax3 = fig.add_subplot(gs[1, 0])   # (c) Enthalpy

# Center bottom panel between the two top panels
fig.canvas.draw()
mid = (ax1.get_position().x0 + ax2.get_position().x1) / 2
pos = ax3.get_position()
ax3.set_position([mid - pos.width / 2, pos.y0, pos.width, pos.height])


# ------------------------------------------------------------------
# (a) Pressure
# ------------------------------------------------------------------
ax1.plot(P_low, d_low, '-', color='#377EB8', linewidth=2.5,
         label=f'WHP = {result_low["whp_MPa"]:.0f} MPa, 7 kg/s')
ax1.plot(P_high, d_high, '--', color='#377EB8', linewidth=2.5,
         label=f'WHP = 4 MPa, {flow_high_str} kg/s')
ax1.axvline(P_RESERVOIR_MPa, color='k', linestyle='--',
            linewidth=1.5, alpha=0.5)
ax1.text(P_RESERVOIR_MPa - 0.2, 1.1, f'P$_{{res}}$ = {P_RESERVOIR_MPa:.1f} MPa',
         rotation=90, va='center', ha='right', fontsize=12,
         color='0.3', fontstyle='italic')

ax1.set_xlabel('Pressure (MPa)', fontsize=18)
ax1.set_ylabel('Depth (km)', fontsize=18)
ax1.set_ylim(2.2, 0)
ax1.set_xlim(left=3)
ax1.tick_params(axis='both', labelsize=16)
ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax1.legend(fontsize=14, loc='upper center', frameon=True,
           fancybox=False, edgecolor='gray', labelspacing=0.4)
ax1.text(0.04, 0.96, 'a)', transform=ax1.transAxes,
         fontsize=18, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                   edgecolor='none', alpha=0.8))


# ------------------------------------------------------------------
# (b) Temperature
# ------------------------------------------------------------------
ax2.plot(T_low, d_low, '-', color='#377EB8', linewidth=2.5,
         label=f'7 kg/s')
ax2.plot(T_high, d_high, '--', color='#377EB8', linewidth=2.5,
         label=f'{flow_high_str} kg/s')
ax2.axvline(T_RESERVOIR_C, color='k', linestyle='--',
            linewidth=1.5, alpha=0.5)
ax2.text(T_RESERVOIR_C - 1.5, 1.1, f'T$_{{res}}$ = {T_RESERVOIR_C:.0f} \u00B0C',
         rotation=90, va='center', ha='right', fontsize=12,
         color='0.3', fontstyle='italic')
ax2.set_xlabel('Temperature (\u00B0C)', fontsize=18)
ax2.set_ylabel('Depth (km)', fontsize=18)
ax2.set_ylim(2.2, 0)
ax2.tick_params(axis='both', labelsize=16)
ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax2.legend(fontsize=14, loc='lower left', frameon=True,
           fancybox=False, edgecolor='gray', labelspacing=0.4)
ax2.text(0.04, 0.96, 'b)', transform=ax2.transAxes,
         fontsize=18, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                   edgecolor='none', alpha=0.8))


# ------------------------------------------------------------------
# (c) Specific enthalpy
# ------------------------------------------------------------------
ax3.plot(h_low, d_low, '-', color='#377EB8', linewidth=2.5,
         label=f'7 kg/s')
ax3.plot(h_high, d_high, '--', color='#377EB8', linewidth=2.5,
         label=f'{flow_high_str} kg/s')
ax3.axvline(h_res_MJkg, color='k', linestyle='--',
            linewidth=1.5, alpha=0.5)
ax3.text(h_res_MJkg - 0.0025, 0.5, f'h$_{{res}}$ = {h_res_MJkg:.1f} MJ/kg',
         rotation=90, va='center', ha='right', fontsize=12,
         color='0.3', fontstyle='italic')
ax3.set_xlabel('Specific enthalpy (MJ/kg)', fontsize=18)
ax3.set_ylabel('Depth (km)', fontsize=18)
ax3.set_ylim(2.2, 0)
ax3.tick_params(axis='both', labelsize=16)
ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax3.legend(fontsize=14, loc='lower left', frameon=True,
           fancybox=False, edgecolor='gray', labelspacing=0.4)
ax3.text(0.04, 0.96, 'c)', transform=ax3.transAxes,
         fontsize=18, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                   edgecolor='none', alpha=0.8))


plt.savefig('figure6_iddp1_profiles.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_iddp1_profiles.pdf', bbox_inches='tight')
print('\nFigure saved.')