# -*- coding: utf-8 -*-
"""
IDDP-1 Calibration and Figure 5: Deliverability Curve
======================================================

Calibrates the coupled reservoir-wellbore model against IDDP-1 discharge
data (Ingason et al., 2014) by optimizing reservoir pressure and
transmissivity. Produces Figure 5 from Scott (2025).

Usage:
    python calibrate_iddp1.py
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize

from reservoir import (
    evaluate_at_flow_rates,
    rock_temperature_boiling,
    DEFAULT_WELL_PARAMS,
)

# ====================================================================
# IDDP-1 FIELD DATA (Ingason et al., 2014)
# ====================================================================
OBSERVED_WHP = np.array([4.1, 6.2, 7.7, 10.1, 13.5, 14.1])   # MPa
OBSERVED_FLOW = np.array([48, 41, 44, 30, 11, 6.4])            # kg/s

# ====================================================================
# WELL AND RESERVOIR SETUP
# ====================================================================
WELL_PARAMS = dict(DEFAULT_WELL_PARAMS)
WELL_PARAMS['depth_m'] = 2100
WELL_PARAMS['diameter_m'] = 0.217

T_RESERVOIR_C = 500.0

# Rock temperature profile: BPD (Krafla shallow formation)
# Computed once, outside any loop.
ROCK_TEMPS = rock_temperature_boiling(WELL_PARAMS)


# ====================================================================
# FORMATTING
# ====================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.0,
    'figure.dpi': 150,
})


# ====================================================================
# CALIBRATION
# ====================================================================
def calibrate(verbose=True):
    """
    Optimize (P_reservoir, transmissivity) to match IDDP-1 data.
    """
    def objective(params):
        P_res, kxb = params
        if not (10 < P_res < 25 and 500 < kxb < 15000):
            return 1e6
        rp = { 'drainage_radius_m': 500.0,
            'transmissivity_md_m': kxb}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r = evaluate_at_flow_rates(
                OBSERVED_FLOW, P_res, T_RESERVOIR_C,
                ROCK_TEMPS, rp, WELL_PARAMS)
        if not np.all(r['success']):
            return 1e6
        rms = np.sqrt(np.mean((r['whp_MPa'] - OBSERVED_WHP) ** 2))
        if verbose:
            print(f'  P_res={P_res:.2f} MPa, kxb={kxb:.0f} md*m '
                  f'-> RMS={rms:.3f} MPa')
        return rms

    print('IDDP-1 Calibration: optimizing (P_res, kxb)')
    print('=' * 50)

    result = minimize(
        objective,
        x0=[16.11, 5318],
        method='Nelder-Mead',
        options={'maxiter': 100, 'xatol': 5.0, 'fatol': 0.005,
                 'adaptive': True, 'disp': True})

    P_opt, kxb_opt = result.x
    print(f'\nResult: P_res = {P_opt:.2f} MPa, kxb = {kxb_opt:.0f} md*m')
    print(f'RMS error = {result.fun:.3f} MPa')

    return {'P_res_MPa': P_opt, 'kxb_md_m': kxb_opt,
            'rms_MPa': result.fun}


# ====================================================================
# FIGURE 5
# ====================================================================
def plot_figure5(cal):
    """
    Reproduce Figure 5 from Scott (2025).
    WHP (x) vs mass flow rate (left y, solid) and WHT (right y, dashed).
    Red circles: IDDP-1 data. Black lines: calibrated model.
    """
    # --- Compute smooth deliverability curve ---
    rp = {'transmissivity_md_m': cal['kxb_md_m']}
    flows = np.linspace(2.0, 55.0, 50)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        curve = evaluate_at_flow_rates(
            flows, cal['P_res_MPa'], T_RESERVOIR_C,
            ROCK_TEMPS, rp, WELL_PARAMS)
    mask = curve['success'] & (curve['whp_MPa'] > 0.5)
    whp = curve['whp_MPa'][mask]
    flow = curve['flow_kg_s'][mask]
    wht = curve['T_surface_C'][mask]

    # Sort by descending flow (= increasing WHP)
    order = np.argsort(-flow)
    whp, flow, wht = whp[order], flow[order], wht[order]

    # Report
    if np.any(mask):
        print(f'Deliverability curve: {np.sum(mask)}/{len(flows)} '
              f'valid points')
        print(f'  WHP range: {whp.min():.1f} - {whp.max():.1f} MPa')
        print(f'  Flow range: {flow.min():.1f} - {flow.max():.1f} kg/s')

    # --- Figure ---
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Left axis: mass flow rate
    ax1.plot(whp, flow, 'k-', linewidth=2.0, label='Model')
    ax1.plot(OBSERVED_WHP, OBSERVED_FLOW, 'ro', markersize=8,
             markerfacecolor='red', markeredgecolor='black',
             markeredgewidth=0.5,
             label='IDDP-1 (Ingason et al., 2014)')

    ax1.set_xlabel('Wellhead pressure (MPa)')
    ax1.set_ylabel('Mass flow rate (kg/s)')
    ax1.set_xlim(2, 16)
    ax1.set_ylim(0, 60)
    ax1.xaxis.set_major_locator(MultipleLocator(2))

    # Right axis: wellhead temperature
    ax2 = ax1.twinx()
    ax2.plot(whp, wht, 'k--', linewidth=2.0, label='WHT')
    ax2.set_ylabel('Wellhead temperature (\u00B0C)')
    ax2.set_ylim(380, 475)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower center', fontsize=11, frameon=True,
               fancybox=False, edgecolor='gray')

    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

# ====================================================================
# MAIN
# ====================================================================
if __name__ == '__main__':

    # --- Calibrate ---
    cal = calibrate(verbose=True)

    # --- Print comparison table ---
    rp = {'transmissivity_md_m': cal['kxb_md_m']}
    r = evaluate_at_flow_rates(
        OBSERVED_FLOW, cal['P_res_MPa'], T_RESERVOIR_C,
        ROCK_TEMPS, rp, WELL_PARAMS)

    print(f'\nCalibrated model vs IDDP-1 data:')
    print(f'{"Flow":>6s}  {"WHP_obs":>8s}  {"WHP_calc":>8s}  '
          f'{"Error":>6s}  {"T_surf":>6s}  {"BHP":>6s}')
    print('-' * 52)
    for i in range(len(OBSERVED_FLOW)):
        if r['success'][i]:
            print(f'{OBSERVED_FLOW[i]:6.1f}  {OBSERVED_WHP[i]:8.1f}  '
                  f'{r["whp_MPa"][i]:8.2f}  '
                  f'{r["whp_MPa"][i] - OBSERVED_WHP[i]:+6.2f}  '
                  f'{r["T_surface_C"][i]:6.1f}  '
                  f'{r["P_bh_MPa"][i]:6.2f}')
        else:
            print(f'{OBSERVED_FLOW[i]:6.1f}  {OBSERVED_WHP[i]:8.1f}  '
                  f'{"FAIL":>8s}')

    # --- Plot Figure 4 ---
    fig = plot_figure5(cal)
    fig.savefig('figure5_iddp1_calibration.png',
                dpi=300, bbox_inches='tight')
    fig.savefig('figure5_iddp1_calibration.pdf',
                bbox_inches='tight')
    print('\nFigure saved.')