# -*- coding: utf-8 -*-
"""
Reservoir Physics and Coupling Library for Superhot Geothermal Systems
=======================================================================

Couples a steady-state radial Darcy reservoir model with the wellbore
simulator in wellbore_physics.py to predict wellhead conditions for
given reservoir properties.

Implements the reservoir model described in:

    Scott, S.W. (2025). Thermo-hydraulic drivers of superhot
    geothermal well performance. Geothermics.

    Equations (1)-(3) in that paper correspond to darcy_pressure_drop()
    and bottomhole_pressure() in this module. Equations (4)-(5) are
    implemented in wellbore_physics.py.

Public interface
----------------
The two main entry points for downstream code are:

    coupled_model(mass_flow_rate, P_reservoir_MPa, T_reservoir_C,
                  rock_temperatures, reservoir_params, well_params)

        Runs the full reservoir-to-wellhead chain for a prescribed
        mass flow rate: Darcy drawdown -> feedzone state -> wellbore
        simulation -> surface conditions. Returns a dict with WHP,
        surface temperature, surface enthalpy, feedzone conditions,
        profiles, and a choked flow flag. Returns success=False if
        the wellbore simulation fails to reach the surface (e.g.,
        negative pressure from excessive friction at high flow
        rates) or if velocity exceeds the local sound speed at
        any point in the wellbore (choked=True in the return dict).
        The surface-reach check prevents the solver from accepting
        flow rates where the simulation terminated early and the
        reported WHP would correspond to a subsurface depth rather
        than the actual wellhead.

    solve_flow_for_whp(target_whp_MPa, P_reservoir_MPa, T_reservoir_C,
                       rock_temperatures, reservoir_params, well_params,
                       previous_solution, tolerance_MPa, verbose)

        Finds the mass flow rate that produces a target WHP by
        scanning and bisecting over coupled_model(). Returns the
        full coupled_model() output dict with additional 'converged'
        and 'flow_rate_kg_s' keys. Handles three regimes:

        - Normal: converges on a flow rate within tolerance of the
          target WHP.
        - Choke-limited: the wellbore cannot sustain enough flow to
          draw WHP down to the target (v > c_sound). Returns the
          maximum feasible flow with choked=True and the achieved
          WHP (which exceeds the target).
        - Infeasible: target WHP cannot be achieved due to numerical
          limits (e.g., near-critical EOS failures). Returns
          converged=False.

        All internal solver parameters (step sizes, probe points,
        bisection thresholds) scale automatically with the Darcy-
        estimated maximum flow rate. This makes the solver robust
        across a wide range of transmissivities (10-4000+ md.m)
        without manual tuning.

        The output dict can be passed directly to
        power_cycle.power_cycle_analysis() for power estimation.

Additional public functions:

    evaluate_at_flow_rates(flow_rates_kg_s, P_reservoir_MPa,
                           T_reservoir_C, rock_temperatures,
                           reservoir_params, well_params,
                           max_consecutive_failures)

        Batch evaluation of coupled_model() across a vector of
        prescribed flow rates. Returns parallel arrays of WHP,
        surface temperature, surface enthalpy, feedzone conditions,
        reservoir pressure drop, and choked flow flags. Also stores
        the full coupled_model() output dict for each successful
        point (without profiles) in 'coupled_results', enabling
        direct pass-through to power_cycle.power_cycle_analysis().

        Runtime warnings from the wellbore EOS are suppressed during
        evaluation. Stops early after max_consecutive_failures
        (default 5) successive failures to avoid grinding through
        infeasible high-flow-rate evaluations.

    deliverability_curve()         - WHP vs mass flow rate curve
    depth_for_pressure()           - Depth-pressure scaling model
    rock_temperature_linear()      - Linear T(z) profile
    rock_temperature_boiling()     - Boiling-point-with-depth profile
    rock_temperature_from_user()   - User-supplied profile validation

    Building blocks (used internally by coupled_model, also public):

    darcy_pressure_drop()          - Reservoir pressure drop (Eq. 2)
    feedzone_state()               - Isenthalpic feedzone T and h
    bottomhole_pressure()          - Chains Darcy drop + feedzone state

Modeling assumptions
--------------------
The reservoir is modeled as a homogeneous, isotropic porous medium
with steady-state radial Darcy flow from a fixed-pressure drainage
boundary to the wellbore. Key assumptions:

    1. Single-phase reservoir fluid: the fluid is treated as single-
       phase at reservoir conditions (supercritical or superheated
       vapor). This is appropriate for superhot systems where
       T_reservoir >> T_critical.

    2. Constant fluid properties in the reservoir: density and
       viscosity are evaluated at reservoir (P, T) and assumed
       uniform between the drainage radius and the wellbore. This
       is an upstream weighting scheme: properties are evaluated
       at the far-field (drainage boundary) conditions rather than
       at the wellbore face. See "Fluid property evaluation" below
       for discussion of this approximation.

    3. Isenthalpic near-well inflow: fluid flowing from the
       drainage boundary to the feedzone conserves specific
       enthalpy. Because the temperature of supercritical water
       varies with pressure at constant enthalpy, the fluid
       temperature at the feedzone (bottomhole) is lower than the
       reservoir temperature. This expansion cooling is significant
       at high pressure drawdowns (up to ~30 C cooling at dP ~ 30
       MPa for T_res ~ 500 C).

    4. No near-wellbore effects: skin factor, non-Darcy flow,
       and thermal drawdown are not modeled. The model therefore
       represents an idealized upper bound on well productivity.

Transmissivity
--------------
The Darcy drawdown equation depends only on the product k * b
(transmissivity), not on permeability k or thickness b individually.
These two parameters cannot be uniquely separated from flow data
alone. Accordingly, all reservoir functions in this module accept
transmissivity as the fundamental parameter.

The reservoir_params dictionary can specify transmissivity in two
ways (resolved by _resolve_transmissivity()):

    1. Directly:     reservoir_params['transmissivity_md_m'] = 1000
    2. As a product:  reservoir_params['permeability_md'] = 40
                      reservoir_params['thickness_m'] = 25

If both are present, the explicit transmissivity_md_m takes
precedence (with a warning if the values are inconsistent).

Note on productivity index (PI): some authors parameterize reservoir
deliverability using a lumped productivity index rather than explicit
transmissivity and well geometry. The relationship is:

    PI = (2 * pi * k * b) / (mu * ln(r_e / r_w))    [m3/s/Pa]

Given PI and fluid properties, one can back-calculate the equivalent
transmissivity for use in this module. We do not accept PI directly
as an input to avoid ambiguity in the assumed viscosity and geometry.

Fluid property evaluation
-------------------------
The Darcy drawdown uses the kinematic viscosity nu = mu/rho to
convert between volumetric and mass flow rates. In steady-state
radial flow, nu varies continuously from the drainage boundary
(reservoir P, T) to the wellbore face (bottomhole P, T). For
superhot vapors, density drops much faster than viscosity as
pressure decreases, so nu increases significantly toward the
wellbore.

This module uses upstream weighting: fluid properties are
evaluated at the reservoir pressure P_res, with the temperature
iteratively corrected for isenthalpic expansion cooling. This
is the standard approach in analytical geothermal reservoir
models (Grant and Bixley, 2011).

The approximation is exact when the drawdown is small relative
to P_res (i.e., at high transmissivity where dP << P_res). At
large drawdowns, upstream weighting underestimates the pressure
drop because it does not account for the decrease in density
between the drainage boundary and the wellbore. Quantitatively,
for conditions typical of IDDP-1 (P_res ~ 16 MPa, T_res ~ 500 C):

    dP ~ 4 MPa (30 kg/s):  underestimate ~10-15%
    dP ~ 8 MPa (48 kg/s):  underestimate ~30-40%

Alternative approaches (mean-pressure evaluation, numerical
integration of the radial Darcy equation with P-dependent
properties) were tested but found to increase rather than
decrease calibrated transmissivity values, because the
wellbore model compensates for higher reservoir drawdown by
requiring more kxb. Since the analytical Darcy model is
already highly idealized (homogeneous reservoir, no skin, no
near-wellbore non-Darcy effects), and since upstream weighting
produces calibrated transmissivities more consistent with
independent estimates, the simpler approach is retained.

The standalone darcy_pressure_drop() function evaluates
properties at a single (P, T) point. The iterative temperature
correction in bottomhole_pressure() captures the leading-order
effect of expansion cooling on viscosity.

Thermodynamic interface
-----------------------
Reservoir boundary conditions are specified in (P, T) coordinates,
the natural variables for reservoir characterization. This module
converts to the P-h formulation used by the wellbore simulator:

    Scripts provide:    P_reservoir [MPa], T_reservoir [C]
    reservoir.py:       (P, T) -> (rho, mu, h) via CoolProp T-P lookup
                        Darcy drawdown -> P_bh
                        h_feedzone = h_reservoir  (isenthalpic)
    wellbore_simulate:  takes (P_bh, h_feedzone, mdot)

The feedzone temperature T_feedzone is a derived output quantity
obtained from fluid_properties_Ph(P_bh, h_feedzone). It is reported
for diagnostics but is not an input to the wellbore model.

Wellbore radius
---------------
The wellbore radius r_w used in the Darcy equation is derived from
the wellbore diameter in well_params (r_w = diameter_m / 2). This
ensures consistency between the reservoir and wellbore models: there
is a single diameter parameter, and r_w is never set independently.

Rock temperature profiles
-------------------------
The wellbore simulator requires a rock temperature profile T_rock(z)
that defines the conductive heat loss boundary condition at each
depth step. This profile must be discretized at exactly the depths
that wellbore_simulate() marches through (determined by well_params
'depth_m' and 'delta_z_m'). To enforce this, all rock temperature
functions in this module take well_params as input and derive the
depth grid internally via _get_well_depths().

Three options are provided:

    rock_temperature_linear(well_params, T_reservoir_C):
        Linear gradient from T_surface to T_reservoir. Used for the
        parametric analyses in Scott (2025), where each (P, T, depth)
        combination requires a profile anchored to T_reservoir.

    rock_temperature_boiling(well_params):
        Boiling-point-with-depth curve computed from hydrostatic
        pressure with temperature-dependent density. Used for the
        IDDP-1 validation, where the shallow formation at Krafla
        is liquid-dominated and follows the BPD curve.

    rock_temperature_from_user(well_params, user_profile):
        Accepts a dict {depth: T_C} or callable f(z) -> T_C, and
        interpolates/evaluates to the correct grid. Use this for
        site-specific profiles from measured temperature logs.

The rock temperature profile is independent of the reservoir
conditions (P_res, T_res) and should be computed once before any
parameter sweep or optimization loop. It is passed to coupled_model()
and solve_flow_for_whp() as an argument, not recomputed internally.

Future extensions
-----------------
    - Saline fluids: replace CoolProp with Driesner EOS correlations
      for NaCl-H2O. The coupling architecture is unchanged.
    - Non-Darcy flow: add Forchheimer correction for high-velocity
      near-wellbore flow (relevant for very high transmissivity).
    - Pressure-dependent properties in the reservoir: integrate the
      radial Darcy equation numerically rather than using a single
      evaluation point for fluid properties.

References
----------
    - Scott, S.W. (2025). Thermo-hydraulic drivers of superhot
      geothermal well performance. Geothermics.
    - Grant, M.A., Donaldson, I.G., Bixley, P.F. (1982). Geothermal
      Reservoir Engineering. Academic Press, New York.
    - Grant, M.A. and Bixley, P.F. (2011). Geothermal Reservoir
      Engineering, 2nd edition. Academic Press.
    - Bell, I.H. et al. (2014). CoolProp. Ind. Eng. Chem. Res.
      53(6), 2498-2508.
"""

import warnings

import numpy as np
import CoolProp.CoolProp as CP

from wellbore_physics import (
    wellbore_simulate,
    fluid_properties_Ph,
    DEFAULT_WELL_PARAMS,
)


# ===================================================================
# CONSTANTS
# ===================================================================

GRAVITY = 9.80665         # m/s2
MD_TO_M2 = 9.869233e-16  # 1 millidarcy in m2


# ===================================================================
# DEFAULT PARAMETERS
# ===================================================================

# Default reservoir parameters. Scripts can override any key by
# passing a reservoir_params dict; unspecified keys fall back to
# these values.
#
# The default transmissivity of 1000 md*m is a round number in the
# middle of the range explored in the parametric analyses (10-4000
# md*m). The drainage radius of 500 m is a conventional assumption
# for a single-well reservoir; the sensitivity of results to this
# parameter is weak (logarithmic dependence).
DEFAULT_RESERVOIR_PARAMS = {
    'drainage_radius_m': 500.0,
    'transmissivity_md_m': 1000.0,
}


# ===================================================================
# 1. INTERNAL UTILITIES
# ===================================================================

def _merge_reservoir_params(user_params):
    """
    Merge user-provided reservoir_params with DEFAULT_RESERVOIR_PARAMS.

    Returns a complete dict with all keys populated. User values
    override defaults.

    Special handling: if the user provides permeability_md and
    thickness_m but NOT transmissivity_md_m, the default
    transmissivity_md_m is removed so that the user's (k, b) pair
    is used via _resolve_transmissivity().
    """
    params = dict(DEFAULT_RESERVOIR_PARAMS)
    if user_params is not None:
        user_has_kb = ('permeability_md' in user_params
                       and 'thickness_m' in user_params)
        user_has_T = 'transmissivity_md_m' in user_params
        if user_has_kb and not user_has_T:
            params.pop('transmissivity_md_m', None)
        params.update(user_params)
    return params


def _resolve_transmissivity(reservoir_params):
    """
    Extract transmissivity [md*m] from a reservoir_params dict.

    Accepts two specification modes:
        1. reservoir_params['transmissivity_md_m']  -- used directly
        2. reservoir_params['permeability_md'] * reservoir_params['thickness_m']

    If transmissivity_md_m is present, it takes precedence. If both
    forms are present and differ by more than 1%, a warning is issued.

    Returns
    -------
    float - Transmissivity in md*m.

    Raises
    ------
    ValueError if neither specification is present.
    """
    has_T = 'transmissivity_md_m' in reservoir_params
    has_kb = ('permeability_md' in reservoir_params
              and 'thickness_m' in reservoir_params)

    if has_T:
        T_direct = reservoir_params['transmissivity_md_m']
        if has_kb:
            T_product = (reservoir_params['permeability_md']
                         * reservoir_params['thickness_m'])
            rel_diff = abs(T_direct - T_product) / max(T_direct, 1e-30)
            if rel_diff > 0.01:
                warnings.warn(
                    f"transmissivity_md_m ({T_direct:.1f}) differs from "
                    f"permeability_md * thickness_m ({T_product:.1f}) by "
                    f"{rel_diff*100:.1f}%. Using transmissivity_md_m.",
                    RuntimeWarning, stacklevel=3)
        return T_direct

    if has_kb:
        return (reservoir_params['permeability_md']
                * reservoir_params['thickness_m'])

    raise ValueError(
        "reservoir_params must contain either 'transmissivity_md_m' "
        "or both 'permeability_md' and 'thickness_m'. Got keys: "
        f"{list(reservoir_params.keys())}")


def _get_wellbore_radius(well_params):
    """
    Derive wellbore radius from the well diameter parameter.

    This ensures r_w = D/2 is always consistent with the diameter
    used in the wellbore simulator. There is no separate wellbore
    radius parameter to set.

    Parameters
    ----------
    well_params : dict or None
        Must contain 'diameter_m', or None to use DEFAULT_WELL_PARAMS.

    Returns
    -------
    float - Wellbore radius [m].
    """
    if well_params is not None and 'diameter_m' in well_params:
        return well_params['diameter_m'] / 2.0
    return DEFAULT_WELL_PARAMS['diameter_m'] / 2.0


def _get_well_depths(well_params):
    """
    Return the depth array that wellbore_simulate() will march through.

    The wellbore simulator accesses rock_temperatures[depth] at each
    marching step. This function returns the exact set of depth keys
    required, derived from well_params['depth_m'] and
    well_params['delta_z_m']. All rock temperature profiles must be
    discretized at these depths.

    Parameters
    ----------
    well_params : dict or None
        Must contain 'depth_m' and 'delta_z_m', or None to use
        DEFAULT_WELL_PARAMS.

    Returns
    -------
    list of int - Depth values [m], from 0 to well_depth inclusive.
    """
    params = dict(DEFAULT_WELL_PARAMS)
    if well_params is not None:
        params.update(well_params)
    depth_m = int(params['depth_m'])
    delta_z = int(params['delta_z_m'])
    return list(range(0, depth_m + delta_z, delta_z))


# ===================================================================
# 2. RESERVOIR FLOW
# ===================================================================
#
# The pressure drawdown for steady-state, radial, single-phase Darcy
# flow in a confined reservoir is derived by integrating Darcy's law
# in cylindrical coordinates.
#
# Starting from the radial form of Darcy's law for mass flux:
#
#     q_m = -k * rho / mu * dP/dr              (mass flux, kg/m2/s)
#
# where k is permeability [m2], rho is fluid density [kg/m3], mu is
# dynamic viscosity [Pa*s], and r is radial distance from the well
# axis [m].
#
# For steady-state flow, mass conservation requires that the total
# mass flow rate through any cylindrical surface of radius r and
# height b (reservoir thickness) is constant:
#
#     mdot = 2 * pi * r * b * q_m = -2 * pi * r * b * k * rho / mu * dP/dr
#
# Rearranging and integrating from the wellbore radius r_w to the
# drainage radius r_e:
#
#     integral(r_w to r_e) dP = -mdot * mu / (2 * pi * k * b * rho) *
#                               integral(r_w to r_e) dr/r
#
#     P(r_e) - P(r_w) = mdot * mu * ln(r_e / r_w) / (2 * pi * rho * k * b)
#
# Therefore the pressure drawdown (P_reservoir - P_bottomhole) is:
#
#     dP_res = mdot * mu * ln(r_e / r_w) / (2 * pi * rho * k * b)
#
# This is Eq. (2) in Scott (2025), following the standard treatment
# in Grant et al. (1982, Ch. 3) and Grant and Bixley (2011, Ch. 3).
#
# Note that k*b = transmissivity [m2*m = m3] appears as a single
# parameter; see the module docstring for discussion.

def darcy_pressure_drop(mass_flow_rate, P_reservoir_MPa, T_reservoir_C,
                        reservoir_params=None, well_params=None):
    """
    Steady-state radial Darcy pressure drawdown in the reservoir.

    Evaluates fluid density and viscosity at reservoir (P, T) using
    CoolProp (upstream weighting; see module docstring for discussion).

    Parameters
    ----------
    mass_flow_rate : float
        Mass flow rate into the well [kg/s].
    P_reservoir_MPa : float
        Reservoir pressure at drainage boundary [MPa].
    T_reservoir_C : float
        Reservoir temperature [C].
    reservoir_params : dict or None
        Must contain transmissivity (see _resolve_transmissivity()).
        If None, DEFAULT_RESERVOIR_PARAMS is used.
    well_params : dict or None
        Well parameters. The wellbore radius r_w is derived as
        diameter_m / 2. If None, DEFAULT_WELL_PARAMS is used.

    Returns
    -------
    float
        Pressure drawdown dP_res [MPa]. Always >= 0.
    """
    params = _merge_reservoir_params(reservoir_params)
    transmissivity_md_m = _resolve_transmissivity(params)

    r_w = _get_wellbore_radius(well_params)
    r_e = params['drainage_radius_m']

    # Fluid properties at reservoir conditions.
    # At superhot reservoir conditions (T >> T_crit, single-phase),
    # CoolProp's T-P solver is unambiguous. We call it directly
    # rather than routing through the P-h EOS in wellbore_physics.py.
    T_K = T_reservoir_C + 273.15
    P_Pa = P_reservoir_MPa * 1e6
    rho = CP.PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water')
    mu = CP.PropsSI('V', 'T', T_K, 'P', P_Pa, 'Water')

    # Transmissivity: md*m -> m2*m = m3
    kb_m3 = transmissivity_md_m * MD_TO_M2

    # dP = mu * mdot * ln(r_e/r_w) / (2*pi * rho * k*b)
    ln_ratio = np.log(r_e / r_w)
    dP_Pa = (mu * mass_flow_rate * ln_ratio) / (2.0 * np.pi * rho * kb_m3)

    return dP_Pa / 1e6  # Pa -> MPa


# ===================================================================
# 3. FEEDZONE PHYSICS
# ===================================================================

def feedzone_state(P_reservoir_MPa, T_reservoir_C, P_bh_MPa):
    """
    Compute feedzone thermodynamic state assuming isenthalpic expansion.

    Fluid flows from the reservoir at (P_res, T_res) to the bottomhole
    at P_bh, conserving specific enthalpy. The feedzone temperature is
    determined by the P-h equation of state at (P_bh, h_res).

    This expansion cooling is a key physical effect in superhot systems:
    for example, at P_res=40 MPa, T_res=500 C, a 10 MPa drawdown
    produces ~30 C of cooling.

    Parameters
    ----------
    P_reservoir_MPa : float
        Reservoir pressure [MPa].
    T_reservoir_C : float
        Reservoir temperature [C].
    P_bh_MPa : float
        Bottomhole flowing pressure [MPa].

    Returns
    -------
    dict with keys:
        'h_reservoir_Jkg'  : float - Reservoir/feedzone enthalpy [J/kg]
        'T_feedzone_C'     : float - Feedzone temperature [C]
        'T_cooling_C'      : float - Expansion cooling T_res - T_fz [C]
        'rho_feedzone_kgm3': float - Feedzone density [kg/m3]
    """
    T_res_K = T_reservoir_C + 273.15
    P_res_Pa = P_reservoir_MPa * 1e6

    # Reservoir enthalpy from CoolProp T-P lookup (single-phase,
    # unambiguous at superhot conditions)
    h_res = CP.PropsSI('H', 'T', T_res_K, 'P', P_res_Pa, 'Water')

    # Feedzone state from P-h lookup (isenthalpic: h_fz = h_res).
    # This uses the P-h EOS from wellbore_physics.py, which handles
    # near-critical and two-phase routing robustly.
    fz = fluid_properties_Ph(P_bh_MPa, h_res)
    T_fz_C = fz['temperature_K'] - 273.15

    return {
        'h_reservoir_Jkg': h_res,
        'T_feedzone_C': T_fz_C,
        'T_cooling_C': T_reservoir_C - T_fz_C,
        'rho_feedzone_kgm3': fz['density_kgm3'],
    }


def bottomhole_pressure(mass_flow_rate, P_reservoir_MPa, T_reservoir_C,
                        reservoir_params=None, well_params=None,
                        max_iterations=3):
    """
    Compute bottomhole flowing pressure with iterative expansion cooling.

    The Darcy pressure drop depends on fluid properties (density and
    viscosity), which change between the drainage boundary and the
    wellbore due to isenthalpic expansion cooling. This function
    iterates between the Darcy equation and the feedzone temperature
    calculation until convergence.

    Fluid property evaluation
    -------------------------
    Properties are evaluated at the reservoir pressure P_res, with the
    temperature updated iteratively to account for expansion cooling
    (upstream weighting). This is the standard approach in geothermal
    reservoir engineering (Grant and Bixley, 2011) and is a reasonable
    approximation when the drawdown is small relative to P_res.

    At large drawdowns, density decreases significantly toward the
    wellbore, which increases the effective kinematic viscosity
    nu = mu/rho and therefore the actual pressure drop. Upstream
    weighting underestimates this effect. Quantitatively, for IDDP-1
    conditions (P_res=16 MPa, T_res=500 C):

        dP ~ 4 MPa (30 kg/s):  upstream underestimates by ~10-15%
        dP ~ 8 MPa (48 kg/s):  upstream underestimates by ~30-40%

    Alternative approaches include evaluating properties at the mean
    pressure (P_res + P_bh)/2, or numerically integrating the Darcy
    equation with pressure-dependent properties. We tested the mean-
    pressure approach and found that while it gives a more accurate
    dP for a given transmissivity, it pushes calibrated transmissivity
    values higher (not lower) because the model must compensate with
    more kxb to recover the same observed WHP. Since the upstream
    approach produces calibrated transmissivities that are more
    consistent with independent permeability estimates, and since the
    analytical Darcy model is already highly idealized (homogeneous
    reservoir, no skin, no non-Darcy effects), we retain upstream
    weighting as the simpler and more transparent choice.

    Iteration scheme
    ----------------
        1. Evaluate Darcy dP at (P_res, T_current)
        2. P_bh = P_res - dP
        3. Compute feedzone T from isenthalpic expansion at P_bh
        4. Update T_current = T_feedzone
        5. Repeat until |T_new - T_old| < 1 C (typically 2-3 iters)

    Parameters
    ----------
    mass_flow_rate : float
        Mass flow rate [kg/s].
    P_reservoir_MPa : float
        Reservoir pressure at drainage boundary [MPa].
    T_reservoir_C : float
        Reservoir temperature [C].
    reservoir_params : dict or None
        Reservoir parameters.
    well_params : dict or None
        Well parameters.
    max_iterations : int
        Maximum number of temperature-correction iterations.

    Returns
    -------
    dict with keys:
        'P_bh_MPa'         : float - Bottomhole pressure [MPa]
        'h_feedzone_Jkg'    : float - Feedzone enthalpy [J/kg]
        'T_feedzone_C'      : float - Feedzone temperature [C]
        'dP_reservoir_MPa'  : float - Reservoir pressure drawdown [MPa]
    """
    # Reservoir enthalpy (constant through iteration -- isenthalpic)
    T_res_K = T_reservoir_C + 273.15
    P_res_Pa = P_reservoir_MPa * 1e6
    h_res = CP.PropsSI('H', 'T', T_res_K, 'P', P_res_Pa, 'Water')

    T_current_C = T_reservoir_C
    P_bh = T_fz_C = dP = None

    for _ in range(max_iterations):
        dP = darcy_pressure_drop(
            mass_flow_rate, P_reservoir_MPa, T_current_C,
            reservoir_params, well_params)

        P_bh = P_reservoir_MPa - dP

        # Feedzone state from isenthalpic expansion (P-h lookup)
        fz = fluid_properties_Ph(P_bh, h_res)
        T_fz_C = fz['temperature_K'] - 273.15

        if abs(T_fz_C - T_current_C) < 1.0:
            break

        T_current_C = T_fz_C

    return {
        'P_bh_MPa': P_bh,
        'h_feedzone_Jkg': h_res,
        'T_feedzone_C': T_fz_C,
        'dP_reservoir_MPa': dP,
    }


# ===================================================================
# 4. COUPLED RESERVOIR-WELLBORE MODEL
# ===================================================================

def coupled_model(mass_flow_rate, P_reservoir_MPa, T_reservoir_C,
                  rock_temperatures, reservoir_params=None,
                  well_params=None):
    """
    Coupled reservoir-wellbore model: reservoir -> BHP -> wellbore -> WHP.

    This is the central function that chains together the model
    components described in Scott (2025, Section 2 and Figure 3):
        1. Darcy reservoir flow -> bottomhole conditions (P_bh, h_fz)
        2. Wellbore simulator -> wellhead conditions
        3. Surface conditions extracted from wellbore profiles

    If the wellbore simulation terminates early (e.g., due to
    negative pressure from excessive friction at high flow rates),
    the function returns success=False. This prevents the solver
    from accepting flow rates that the wellbore cannot physically
    sustain -- the simulation must reach the surface for the
    result to be valid.

    Parameters
    ----------
    mass_flow_rate : float
        Mass flow rate [kg/s].
    P_reservoir_MPa : float
        Reservoir pressure at drainage boundary [MPa].
    T_reservoir_C : float
        Reservoir temperature [C].
    rock_temperatures : dict
        Mapping {depth_m: temperature_C} for surrounding rock.
        Must cover [0, well_depth] at intervals of delta_z.
    reservoir_params : dict or None
        Reservoir parameters (transmissivity, drainage radius).
    well_params : dict or None
        Well parameters (diameter, depth, heat loss).

    Returns
    -------
    dict with keys:
        'mass_flow_kgs'     : float - Mass flow rate [kg/s] (echo of input)
        'whp_MPa'           : float - Wellhead pressure [MPa]
        'P_bh_MPa'          : float - Bottomhole pressure [MPa]
        'h_feedzone_MJkg'   : float - Feedzone enthalpy [MJ/kg]
        'T_feedzone_C'      : float - Feedzone temperature [C]
        'h_surface_MJkg'    : float - Surface enthalpy [MJ/kg]
        'T_surface_C'       : float - Surface temperature [C]
        'dP_reservoir_MPa'  : float - Reservoir pressure drop [MPa]
        'choked'            : bool  - True if flow exceeded sound speed
        'profiles'          : tuple - Full wellbore profiles (7-tuple)
        'success'           : bool  - True if simulation completed
    """
    fail = {
        'mass_flow_kgs': mass_flow_rate,
        'whp_MPa': np.nan, 'P_bh_MPa': np.nan,
        'h_feedzone_MJkg': np.nan, 'T_feedzone_C': np.nan,
        'h_surface_MJkg': np.nan, 'T_surface_C': np.nan,
        'dP_reservoir_MPa': np.nan, 'choked': False,
        'profiles': None, 'success': False,
    }

    try:
        # --- Step 1: Reservoir -> bottomhole conditions ---
        bh = bottomhole_pressure(
            mass_flow_rate, P_reservoir_MPa, T_reservoir_C,
            reservoir_params, well_params)

        P_bh = bh['P_bh_MPa']
        h_fz = bh['h_feedzone_Jkg']

        if P_bh < 0.5:
            return fail

        # --- Step 2: Wellbore simulation ---
        *profiles_list, choked = wellbore_simulate(
            P_bh, h_fz, mass_flow_rate,
            rock_temperatures, well_params)
        profiles = tuple(profiles_list)

        # --- Step 3: Validate and extract surface conditions ---
        # Check that the simulation actually reached the surface.
        # If it terminated early (e.g., negative pressure at depth),
        # the profiles are truncated and the last point is NOT the
        # wellhead. The solver must treat this as a failure.
        last_depth = profiles[0][-1][0]  # depth of last profile point
        wp = dict(DEFAULT_WELL_PARAMS)
        if well_params is not None:
            wp.update(well_params)
        delta_z = wp.get('delta_z_m', 10)

        if last_depth > delta_z * 2:
            # Simulation terminated early -- did not reach surface
            fail['choked'] = choked
            return fail

        whp = profiles[1][-1][1]          # pressure profile, last point
        T_surface_C = profiles[0][-1][1]  # temperature profile
        h_surface_MJkg = profiles[2][-1][1]  # enthalpy profile

        return {
            'mass_flow_kgs': mass_flow_rate,
            'whp_MPa': whp,
            'P_bh_MPa': P_bh,
            'h_feedzone_MJkg': h_fz * 1e-6,
            'T_feedzone_C': bh['T_feedzone_C'],
            'h_surface_MJkg': h_surface_MJkg,
            'T_surface_C': T_surface_C,
            'dP_reservoir_MPa': bh['dP_reservoir_MPa'],
            'choked': choked,
            'profiles': profiles + (choked,),
            'success': True,
        }

    except Exception as e:
        warnings.warn(
            f"coupled_model failed: {e}",
            RuntimeWarning, stacklevel=2)
        return fail


def evaluate_at_flow_rates(flow_rates_kg_s, P_reservoir_MPa, T_reservoir_C,
                           rock_temperatures, reservoir_params=None,
                           well_params=None, max_consecutive_failures=5):
    """
    Evaluate the coupled model at a list of specified flow rates.

    Runs coupled_model() at each flow rate and collects results into
    parallel arrays. Useful for calibration workflows (comparing
    modeled WHP against observed data) and for parametric sweeps at
    fixed reservoir conditions.

    The function stores the full coupled_model() output dict for each
    point in 'coupled_results', allowing direct pass-through to
    power_cycle_analysis() without reconstructing dicts.

    Runtime warnings from the wellbore EOS are suppressed during
    evaluation to keep the console clean; failures are captured in
    the 'success' array.

    If max_consecutive_failures successive flow rates fail (simulation
    does not reach the surface), evaluation stops early. Remaining
    entries are left as NaN/False.

    Typical usage in calibration:

        observed_flow = [48, 41, 44, 30, 11, 6.4]  # kg/s
        observed_whp = [4.1, 6.2, 7.7, 10.1, 13.5, 14.1]  # MPa
        results = evaluate_at_flow_rates(
            observed_flow, P_res, T_res, rock_temps, rp, wp)
        rms = np.sqrt(np.mean(
            (results['whp_MPa'] - observed_whp)**2))

    Typical usage with power_cycle_analysis:

        results = evaluate_at_flow_rates(flows, P_res, T_res, ...)
        for i, cm in enumerate(results['coupled_results']):
            if cm is not None:
                pc = power_cycle_analysis(cm)

    Parameters
    ----------
    flow_rates_kg_s : array-like
        Mass flow rates to evaluate [kg/s].
    P_reservoir_MPa : float
        Reservoir pressure [MPa].
    T_reservoir_C : float
        Reservoir temperature [C].
    rock_temperatures : dict
        Rock temperature profile {depth_m: T_C}.
    reservoir_params : dict or None
        Reservoir parameters.
    well_params : dict or None
        Well parameters.
    max_consecutive_failures : int
        Stop evaluation after this many consecutive failures.
        Default 5. Set to None or 0 to disable.

    Returns
    -------
    dict with keys (all np.ndarrays, same length as input):
        'flow_kg_s'        : Evaluated flow rates [kg/s]
        'whp_MPa'          : Wellhead pressures [MPa]
        'P_bh_MPa'         : Bottomhole pressures [MPa]
        'T_feedzone_C'     : Feedzone temperatures [C]
        'h_feedzone_MJkg'  : Feedzone enthalpies [MJ/kg]
        'T_surface_C'      : Surface temperatures [C]
        'h_surface_MJkg'   : Surface enthalpies [MJ/kg]
        'dP_reservoir_MPa' : Reservoir pressure drops [MPa]
        'choked'           : Choked flow flags [bool]
        'success'          : Simulation success flags [bool]
        'coupled_results'  : list of coupled_model() output dicts
                             (None for failed points). Each dict can
                             be passed directly to power_cycle_analysis().
    """
    flows = np.atleast_1d(flow_rates_kg_s)
    n = len(flows)

    out = {
        'flow_kg_s': np.array(flows, dtype=float),
        'whp_MPa': np.full(n, np.nan),
        'P_bh_MPa': np.full(n, np.nan),
        'T_feedzone_C': np.full(n, np.nan),
        'h_feedzone_MJkg': np.full(n, np.nan),
        'T_surface_C': np.full(n, np.nan),
        'h_surface_MJkg': np.full(n, np.nan),
        'dP_reservoir_MPa': np.full(n, np.nan),
        'choked': np.zeros(n, dtype=bool),
        'success': np.zeros(n, dtype=bool),
        'coupled_results': [None] * n,
    }

    consecutive_failures = 0
    max_fail = max_consecutive_failures or n  # disable if 0 or None

    for i, flow in enumerate(flows):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r = coupled_model(flow, P_reservoir_MPa, T_reservoir_C,
                              rock_temperatures, reservoir_params,
                              well_params)

        out['whp_MPa'][i] = r['whp_MPa']
        out['P_bh_MPa'][i] = r['P_bh_MPa']
        out['T_feedzone_C'][i] = r['T_feedzone_C']
        out['h_feedzone_MJkg'][i] = r['h_feedzone_MJkg']
        out['T_surface_C'][i] = r['T_surface_C']
        out['h_surface_MJkg'][i] = r['h_surface_MJkg']
        out['dP_reservoir_MPa'][i] = r['dP_reservoir_MPa']
        out['choked'][i] = r['choked']
        out['success'][i] = r['success']

        if r['success']:
            # Store full dict (without profiles to keep memory light)
            cm_light = {k: v for k, v in r.items() if k != 'profiles'}
            out['coupled_results'][i] = cm_light
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_fail:
                break

    return out


# ===================================================================
# 5. FLOW RATE SOLVER (target WHP -> mass flow rate)
# ===================================================================
#
# The solver finds the mass flow rate that produces a given wellhead
# pressure. The key physical insight is that WHP is a monotonically
# decreasing function of mass flow rate: higher flow increases both
# the Darcy drawdown (lower BHP) and friction losses in the wellbore,
# both of which reduce WHP. This monotonicity makes bisection robust.
#
# The solver has two paths depending on whether a previous solution
# is available:
#
#   Warm start (previous_solution provided):
#     1. Evaluate WHP at the previous flow rate (1 eval).
#     2. Step outward to establish a bracket (1-3 evals).
#     3. Bisect the bracket to convergence (~6-8 evals).
#     Total: ~8-12 coupled_model evaluations.
#
#   Cold start (no previous solution):
#     1. Small scan of ~7 flow rates to map WHP(mdot) coarsely.
#     2. Identify bracket from scan.
#     3. Bisect to convergence (~6-8 evals).
#     Total: ~13-15 evaluations.
#
# For parametric sweeps (the primary use case), sequential calls
# use the warm path, so the total cost is approximately
# N_points * 10 coupled_model evaluations rather than N_points * 25.
#
# Near phase boundaries the WHP(mdot) curve can have steep gradients
# or minor non-monotonicity. The bisection handles this gracefully
# as long as the bracket is valid. If the model fails at a midpoint,
# the solver tries adjacent points before giving up.

def _max_flow_from_darcy(P_reservoir_MPa, T_reservoir_C,
                        reservoir_params, well_params):
    """
    Physics-based upper bound on mass flow rate from the Darcy equation.

    The maximum possible flow occurs when the entire reservoir pressure
    is consumed by drawdown (P_bh -> 0). Rearranging the Darcy equation:

        mdot_max = 2 * pi * rho * k*b * P_res / (mu * ln(r_e / r_w))

    This is an overestimate (the wellbore model will fail long before
    P_bh = 0), but it provides a physically meaningful upper bound
    for the flow rate search range. A safety factor of 0.8 is applied.

    Returns
    -------
    float - Estimated maximum mass flow rate [kg/s].
    """
    params = _merge_reservoir_params(reservoir_params)
    transmissivity_md_m = _resolve_transmissivity(params)
    r_w = _get_wellbore_radius(well_params)
    r_e = params['drainage_radius_m']

    T_K = T_reservoir_C + 273.15
    P_Pa = P_reservoir_MPa * 1e6
    rho = CP.PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water')
    mu = CP.PropsSI('V', 'T', T_K, 'P', P_Pa, 'Water')

    kb_m3 = transmissivity_md_m * MD_TO_M2
    ln_ratio = np.log(r_e / r_w)

    # mdot when dP = P_res (i.e., P_bh = 0)
    mdot_max = (2.0 * np.pi * rho * kb_m3 * P_Pa) / (mu * ln_ratio)

    # Apply safety factor; in practice BHP cannot go below ~1-2 MPa
    return 0.8 * mdot_max


def _bisect_to_convergence(eval_whp, f_lo, f_hi, whp_lo, whp_hi,
                           target_whp_MPa, tolerance_MPa,
                           max_iterations=30, verbose=False):
    """
    Bisect a bracketed interval [f_lo, f_hi] where whp_lo > target
    and whp_hi < target (or model-failed). Converge until
    |WHP - target| < tolerance or bracket width < 0.01 kg/s.

    Returns dict with 'flow_rate_kg_s', 'whp_MPa', 'converged'.
    """
    for iteration in range(max_iterations):
        f_mid = (f_lo + f_hi) / 2.0
        whp_mid = eval_whp(f_mid)

        if np.isnan(whp_mid):
            # Model failed at midpoint; try quarter-points
            for fallback in [(f_lo + f_mid) / 2, (f_mid + f_hi) / 2]:
                whp_mid = eval_whp(fallback)
                if np.isfinite(whp_mid):
                    f_mid = fallback
                    break
            if np.isnan(whp_mid):
                break

        if verbose:
            print(f'  bisect {iteration+1}: {f_mid:.2f} kg/s -> '
                  f'WHP = {whp_mid:.3f} MPa '
                  f'(err = {whp_mid - target_whp_MPa:+.3f})')

        if abs(whp_mid - target_whp_MPa) < tolerance_MPa:
            return {'flow_rate_kg_s': f_mid, 'whp_MPa': whp_mid,
                    'converged': True}

        if whp_mid > target_whp_MPa:
            f_lo, whp_lo = f_mid, whp_mid
        else:
            f_hi, whp_hi = f_mid, whp_mid

        if abs(f_hi - f_lo) < 0.01:
            if abs(whp_lo - target_whp_MPa) < abs(whp_hi - target_whp_MPa):
                return {'flow_rate_kg_s': f_lo, 'whp_MPa': whp_lo,
                        'converged': abs(whp_lo - target_whp_MPa) < tolerance_MPa}
            else:
                return {'flow_rate_kg_s': f_hi, 'whp_MPa': whp_hi,
                        'converged': abs(whp_hi - target_whp_MPa) < tolerance_MPa}

    # Exhausted: return closest
    if abs(whp_lo - target_whp_MPa) < abs(whp_hi - target_whp_MPa):
        return {'flow_rate_kg_s': f_lo, 'whp_MPa': whp_lo,
                'converged': abs(whp_lo - target_whp_MPa) < tolerance_MPa}
    return {'flow_rate_kg_s': f_hi, 'whp_MPa': whp_hi,
            'converged': abs(whp_hi - target_whp_MPa) < tolerance_MPa}


def solve_flow_for_whp(target_whp_MPa, P_reservoir_MPa, T_reservoir_C,
                       rock_temperatures, reservoir_params=None,
                       well_params=None, previous_solution=None,
                       tolerance_MPa=0.15, max_bisection=30,
                       min_flow_kg_s=None,
                       verbose=False):
    """
    Find the mass flow rate that produces a target wellhead pressure.

    Algorithm
    ---------
    The solver uses a two-phase approach with early termination.
    All internal step sizes, probes, and thresholds scale with the
    Darcy-estimated maximum flow rate, so the solver adapts
    automatically to both low-transmissivity systems (kxb ~ 10,
    max flow ~ 1 kg/s) and high-transmissivity systems (kxb ~ 4000,
    max flow ~ 200 kg/s).

    Phase 1 -- Scan flow rates upward from a starting point:
        Step size is adaptive: base_step = 5% of Darcy max
        (capped at 5 kg/s), reduced to base_step * 0.2 when WHP
        is within 2 MPa of the target. This gives coarse coverage
        far from the solution and fine resolution near it.

        Scanning stops early when (a) a direct hit is found (WHP
        within tolerance), (b) WHP drops well below target, or
        (c) the simulation becomes infeasible (wellbore cannot
        sustain that flow rate).

        If the warm start gives WHP below target, also scans
        downward to populate the cache with above-target points.

        Direct hits during scanning return immediately without
        entering Phase 2.

        After scanning, bisects to refine the maximum feasible
        flow rate. Includes gap detection: probes above the
        initial infeasibility boundary for feasible regions
        beyond localized EOS failures.

    Phase 2 -- Bisect for target WHP using cached evaluations:
        Searches the cached (flow, WHP) pairs from Phase 1 to
        find the tightest bracket around the target, preferring
        the lowest-flow crossing (important near phase transitions
        where WHP can be non-monotonic).

        If no bracket exists and WHP(max_feasible) > target:
        the well is at its capacity limit. Only declared as
        physically choked (choked=True) if the wellbore simulator
        detected supersonic flow (v > c_sound). Otherwise returns
        converged=False (numerical feasibility limit).

    Parameters
    ----------
    target_whp_MPa : float
        Target wellhead pressure [MPa].
    P_reservoir_MPa : float
        Reservoir pressure [MPa].
    T_reservoir_C : float
        Reservoir temperature [C].
    rock_temperatures : dict
        Rock temperature profile {depth_m: T_C}.
    reservoir_params : dict or None
        Reservoir parameters. The transmissivity controls the
        Darcy max flow estimate, which scales all internal solver
        parameters (step sizes, probes, thresholds).
    well_params : dict or None
        Well parameters.
    previous_solution : dict or None
        If provided, should contain {'whp': float, 'flow': float}
        from a nearby operating point. Used as starting point for
        Phase 1; if WHP is already within tolerance, returns
        immediately.
    tolerance_MPa : float
        Convergence tolerance on WHP [MPa]. Default 0.15 (1.5%
        of a typical 10 MPa target).
    max_bisection : int
        Maximum bisection iterations in Phase 2.
    verbose : bool
        If True, print solver progress (step-up/down scan,
        bracket search, bisection iterations).

    Returns
    -------
    dict
        The full coupled_model() output dict, with additional keys:
            'converged'      : bool  - True if target WHP was met
                                       (or physical choke limit found)
            'flow_rate_kg_s' : float - Alias for 'mass_flow_kgs'
        If physically choke-limited (v > c_sound), 'choked' is True,
        'converged' is True, and 'whp_MPa' will exceed the target.
        If the target is unachievable due to numerical limits,
        'choked' is False and 'converged' is False.
    """
    # Cache of coupled_model results indexed by flow rate.
    _cm_cache = {}

    def _run(flow_rate):
        """Run coupled_model, cache successful results, return dict."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = coupled_model(
                flow_rate, P_reservoir_MPa, T_reservoir_C,
                rock_temperatures, reservoir_params, well_params)
        if result['success']:
            _cm_cache[flow_rate] = result
        return result

    def _whp(flow_rate):
        """Return WHP for a flow rate, or NaN if infeasible."""
        if flow_rate in _cm_cache:
            return _cm_cache[flow_rate]['whp_MPa']
        r = _run(flow_rate)
        if r['success'] and np.isfinite(r['whp_MPa']):
            return r['whp_MPa']
        return np.nan

    def _feasible(flow_rate):
        """True if coupled_model succeeds at this flow rate."""
        return np.isfinite(_whp(flow_rate))

    def _result_for(flow_rate, converged=True, choked=False):
        """Build enriched output dict from cache or fresh run."""
        if flow_rate in _cm_cache:
            cm = _cm_cache[flow_rate]
        else:
            cm = _run(flow_rate)
        if not cm['success']:
            return {
                'mass_flow_kgs': np.nan, 'whp_MPa': np.nan,
                'P_bh_MPa': np.nan, 'h_feedzone_MJkg': np.nan,
                'T_feedzone_C': np.nan, 'h_surface_MJkg': np.nan,
                'T_surface_C': np.nan, 'dP_reservoir_MPa': np.nan,
                'choked': False, 'profiles': None,
                'success': False, 'converged': False,
                'flow_rate_kg_s': np.nan,
            }
        cm['converged'] = converged
        cm['flow_rate_kg_s'] = flow_rate
        if choked:
            cm['choked'] = True
        return cm

    # -----------------------------------------------------------------
    # Phase 1: Find maximum feasible flow rate
    # -----------------------------------------------------------------
    flow_max_darcy = _max_flow_from_darcy(
        P_reservoir_MPa, T_reservoir_C, reservoir_params, well_params)
    flow_min = max(0.1, flow_max_darcy * 0.005)
    if min_flow_kg_s is not None:
        flow_min = max(flow_min, min_flow_kg_s)

    # Start from a known-feasible flow rate
    if (previous_solution is not None
            and np.isfinite(previous_solution.get('flow', np.nan))
            and previous_solution['flow'] > 0):
        f_good = previous_solution['flow']
        # Verify it's still feasible at this pressure/depth
        if _feasible(f_good):
            # Check for direct hit
            whp_warm = _whp(f_good)
            if (np.isfinite(whp_warm)
                    and abs(whp_warm - target_whp_MPa) < tolerance_MPa):
                if verbose:
                    print(f'  Warm start direct hit: {f_good:.1f} kg/s '
                          f'-> WHP = {whp_warm:.2f} MPa')
                return _result_for(f_good, converged=True)
        else:
            f_good = None
    else:
        f_good = None

    # If no warm start, find a feasible starting point.
    # Probe at fractions of the Darcy max flow to handle both
    # low-kxb (max ~2 kg/s) and high-kxb (max ~200 kg/s) cases.
    if f_good is None:
        probes = sorted(set([
            max(0.1, flow_max_darcy * 0.01),
            max(0.5, flow_max_darcy * 0.05),
            max(1.0, flow_max_darcy * 0.1),
            max(5.0, flow_max_darcy * 0.2),
        ]))
        probes = [p for p in probes if p >= flow_min]  # respect min_flow
        for trial in probes:
            if _feasible(trial):
                f_good = trial
                break
        if f_good is None:
            if verbose:
                print('  No feasible flow rate found')
            return _result_for(np.nan, converged=False)

    # Step upward from f_good in adaptive increments until infeasible
    # or WHP drops well below target (no need to keep scanning).
    # Base step size scales with Darcy max to handle low-kxb cases.
    f_feasible = f_good
    f_infeasible = None
    base_step = max(0.1, min(5.0, flow_max_darcy * 0.05))

    # If the starting point already has WHP below target, step DOWN
    # first to find the upper bracket edge, then the scan from cold
    # probes will provide the lower bracket edge.
    whp_start = _whp(f_good)
    if (np.isfinite(whp_start) and whp_start < target_whp_MPa
            and f_good > base_step * 2):
        if verbose:
            print(f'  Warm start WHP={whp_start:.2f} < target, '
                  f'stepping down...')
        f_down = f_good
        while f_down > flow_min:
            whp_down = _whp(f_down)
            step = (max(0.2, base_step * 0.2)
                    if (np.isfinite(whp_down)
                        and abs(whp_down - target_whp_MPa) < 2.0)
                    else base_step)
            f_down -= step
            if f_down < flow_min:
                break
            if _feasible(f_down):
                whp_test = _whp(f_down)
                if verbose:
                    print(f'  step down: {f_down:.1f} kg/s -> '
                          f'WHP = {whp_test:.2f} MPa')
                # Direct hit
                if (np.isfinite(whp_test)
                        and abs(whp_test - target_whp_MPa) < tolerance_MPa):
                    if verbose:
                        print(f'  Direct hit at {f_down:.1f} kg/s')
                    return _result_for(f_down, converged=True)
                if (np.isfinite(whp_test)
                        and whp_test > target_whp_MPa + 2.0):
                    break
            else:
                break

    trial = f_feasible
    while trial < flow_max_darcy:
        # Adaptive step: 5 kg/s normally, 1 kg/s when WHP is
        # within 2 MPa of target
        whp_current = _whp(f_feasible)
        if (np.isfinite(whp_current)
                and abs(whp_current - target_whp_MPa) < 2.0):
            step = max(0.05, base_step * 0.2)  # fine step near target
        else:
            step = base_step

        trial = f_feasible + step
        if trial > flow_max_darcy:
            break

        if _feasible(trial):
            f_feasible = trial
            whp_trial = _whp(trial)
            if verbose:
                print(f'  step up: {trial:.1f} kg/s -> '
                      f'WHP = {whp_trial:.2f} MPa (ok)')

            # Direct hit: if WHP is within tolerance, return now
            if (np.isfinite(whp_trial)
                    and abs(whp_trial - target_whp_MPa) < tolerance_MPa):
                if verbose:
                    print(f'  Direct hit at {trial:.1f} kg/s')
                return _result_for(trial, converged=True)

            # Stop early: once WHP is well below target, further
            # stepping only wastes evaluations. We already have
            # enough cached points to bracket the solution.
            if (np.isfinite(whp_trial)
                    and whp_trial < target_whp_MPa - 2.0):
                if verbose:
                    print(f'  WHP well below target, stopping scan')
                break
        else:
            f_infeasible = trial
            if verbose:
                print(f'  step up: {trial:.1f} kg/s -> FAIL')
            break

    # Bisect to refine the max feasible flow
    if f_infeasible is not None:
        f_lo, f_hi = f_feasible, f_infeasible
        for _ in range(20):
            if (f_hi - f_lo) < max(0.1, base_step * 0.1):
                break
            f_mid = (f_lo + f_hi) / 2.0
            if _feasible(f_mid):
                f_lo = f_mid
                if verbose:
                    print(f'  max-flow bisect: {f_mid:.1f} kg/s -> '
                          f'WHP = {_whp(f_mid):.2f} MPa (ok)')
            else:
                f_hi = f_mid
                if verbose:
                    print(f'  max-flow bisect: {f_mid:.1f} kg/s -> FAIL')
        f_max_feasible = f_lo

        # Gap detection: probe above the infeasible boundary for
        # feasible regions. Near the critical point, EOS failures
        # can create localized infeasible gaps even though higher
        # flows work fine. If we find a feasible point above, extend.
        for gap_mult in [1.2, 1.5, 2.0, 3.0]:
            gap_trial = f_infeasible * gap_mult
            if gap_trial > flow_max_darcy:
                break
            if _feasible(gap_trial):
                if verbose:
                    print(f'  Gap detected: {gap_trial:.1f} kg/s '
                          f'feasible above boundary')
                # Re-search for the true max feasible above the gap
                f_feasible2 = gap_trial
                f_infeasible2 = None
                for step_frac2 in [0.05, 0.1, 0.2, 0.4, 0.8]:
                    trial2 = f_feasible2 * (1 + step_frac2)
                    if trial2 > flow_max_darcy:
                        break
                    if _feasible(trial2):
                        f_feasible2 = trial2
                    else:
                        f_infeasible2 = trial2
                        break
                if f_infeasible2 is not None:
                    f_lo2, f_hi2 = f_feasible2, f_infeasible2
                    for _ in range(20):
                        if (f_hi2 - f_lo2) < max(0.1, base_step * 0.1):
                            break
                        f_mid2 = (f_lo2 + f_hi2) / 2.0
                        if _feasible(f_mid2):
                            f_lo2 = f_mid2
                        else:
                            f_hi2 = f_mid2
                    f_feasible2 = f_lo2
                if f_feasible2 > f_max_feasible:
                    f_max_feasible = f_feasible2
                    if verbose:
                        print(f'  Extended max feasible to '
                              f'{f_max_feasible:.1f} kg/s')
                break  # only need to find one gap
    else:
        # Never found infeasible -- Darcy limit is the bound
        f_max_feasible = f_feasible

    whp_at_max = _whp(f_max_feasible)

    if verbose:
        print(f'  Max feasible flow: {f_max_feasible:.1f} kg/s, '
              f'WHP = {whp_at_max:.2f} MPa')

    # -----------------------------------------------------------------
    # Phase 2: Solve for target WHP within feasible range
    # -----------------------------------------------------------------

    # First, search the cached evaluations from Phase 1 for a tight
    # bracket around the target. This is the most reliable approach
    # because Phase 1 has already evaluated WHP at many flow rates.
    cached_flows = sorted(_cm_cache.keys())
    cached_pairs = [(f, _cm_cache[f]['whp_MPa']) for f in cached_flows
                     if np.isfinite(_cm_cache[f]['whp_MPa'])]

    # Find the first adjacent pair that brackets the target
    # (prefer the lowest-flow bracket)
    f_lo = f_hi = whp_lo = whp_hi = None
    bracket_found = False

    for k in range(len(cached_pairs) - 1):
        f1, w1 = cached_pairs[k]
        f2, w2 = cached_pairs[k + 1]
        if ((w1 >= target_whp_MPa >= w2)
                or (w2 >= target_whp_MPa >= w1)):
            if w1 >= target_whp_MPa:
                f_lo, whp_lo = f1, w1
                f_hi, whp_hi = f2, w2
            else:
                f_lo, whp_lo = f2, w2
                f_hi, whp_hi = f1, w1
            bracket_found = True
            if verbose:
                print(f'  Bracket from scan: [{f_lo:.1f}, {f_hi:.1f}] '
                      f'kg/s -> WHP [{whp_lo:.2f}, {whp_hi:.2f}] MPa')
            break

    if not bracket_found:
        # No bracket in cache. Check if target is achievable at all.
        if whp_at_max > target_whp_MPa:
            # Even at max feasible flow, WHP exceeds target.
            cm_max = _cm_cache.get(f_max_feasible)
            is_physical_choke = (cm_max is not None
                                 and cm_max.get('choked', False))
            if verbose:
                reason = ('choked flow' if is_physical_choke
                          else 'feasibility limit')
                print(f'  WHP({f_max_feasible:.1f})={whp_at_max:.2f} > '
                      f'target {target_whp_MPa:.2f} ({reason})')
            return _result_for(f_max_feasible,
                               converged=is_physical_choke,
                               choked=is_physical_choke)

        # Target is below all cached WHPs -- not achievable
        if verbose:
            print(f'  No bracket found in {len(cached_pairs)} '
                  f'cached evaluations')
        return _result_for(f_max_feasible, converged=False)

    # Narrow further using previous_solution if it falls inside bracket
    if (previous_solution is not None
            and np.isfinite(previous_solution.get('flow', np.nan))):
        prev_flow = previous_solution['flow']
        if f_lo < prev_flow < f_hi:
            prev_whp = _whp(prev_flow)
            if np.isfinite(prev_whp):
                if prev_whp >= target_whp_MPa:
                    f_lo, whp_lo = prev_flow, prev_whp
                else:
                    f_hi, whp_hi = prev_flow, prev_whp

    # Bisect
    for iteration in range(max_bisection):
        f_mid = (f_lo + f_hi) / 2.0
        whp_mid = _whp(f_mid)

        if np.isnan(whp_mid):
            # Simulation failed at midpoint -- shrink from high side
            f_hi = f_mid
            continue

        if verbose:
            print(f'  bisect {iteration+1}: {f_mid:.1f} kg/s -> '
                  f'WHP = {whp_mid:.3f} MPa '
                  f'(err = {whp_mid - target_whp_MPa:+.3f})')

        if abs(whp_mid - target_whp_MPa) < tolerance_MPa:
            return _result_for(f_mid, converged=True)

        if whp_mid > target_whp_MPa:
            f_lo, whp_lo = f_mid, whp_mid
        else:
            f_hi, whp_hi = f_mid, whp_mid

        if (f_hi - f_lo) < 0.01:
            # Bracket is narrow enough -- return the closer side
            if abs(whp_lo - target_whp_MPa) < abs(whp_hi - target_whp_MPa):
                return _result_for(f_lo,
                    converged=abs(whp_lo - target_whp_MPa) < tolerance_MPa)
            else:
                return _result_for(f_hi,
                    converged=abs(whp_hi - target_whp_MPa) < tolerance_MPa)

    # Exhausted iterations: return closest
    if abs(whp_lo - target_whp_MPa) < abs(whp_hi - target_whp_MPa):
        return _result_for(f_lo,
            converged=abs(whp_lo - target_whp_MPa) < tolerance_MPa)
    return _result_for(f_hi,
        converged=abs(whp_hi - target_whp_MPa) < tolerance_MPa)


# ===================================================================
# 6. DELIVERABILITY CURVE
# ===================================================================

def deliverability_curve(P_reservoir_MPa, T_reservoir_C,
                         rock_temperatures, reservoir_params=None,
                         well_params=None,
                         whp_range_MPa=(4.0, 15.0), n_points=10):
    """
    Generate a deliverability curve: mass flow rate vs. wellhead pressure.

    For each target WHP in the specified range, solves for the mass
    flow rate using solve_flow_for_whp(). Uses the previous solution
    to initialize the next, which speeds convergence across the curve.

    WHP values are swept from high to low, which ensures that the
    solver encounters progressively higher flow rates (monotonic
    traversal of the WHP-mdot curve).

    Parameters
    ----------
    P_reservoir_MPa : float
        Reservoir pressure [MPa].
    T_reservoir_C : float
        Reservoir temperature [C].
    rock_temperatures : dict
        Rock temperature profile.
    reservoir_params : dict or None
        Reservoir parameters.
    well_params : dict or None
        Well parameters.
    whp_range_MPa : tuple of (float, float)
        (min, max) wellhead pressures to evaluate [MPa].
    n_points : int
        Number of WHP values to evaluate.

    Returns
    -------
    dict with keys:
        'whp_MPa'              : np.ndarray - Wellhead pressures [MPa]
        'flow_kg_s'            : np.ndarray - Mass flow rates [kg/s]
        'transmissivity_md_m'  : float - Transmissivity used
    """
    params = _merge_reservoir_params(reservoir_params)
    transmissivity = _resolve_transmissivity(params)

    # Don't let target WHP exceed reservoir pressure
    whp_max = min(whp_range_MPa[1], P_reservoir_MPa * 0.95)
    # Sweep high to low: start with low flow, increase monotonically
    whp_values = np.linspace(whp_max, whp_range_MPa[0], n_points)

    whp_list = []
    flow_list = []

    previous_solution = None
    consecutive_failures = 0

    for whp in whp_values:
        sol = solve_flow_for_whp(
            whp, P_reservoir_MPa, T_reservoir_C,
            rock_temperatures, reservoir_params, well_params,
            previous_solution)

        flow = sol['flow_rate_kg_s']

        if np.isfinite(flow) and flow > 0:
            # Monotonicity check: flow should increase as WHP decreases
            if len(flow_list) > 0 and flow < flow_list[-1] * 0.9:
                # Suspicious: flow decreased while WHP decreased.
                # This can happen near phase boundaries. Skip this
                # point rather than introduce a kink in the curve.
                continue

            whp_list.append(whp)
            flow_list.append(flow)
            previous_solution = {'whp': whp, 'flow': flow}
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break

    # Sort by WHP ascending for conventional plotting
    if len(whp_list) > 0:
        order = np.argsort(whp_list)
        whp_arr = np.array(whp_list)[order]
        flow_arr = np.array(flow_list)[order]
    else:
        whp_arr = np.array([])
        flow_arr = np.array([])

    return {'whp_MPa': whp_arr, 'flow_kg_s': flow_arr,
            'transmissivity_md_m': transmissivity}


# ===================================================================
# 7. ROCK TEMPERATURE PROFILES
# ===================================================================
#
# The rock temperature profile must be discretized at exactly the
# same depths that wellbore_simulate() marches through. All functions
# in this section derive the depth array from well_params via
# _get_well_depths(), guaranteeing consistency.
#
# For custom profiles, use rock_temperature_from_user() which
# validates or interpolates user-supplied data to the correct depths.

def depth_for_pressure(pressure_MPa, P_shallow=15.0, P_deep=45.0,
                       depth_shallow=2000.0, depth_deep=5000.0):
    """
    Estimate well depth from reservoir pressure using a linear model.

    The depth-pressure relationship reflects the hydrostatic gradient
    in continental crust. The default parameters span the range of
    superhot systems considered in Scott (2025): shallow magmatic
    systems (~2 km, 15 MPa) to deep high-pressure systems (~5 km,
    45 MPa). Pressures outside this range are clamped to the
    boundary depths.

    Parameters
    ----------
    pressure_MPa : float
        Reservoir pressure [MPa].
    P_shallow : float
        Lower pressure anchor [MPa]. Default 15 MPa.
    P_deep : float
        Upper pressure anchor [MPa]. Default 45 MPa.
    depth_shallow : float
        Depth at P_shallow [m]. Default 2000 m.
    depth_deep : float
        Depth at P_deep [m]. Default 5000 m.

    Returns
    -------
    float
        Estimated well depth [m].
    """
    if pressure_MPa <= P_shallow:
        return depth_shallow
    elif pressure_MPa >= P_deep:
        return depth_deep
    else:
        slope = (depth_deep - depth_shallow) / (P_deep - P_shallow)
        return depth_shallow + slope * (pressure_MPa - P_shallow)


def rock_temperature_linear(well_params, T_reservoir_C, T_surface_C=10.0):
    """
    Linear rock temperature profile anchored to the reservoir temperature.

    Returns a temperature profile that increases linearly from
    T_surface at z=0 to T_reservoir at z=well_depth:

        T(z) = T_surface + (T_reservoir - T_surface) / well_depth * z

    This is the appropriate profile for the parametric analyses in
    Scott (2025), where well depth and reservoir temperature are
    co-varied and the rock temperature at the well bottom must equal
    the reservoir temperature.

    The depth array is derived from well_params['depth_m'] and
    well_params['delta_z_m'] to guarantee consistency with the
    wellbore simulator's marching grid.

    Note: for a purely conductive regime, the gradient equals q/lambda
    (heat flux divided by thermal conductivity). For example, with
    q = 155 mW/m2 and lambda = 1.5 W/m/K, the gradient is 0.103 C/m,
    giving T_reservoir ~ 320 C at 3000 m depth (with T_surface = 10 C).
    Rather than parameterizing by (q, lambda), this function takes the
    resulting (T_reservoir, well_depth) directly, which is more natural
    for parametric studies where both are specified independently.

    Parameters
    ----------
    well_params : dict or None
        Well parameters. Uses 'depth_m' and 'delta_z_m' to define
        the depth grid. If None, uses DEFAULT_WELL_PARAMS.
    T_reservoir_C : float
        Reservoir temperature at the well bottom [C].
    T_surface_C : float
        Surface temperature [C]. Default 10 C.

    Returns
    -------
    dict
        Mapping {depth_m: temperature_C}, discretized at the same
        depths used by wellbore_simulate().
    """
    depths = _get_well_depths(well_params)
    well_depth = depths[-1]
    gradient = (T_reservoir_C - T_surface_C) / well_depth
    return {d: T_surface_C + gradient * d for d in depths}


def rock_temperature_boiling(well_params, surface_pressure_MPa=0.1,
                             T_surface_C=10.0):
    """
    Boiling-point-with-depth (BPD) temperature profile.

    Computes the boiling temperature at each depth by integrating
    hydrostatic pressure downward with temperature-dependent water
    density. At each depth step:
        1. Compute water density at current (T, P) using CoolProp
        2. Add hydrostatic pressure increment: dP = rho * g * dz
        3. Find boiling temperature at new pressure: T_sat(P)

    This profile represents the maximum temperature achievable in a
    liquid-dominated hydrostatic system. It is used as the rock
    temperature profile for the IDDP-1 validation, where the shallow
    formation at Krafla is liquid-dominated and follows the BPD curve.

    The depth array is derived from well_params to guarantee
    consistency with the wellbore simulator's marching grid.

    Parameters
    ----------
    well_params : dict or None
        Well parameters (depth_m, delta_z_m). If None, uses defaults.
    surface_pressure_MPa : float
        Surface pressure [MPa]. Default 0.1 (1 atm).
    T_surface_C : float
        Surface temperature [C]. Default 10.

    Returns
    -------
    dict
        Mapping {depth_m: temperature_C}.
    """
    depths = _get_well_depths(well_params)
    boiling_temps = {}
    g = GRAVITY

    P_current_MPa = surface_pressure_MPa
    T_current_C = T_surface_C

    for i, depth in enumerate(depths):
        if depth == 0:
            boiling_temps[depth] = T_surface_C
            continue

        dz = depth - depths[i - 1]

        # Water density at current boiling conditions
        try:
            T_K = T_current_C + 273.15
            P_Pa = P_current_MPa * 1e6
            rho = CP.PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water')
        except Exception:
            rho = 1000.0 - 0.3 * T_current_C

        # Hydrostatic pressure increment
        P_current_MPa += rho * g * dz / 1e6

        # Boiling temperature at this pressure
        try:
            T_sat_K = CP.PropsSI('T', 'P', P_current_MPa * 1e6,
                                 'Q', 0, 'Water')
            T_current_C = T_sat_K - 273.15
        except Exception:
            T_current_C = min(T_current_C + 0.3 * dz / 10.0, 373.9)

        boiling_temps[depth] = T_current_C

    return boiling_temps


def rock_temperature_from_user(well_params, user_profile):
    """
    Validate or interpolate a user-supplied rock temperature profile.

    Accepts either:
        1. A dict {depth_m: temperature_C} -- validated to ensure it
           contains all required depth keys. Missing depths within the
           range are linearly interpolated from the nearest neighbors.
        2. A callable f(depth_m) -> temperature_C -- evaluated at each
           required depth.

    This guarantees that user-supplied profiles are discretized at the
    exact depths that wellbore_simulate() requires.

    Parameters
    ----------
    well_params : dict or None
        Well parameters (depth_m, delta_z_m).
    user_profile : dict or callable
        Either a {depth: T_C} mapping or a function f(z) -> T_C.

    Returns
    -------
    dict
        Mapping {depth_m: temperature_C} at the wellbore grid depths.

    Raises
    ------
    TypeError
        If user_profile is neither a dict nor callable.
    ValueError
        If the user dict does not span the required depth range and
        interpolation is not possible.
    """
    depths = _get_well_depths(well_params)

    # --- Callable: evaluate directly ---
    if callable(user_profile):
        return {d: float(user_profile(d)) for d in depths}

    # --- Dict: validate and interpolate ---
    if isinstance(user_profile, dict):
        result = {}
        user_depths = sorted(user_profile.keys())
        user_temps = [user_profile[d] for d in user_depths]

        if len(user_depths) < 2:
            raise ValueError(
                "User profile dict must contain at least 2 depth points.")

        d_min, d_max = user_depths[0], user_depths[-1]
        if d_min > depths[0] or d_max < depths[-1]:
            raise ValueError(
                f"User profile spans [{d_min}, {d_max}] m but the well "
                f"requires [{depths[0]}, {depths[-1]}] m. Cannot "
                f"extrapolate outside the supplied range.")

        for d in depths:
            if d in user_profile:
                result[d] = user_profile[d]
            else:
                # Linear interpolation from user data
                result[d] = float(np.interp(d, user_depths, user_temps))

        return result

    raise TypeError(
        f"user_profile must be a dict or callable, got {type(user_profile)}")


# ===================================================================
# TESTS (placeholder -- to be added)
# ===================================================================
# Run with: python reservoir.py