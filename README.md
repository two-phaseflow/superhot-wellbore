# superhot-wellbore

Code accompanying:

> Scott, S.W. (2026), Thermo-hydraulic drivers of superhot geothermal well performance, *Geothermics*, 141, 103784. https://doi.org/10.1016/j.geothermics.2026.103784

## Description

Python framework coupling three components for modeling single-well power output from superhot geothermal systems:

1. **Reservoir model** — Steady-state radial Darcy flow computing pressure drawdown and mass flow rate for a given reservoir pressure, temperature, and transmissivity.
2. **Wellbore model** — Thermohydraulic simulator integrating pressure and enthalpy gradients from bottomhole to wellhead, including frictional losses, gravitational head, kinetic energy, and conductive heat loss to the formation.
3. **Power cycle model** — Binary (water-based Rankine) or flash steam cycle analysis based on the predicted wellhead fluid state.

The framework is applied across the full range of superhot conditions observed globally (375–600 °C, 15–45 MPa) to quantify how reservoir pressure, temperature, and transmissivity jointly control deliverable power.

## Requirements

- Python 3.8+
- [CoolProp](http://www.coolprop.org/) — IAPWS-95 equation of state for water
- [iapws](https://github.com/jjgomez/iapws) — IAPWS-97 backward equations for near-critical routing
- NumPy
- SciPy
- Matplotlib

Install dependencies with:

```bash
pip install CoolProp iapws numpy scipy matplotlib
```

## Contents

### Core modules

| File | Description |
|------|-------------|
| `wellbore_physics.py` | Wellbore pressure and enthalpy gradient integration (Eqs. 4–5 in manuscript) |
| `reservoir.py` | Radial Darcy flow model, depth–pressure scaling, and reservoir–wellbore coupling via bisection |
| `power_cycle.py` | Binary and flash power cycle analysis with Baumann wet-stage efficiency |

### Figure scripts

| File | Figure | Description |
|------|--------|-------------|
| `figure4_sensitivity_U_roughness.py` | Fig. 4 | Sensitivity to casing roughness and heat-loss coefficient |
| `figure5_calibrate_iddp1.py` | Fig. 5 | IDDP-1 deliverability curve calibration |
| `figure6_iddp1_profiles.py` | Fig. 6 | Downhole pressure, temperature, and enthalpy profiles |
| `figure7_pressure_parametric.py` | Fig. 7 | Pressure parametric analysis (450, 475, 500 °C) |
| `figure8_temperature_parametric.py` | Fig. 8 | Temperature parametric analysis (20, 30, 40 MPa) |
| `figure9_transmissivity_parametric_analysis.py` | Fig. 9 | Transmissivity parametric analysis |
| `figure11_whpsweep.py` | Fig. 11 | Wellhead pressure sweep for selected scenarios |
| `figure12_velocity_diameter.py` | Fig. 12 | Wellhead velocity and mass flow for two casing sizes |

## Usage

Each figure script can be run independently. For example:

```bash
python figure7_pressure_parametric.py
```

Results are cached as `.pkl` files to avoid rerunning simulations. Delete the cache file to force a fresh run.

## Citation

If you use this code, please cite:

```bibtex
@article{Scott2026superhot,
  author  = {Scott, Samuel W.},
  title   = {Thermo-hydraulic drivers of superhot geothermal well performance},
  journal = {Geothermics},
  year    = {2026},
  volume  = {141},
  pages   = {103784},
  doi     = {10.1016/j.geothermics.2026.103784}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
