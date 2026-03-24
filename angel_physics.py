"""
Physics simulation engine — propulsion, EM, structural, orbital, energy, theoretical (labeled).
Uses numpy, scipy, sympy, pint, astropy; optional Materials Project via angel_chemistry.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import integrate as sci_integrate
from scipy import optimize as sci_optimize
import sympy as sp
import pint


def _rocket_equation_symbolic() -> str:
    m0, mf, isp, g0 = sp.symbols("m0 mf Isp g0", positive=True)
    dv = isp * g0 * sp.log(m0 / mf)
    return str(sp.simplify(dv))


_ROCKET_EQ_SANITY = _rocket_equation_symbolic()

try:
    import astropy.constants as const
    import astropy.units as u
except Exception:  # pragma: no cover
    const = None  # type: ignore[assignment]
    u = None  # type: ignore[assignment]

UREG = pint.UnitRegistry()
Q_ = UREG.Quantity

PHYSICS_SIM_FOLDER = "Physics Simulations"
PHYS_PREFIX = "PHYS-"

SIMULATION_TYPES = frozenset(
    {
        "propulsion",
        "em_field",
        "structural",
        "orbital",
        "energy",
        "theoretical",
    }
)

# --- Standard atmosphere (simple exponential), sea-level ---
_RHO0 = 1.225 * UREG.kg / UREG.m**3
_SCALE_H = 8500 * UREG.m
_EARTH_R = 6371000 * UREG.m
_G0 = 9.80665 * UREG.m / UREG.s**2
_EPS0 = 8.8541878128e-12 * UREG.farad / UREG.m
_MU0 = (4 * math.pi * 1e-7) * UREG.henry / UREG.m

_BUILTIN_MATERIALS: dict[str, dict[str, float]] = {
    "steel": {"E_GPa": 200, "nu": 0.3, "yield_MPa": 250, "rho_kg_m3": 7850, "alpha_1_K": 12e-6, "sigma_S_m": 1.0e6},
    "aluminum": {"E_GPa": 69, "nu": 0.33, "yield_MPa": 270, "rho_kg_m3": 2700, "alpha_1_K": 23e-6, "sigma_S_m": 3.5e7},
    "titanium": {"E_GPa": 116, "nu": 0.34, "yield_MPa": 900, "rho_kg_m3": 4500, "alpha_1_K": 8.6e-6, "sigma_S_m": 2.3e6},
    "carbon_fiber": {"E_GPa": 150, "nu": 0.3, "yield_MPa": 1500, "rho_kg_m3": 1600, "alpha_1_K": 0.1e-6, "sigma_S_m": 1e4},
    "copper": {"E_GPa": 110, "nu": 0.34, "yield_MPa": 210, "rho_kg_m3": 8960, "alpha_1_K": 16.5e-6, "sigma_S_m": 5.96e7},
}

_SIM_INTENT = re.compile(
    r"(?i)\b(?:"
    r"what\s+if|could\s+we|would\s+it\s+work|calculate|simulate|model\s+this|"
    r"is\s+it\s+possible|what\s+would\s+it\s+take|how\s+much\s+thrust|how\s+much\s+power|"
    r"how\s+much\s+energy|what\s+altitude|run\s+(?:a\s+)?sim|physics\s+sim|"
    r"delta[- ]?v|orbital\s+period|hohmann|feasib"
    r")\b"
)
_HAS_NUMBER = re.compile(r"\d+(\.\d+)?([eE][+-]?\d+)?|\d+\s*(?:kN|N|MW|kW|W|km|m|kg|tons?|g)\b", re.I)

_PHYS_CTX = re.compile(
    r"(?i)\b(?:"
    r"thrust|rocket|propulsion|orbit|satellite|iss|spacecraft|delta[- ]?v|hohmann|"
    r"altitude|re-?entry|aero|drag|kn\b|newton|kw\b|mw\b|gw\b|wh\b|joule|battery|"
    r"solar|reactor|power|tesla|gauss|magnetic|plasma|confinement|fusion|"
    r"stress|strain|beam|column|buckling|load\b|gpa|mpa|young|modulus|"
    r"warp|casimir|electromagnetic|em\s+field|lorentz"
    r")\b"
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _q_str(q: Any) -> str:
    try:
        return f"{q:~P}"
    except Exception:
        return str(q)


def _float_param(d: dict[str, Any], key: str, default: float | None = None) -> float | None:
    if key not in d or d[key] is None:
        return default
    try:
        return float(d[key])
    except (TypeError, ValueError):
        return default


def _bool_param(d: dict[str, Any], key: str, default: bool = False) -> bool:
    v = d.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")


def atmospheric_density(altitude_m: float) -> pint.Quantity:
    h = max(0.0, altitude_m) * UREG.m
    return _RHO0 * math.exp(-h.to(UREG.m).magnitude / _SCALE_H.to(UREG.m).magnitude)


def gravity_at_altitude(altitude_m: float) -> pint.Quantity:
    h = altitude_m * UREG.m
    r = _EARTH_R + h
    g = _G0 * (_EARTH_R / r) ** 2
    return g.to(UREG.m / UREG.s**2)


def simulate_propulsion(params: dict[str, Any]) -> dict[str, Any]:
    thrust = _float_param(params, "thrust", None) or _float_param(params, "thrust_N", None)
    isp = _float_param(params, "specific_impulse", None) or _float_param(params, "specific_impulse_s", None)
    mass = _float_param(params, "mass", None) or _float_param(params, "mass_kg", None)
    cd = _float_param(params, "drag_coefficient", 0.3) or 0.3
    area = _float_param(params, "cross_section_area", None) or _float_param(params, "cross_section_area_m2", 1.0)
    atmosphere = _bool_param(params, "atmosphere", True)
    alt_range = params.get("altitude_range")
    if not isinstance(alt_range, (list, tuple)) or len(alt_range) < 2:
        alt_range = (0.0, 30_000.0)
    h0, h1 = float(alt_range[0]), float(alt_range[1])

    assumptions: list[str] = [
        "Point mass vehicle; thrust aligned with velocity for delta-v estimate.",
        "Exponential atmosphere; constant Cd and reference area.",
        "Rocket equation uses standard sea-level g0 for Isp scaling of exhaust velocity.",
    ]
    if thrust is None or isp is None or mass is None:
        return {
            "ok": False,
            "error": "thrust (N), specific_impulse (s), and mass (kg) are required",
            "assumptions": assumptions,
        }

    thrust_q = thrust * UREG.newton
    m0 = mass * UREG.kg
    isp_q = isp * UREG.second
    g0 = 9.80665 * UREG.m / UREG.s**2
    ve = (isp_q * g0).to(UREG.m / UREG.s)
    area_q = area * UREG.m**2

    g_surf = gravity_at_altitude(0)
    tw = (thrust_q / (m0 * g_surf)).to(UREG.dimensionless)

    prop_mass = _float_param(params, "propellant_mass_kg", None)
    if prop_mass is None:
        pmf = _float_param(params, "propellant_mass_fraction", 0.55) or 0.55
        prop_mass = mass * max(0.0, min(0.95, pmf))
    mf = max(mass - prop_mass, 0.01 * mass)
    dv = (ve * math.log(mass / mf)).to(UREG.m / UREG.s)

    # Acceleration at liftoff (vertical, no drag at v=0)
    a0 = (thrust_q / m0 - g_surf).to(UREG.m / UREG.s**2)

    # Terminal velocity (steady vertical fall, drag = weight) at sea level and at h1
    rho0 = atmospheric_density(0) if atmosphere else 0 * UREG.kg / UREG.m**3
    rho1 = atmospheric_density(h1) if atmosphere else rho0
    vt0 = (
        math.sqrt(2 * float(m0.magnitude) * float(g_surf.to(UREG.m / UREG.s**2).magnitude) / max(float(rho0.magnitude) * cd * area, 1e-9))
        * UREG.m
        / UREG.s
        if atmosphere and float(rho0.magnitude) > 0
        else 0 * UREG.m / UREG.s
    )
    vt1 = (
        math.sqrt(2 * float(m0.magnitude) * float(gravity_at_altitude(h1).to(UREG.m / UREG.s**2).magnitude) / max(float(rho1.magnitude) * cd * area, 1e-9))
        * UREG.m
        / UREG.s
        if atmosphere and float(rho1.magnitude) > 0
        else 0 * UREG.m / UREG.s
    )

    # Ballistic range order-of-magnitude (vacuum, flat Earth, ~45°): R ≈ v²/g
    v_horiz = min(float(dv.to(UREG.m / UREG.s).magnitude) * 0.35, 9000.0)
    g0_m_s2 = float(g0.magnitude)
    range_m = (v_horiz**2 / max(g0_m_s2, 1e-6)) * 0.5
    range_est = range_m * UREG.meter

    profile: dict[str, Any] = {}
    if atmosphere:

        f_thrust_n = float(thrust_q.to(UREG.newton).magnitude)

        def dydt(_t, y):
            v, h = y
            g = float(gravity_at_altitude(max(0, h)).to(UREG.m / UREG.s**2).magnitude)
            rho = float(atmospheric_density(max(0, h)).to(UREG.kg / UREG.m**3).magnitude)
            fd = 0.5 * rho * v * abs(v) * cd * area
            a = f_thrust_n / mass - g - np.sign(v) * fd / mass
            return [a, v]

        y0 = [0.0, 0.0]
        t_span = (0.0, 120.0)
        try:
            sol = sci_integrate.solve_ivp(dydt, t_span, y0, max_step=1.0, dense_output=True)
            if sol.success:
                profile = {
                    "t_final_s": float(sol.t[-1]),
                    "v_max_m_s": float(np.max(np.abs(sol.y[0]))),
                    "altitude_max_m": float(np.max(sol.y[1])),
                }
        except Exception:
            profile = {"note": "ODE profile skipped (solver error)"}

    fuel_fraction = prop_mass / mass if mass else 0.0

    raw = {
        "thrust": _q_str(thrust_q),
        "specific_impulse": _q_str(isp_q),
        "effective_exhaust_velocity": _q_str(ve),
        "mass_initial": _q_str(m0),
        "propellant_mass": _q_str(prop_mass * UREG.kg),
        "mass_final_approx": _q_str(mf * UREG.kg),
        "thrust_to_weight_sea_level": _q_str(tw),
        "acceleration_at_liftoff": _q_str(a0),
        "delta_v_ideal_rocket_equation": _q_str(dv),
        "terminal_velocity_sea_level": _q_str(vt0) if atmosphere else "n/a (no atmosphere)",
        "terminal_velocity_at_altitude_max": _q_str(vt1) if atmosphere else "n/a",
        "range_order_of_magnitude": _q_str(range_est),
        "fuel_mass_fraction": fuel_fraction,
        "acceleration_profile_summary": profile,
    }

    limiting: list[str] = []
    if float(tw.magnitude) < 1.0:
        limiting.append("Thrust-to-weight < 1 at sea level — cannot hover/climb vertically without assistance.")
    if fuel_fraction > 0.9:
        limiting.append("Very high propellant fraction — structural margins likely unrealistic.")

    return {
        "ok": True,
        "domain": "propulsion",
        "performance_envelope": {
            "thrust_to_weight": _q_str(tw),
            "delta_v": _q_str(dv),
            "exhaust_velocity": _q_str(ve),
        },
        "limiting_factors": limiting,
        "raw_results": raw,
        "assumptions": assumptions,
    }


def simulate_em_field(params: dict[str, Any]) -> dict[str, Any]:
    B = _float_param(params, "field_strength_T", None) or _float_param(params, "B_tesla", None)
    E = _float_param(params, "electric_field_V_m", None) or _float_param(params, "field_strength_V_m", None)
    freq = _float_param(params, "frequency_Hz", 0.0) or 0.0
    power = _float_param(params, "power_input_W", None) or _float_param(params, "power_W", None)
    area = _float_param(params, "beam_area_m2", 1.0) or 1.0
    material = str(params.get("target_material") or params.get("material") or "copper").lower()
    geometry = str(params.get("geometry") or "plane_wave")

    props = _BUILTIN_MATERIALS.get(material, _BUILTIN_MATERIALS["copper"])
    sigma = props["sigma_S_m"]

    assumptions = [
        "Linear isotropic material; plane-wave or quasi-static approximations.",
        "Skin depth from good-conductor formula δ = sqrt(2/(ωμσ)).",
    ]

    omega = 2 * math.pi * freq
    mu = float(_MU0.to(UREG.henry / UREG.m).magnitude)

    raw: dict[str, Any] = {"geometry_note": geometry}

    if B is not None:
        Bq = B * UREG.tesla
        um = (Bq**2 / (2 * _MU0)).to(UREG.joule / UREG.m**3)
        raw["magnetic_energy_density"] = _q_str(um)
    if E is not None:
        Eq = E * UREG.volt / UREG.m
        ue = (_EPS0 * Eq**2 / 2).to(UREG.joule / UREG.m**3)
        raw["electric_energy_density"] = _q_str(ue)

    if omega > 0 and sigma > 0:
        delta = math.sqrt(2.0 / (omega * mu * sigma))
        raw["skin_depth_m"] = delta
    else:
        raw["skin_depth_m"] = "n/a (DC or zero conductivity)"

    if power is not None and area > 0:
        I = power / area
        prad = I / c_light
        raw["radiation_pressure_Pa"] = prad
        raw["intensity_W_m2"] = I

    q = 1.602e-19
    v = _float_param(params, "particle_velocity_m_s", 1e5) or 1e5
    if B is not None and E is not None:
        F_lorentz = q * (E + v * B)
        raw["lorentz_force_per_unit_charge_N_C_approx"] = F_lorentz
    elif B is not None:
        raw["lorentz_force_per_unit_charge_N_C_approx"] = q * v * B

    return {
        "ok": True,
        "domain": "em_field",
        "field_behavior": raw,
        "power_requirements_W": power,
        "interaction_material": material,
        "raw_results": raw,
        "assumptions": assumptions,
    }


def _resolve_structural_material(material: str) -> dict[str, Any]:
    key = (material or "steel").strip().lower().replace(" ", "_")
    if key in _BUILTIN_MATERIALS:
        return {"source": "builtin", "name": key, **_BUILTIN_MATERIALS[key]}
    try:
        import angel_chemistry as ac

        sm = ac.search_material(material)
        if sm.get("ok") and sm.get("results"):
            mid = (sm["results"][0] or {}).get("material_id")
            if mid:
                mp = ac.get_material_properties(str(mid))
                if mp.get("ok"):
                    summ = mp.get("summary") or {}
                    el = mp.get("elasticity") or {}
                    return {
                        "source": "materials_project",
                        "material_id": mid,
                        "summary": summ,
                        "elasticity": el,
                    }
    except Exception:
        pass
    return {"source": "builtin", "name": "steel", **_BUILTIN_MATERIALS["steel"]}


def simulate_structural(params: dict[str, Any]) -> dict[str, Any]:
    material = str(params.get("material") or "steel")
    shape = str(params.get("geometry_shape") or params.get("shape") or "beam_rectangular").lower()
    L = _float_param(params, "length_m", 1.0) or 1.0
    w = _float_param(params, "width_m", 0.1) or 0.1
    h = _float_param(params, "height_m", 0.2) or 0.2
    load = _float_param(params, "load_N", 10_000.0) or 10_000.0
    temp_k = _float_param(params, "temperature_K", 293.0) or 293.0
    p_ext = _float_param(params, "pressure_Pa", 101_325.0) or 101_325.0
    T_ref = _float_param(params, "reference_temperature_K", 293.0) or 293.0

    mat = _resolve_structural_material(material)
    assumptions = [
        "Elastic Euler–Bernoulli beam unless noted; simple uniaxial stress estimates.",
        "Buckling check uses pinned-pinned K=1.",
    ]

    if mat.get("source") == "builtin":
        E = (mat["E_GPa"] * 1e9) * UREG.pascal
        nu = mat["nu"]
        sig_y = (mat["yield_MPa"] * 1e6) * UREG.pascal
        rho = mat["rho_kg_m3"] * UREG.kg / UREG.m**3
        alpha = mat["alpha_1_K"]
    else:
        E = 200e9 * UREG.pascal
        nu = 0.3
        sig_y = 250e6 * UREG.pascal
        rho = 7850 * UREG.kg / UREG.m**3
        alpha = 12e-6
        assumptions.append("Materials Project entry did not yield full elastic constants — using steel-like defaults for stress estimates.")

    I = (w * h**3) / 12 * UREG.m**4
    c = (h / 2) * UREG.m
    M = (load * L / 4) * UREG.newton * UREG.m
    sigma_bend = (M * c / I).to(UREG.pascal)
    fos = float(sig_y.magnitude) / max(float(sigma_bend.magnitude), 1e-6)

    e_pa = float(E.to(UREG.pascal).magnitude)
    i_si = w * (h**3) / 12.0
    delta_m = (load * L**3) / (48.0 * e_pa * max(i_si, 1e-18))
    delta_str = _q_str(delta_m * UREG.m)

    strain_elastic = (sigma_bend / E).to(UREG.dimensionless)
    delta_T = temp_k - T_ref
    eps_thermal = alpha * delta_T
    p_crit_euler = (math.pi**2 * float(E.magnitude) * (w * h**3 / 12) / max(L**2, 1e-12)) * UREG.newton

    rho_m = float(rho.to(UREG.kg / UREG.m**3).magnitude)
    A_cross = w * h
    m_beam = rho_m * A_cross * L
    f1 = (1 / (2 * L)) * math.sqrt(float(E.magnitude) * (w * h**3 / 12) / max(m_beam * L / 3, 1e-9))

    raw = {
        "material_resolution": mat,
        "bending_stress_max_est": _q_str(sigma_bend),
        "strain_elastic": _q_str(strain_elastic),
        "factor_of_safety_vs_yield": fos,
        "deflection_midspan_simple_supported": delta_str,
        "thermal_strain_delta": eps_thermal,
        "euler_buckling_load_est": _q_str(p_crit_euler),
        "ambient_pressure_Pa": p_ext,
        "first_mode_frequency_Hz_order_of_magnitude": f1,
        "geometry": {"shape": shape, "L_m": L, "w_m": w, "h_m": h, "load_N": load},
    }

    failure_modes: list[str] = []
    if fos < 1.0:
        failure_modes.append("Yield likely exceeded under stated bending model.")
    if load > float(p_crit_euler.magnitude) * 0.5:
        failure_modes.append("Approaching Euler buckling regime — verify boundary conditions and effective length.")

    return {
        "ok": True,
        "domain": "structural",
        "structural_assessment": {"factor_of_safety": fos, "dominant_stress": _q_str(sigma_bend)},
        "failure_modes": failure_modes,
        "raw_results": raw,
        "assumptions": assumptions,
    }


def simulate_orbital(params: dict[str, Any]) -> dict[str, Any]:
    if const is None or u is None:
        return {
            "ok": False,
            "error": "astropy not available",
            "domain": "orbital",
            "assumptions": [],
        }

    alt_km = _float_param(params, "altitude_km", 400.0) or 400.0
    inc_deg = _float_param(params, "inclination_deg", 51.6) or 51.6
    mass_kg = _float_param(params, "mass_kg", 1000.0) or 1000.0
    ecc = _float_param(params, "eccentricity", 0.0) or 0.0
    dv_extra = _float_param(params, "maneuver_delta_v_m_s", None)

    R_e = const.R_earth.to(u.km).value
    mu = const.GM_earth.to(u.km**3 / u.s**2).value
    r = R_e + alt_km

    assumptions = [
        "Two-body Earth point mass; oblateness J2 neglected for first-order estimates.",
        "Circular orbit if eccentricity ~ 0.",
    ]

    v_circ_km_s = math.sqrt(mu / r)
    period_min = (2 * math.pi * math.sqrt(r**3 / mu)) / 60.0
    v_esc_km_s = math.sqrt(2 * mu / r)

    # Hohmann from r to r2
    r2_km = _float_param(params, "target_altitude_km", None)
    hohmann: dict[str, Any] = {}
    if r2_km is not None:
        r2 = R_e + r2_km
        a_tx = (r + r2) / 2
        v1 = math.sqrt(mu * (2 / r - 1 / a_tx))
        v2 = math.sqrt(mu * (2 / r2 - 1 / a_tx))
        vc1 = math.sqrt(mu / r)
        vc2 = math.sqrt(mu / r2)
        hohmann = {
            "delta_v1_km_s": abs(v1 - vc1),
            "delta_v2_km_s": abs(vc2 - v2),
            "total_delta_v_km_s": abs(v1 - vc1) + abs(vc2 - v2),
        }

    # Simple eclipse fraction: cylindrical shadow, ISS-like
    eclipse_frac = 0.38 if alt_km < 800 else 0.32

    raw = {
        "radius_km": r,
        "orbital_velocity_circular_km_s": v_circ_km_s,
        "orbital_period_minutes": period_min,
        "escape_velocity_km_s": v_esc_km_s,
        "inclination_deg": inc_deg,
        "eccentricity": ecc,
        "spacecraft_mass_kg": mass_kg,
        "approx_eclipse_fraction_of_orbit": eclipse_frac,
        "hohmann_transfer": hohmann or None,
        "maneuver_delta_v_requested_m_s": dv_extra,
    }

    return {
        "ok": True,
        "domain": "orbital",
        "orbital_parameters": raw,
        "raw_results": raw,
        "assumptions": assumptions,
    }


def simulate_energy(params: dict[str, Any]) -> dict[str, Any]:
    src = str(params.get("power_source") or "unspecified")
    p_out = _float_param(params, "output_power_W", 1000.0) or 1000.0
    eff = _float_param(params, "efficiency", 0.85) or 0.85
    eff = max(0.01, min(0.99, eff))
    storage = _float_param(params, "storage_capacity_J", None)
    duration = _float_param(params, "duration_s", 3600.0) or 3600.0
    mass_sys = _float_param(params, "system_mass_kg", None)

    e_delivered = p_out * duration * eff
    heat = p_out * (1 - eff) * duration

    assumptions = [
        "Constant output power and efficiency over duration unless load_profile given.",
    ]
    lp = params.get("load_profile")
    if isinstance(lp, list) and lp:
        e_delivered = sum(float(x.get("power_W", 0)) * float(x.get("duration_s", 0)) for x in lp if isinstance(x, dict)) * eff
        assumptions.append("Energy budget summed from piecewise load_profile.")

    runtime_lim = None
    if storage is not None and storage > 0 and p_out > 0:
        runtime_lim = storage / (p_out / eff)

    pdens = None
    if mass_sys and mass_sys > 0:
        pdens = p_out / mass_sys

    raw = {
        "power_source": src,
        "output_power_W": p_out,
        "efficiency": eff,
        "duration_s": duration,
        "energy_delivered_J": e_delivered,
        "waste_heat_J": heat,
        "storage_capacity_J": storage,
        "runtime_limited_by_storage_s": runtime_lim,
        "power_density_W_kg": pdens,
        "specific_energy_Wh_kg": (storage / 3600.0 / mass_sys) if (storage and mass_sys) else None,
    }

    bottlenecks: list[str] = []
    if heat > 1e6:
        bottlenecks.append("Substantial waste heat — thermal management may dominate design.")

    return {
        "ok": True,
        "domain": "energy",
        "energy_assessment": raw,
        "bottlenecks": bottlenecks,
        "raw_results": raw,
        "assumptions": assumptions,
    }


def simulate_theoretical(params: dict[str, Any]) -> dict[str, Any]:
    effect = str(params.get("effect_type") or "casimir").lower()
    magnitude = _float_param(params, "magnitude", 1.0) or 1.0
    energy_req = _float_param(params, "energy_requirement_J", None)
    constraints = str(params.get("known_constraints") or "")

    assumptions = [
        "THEORETICAL / SPECULATIVE — not a claim of achievable engineering.",
        "Orders of magnitude only; compare to known physics benchmarks.",
    ]
    c = 299792458.0
    hbar = 1.054571817e-34

    gap_ratio = None
    achievable = "current_physics_standard_model"
    requirements = ""
    analogues: list[str] = []

    if effect == "casimir":
        d = max(float(magnitude), 1e-15)
        f_over_a = (math.pi**2 * hbar * c) / (240 * d**4)
        requirements = f"Casimir pressure magnitude ~ {f_over_a:.3g} N/m² at separation d={d} m (parallel plate idealization)."
        gap_ratio = 1.0
        analogues = ["MEMS stiction experiments", "precision cavity QED"]
    elif effect == "alcubierre":
        requirements = (
            "Alcubierre-type warp metrics (literature) typically require exotic stress–energy violating standard energy conditions; "
            "no known macroscopic realization. Energy scales dwarf planetary-mass equivalents in naive estimates."
        )
        gap_ratio = 1e30
        achievable = "not_achievable_with_known_matter"
        analogues = ["GR exact solutions as math constructs", "quantum energy inequalities (constraints)"]
    elif effect == "plasma_confinement":
        B = magnitude if magnitude > 0.1 else 5.0
        n = 1e20
        t_ev = 10e6 * 1.602e-19
        p_plasma = n * t_ev
        mu0_loc = 4 * math.pi * 1e-7
        p_mag = B**2 / (2 * mu0_loc)
        beta = p_plasma / p_mag if p_mag > 0 else 0
        requirements = f"Rough beta ~ {beta:.3g} for n~1e20 m⁻³, T~10 keV, B={B} T (toy fusion scaling)."
        gap_ratio = max(0.01, 1.0 / max(beta, 1e-6))
        analogues = ["tokamak", "stellarator", "laser fusion"]
    elif effect == "inertial_reduction":
        requirements = "No accepted peer-reviewed mechanism for macroscopic inertial mass reduction; claims sit outside verified physics."
        gap_ratio = None
        achievable = "unverified_speculation"
        analogues = ["Mach effect conjectures (contested)", "ordinary propulsion"]
    elif effect == "gravitational_shielding":
        requirements = "Macroscopic gravitational shielding is not supported by GR or tested positive in reproducible experiments."
        gap_ratio = None
        achievable = "unverified_speculation"
        analogues = ["torsion balance tests", "LIGO / precision gravity"]
    else:
        requirements = f"Unknown effect_type {effect!r}; no calculation."
        gap_ratio = None

    raw = {
        "effect_type": effect,
        "magnitude_input": magnitude,
        "energy_requirement_J": energy_req,
        "known_constraints_note": constraints,
        "theoretical_requirements": requirements,
        "gap_ratio_order_of_magnitude": gap_ratio,
        "closest_real_world_analogues": analogues,
        "achievability_label": achievable,
    }

    return {
        "ok": True,
        "domain": "theoretical",
        "feasibility_label": "THEORETICAL",
        "raw_results": raw,
        "assumptions": assumptions,
    }


def _dispatch_simulation(simulation_type: str, params: dict[str, Any]) -> dict[str, Any]:
    st = (simulation_type or "").strip().lower().replace("-", "_")
    if st == "propulsion":
        return simulate_propulsion(params)
    if st in ("em_field", "em", "electromagnetic"):
        return simulate_em_field(params)
    if st == "structural":
        return simulate_structural(params)
    if st == "orbital":
        return simulate_orbital(params)
    if st == "energy":
        return simulate_energy(params)
    if st == "theoretical":
        return simulate_theoretical(params)
    return {"ok": False, "error": f"unknown simulation_type: {simulation_type!r}", "domain": None}


def _heuristic_feasibility(simulation_type: str, engine_out: dict[str, Any]) -> str:
    if simulation_type == "theoretical" or engine_out.get("domain") == "theoretical":
        return "THEORETICAL"
    if not engine_out.get("ok"):
        return "INFEASIBLE"
    if simulation_type == "propulsion":
        tws = engine_out.get("performance_envelope", {}).get("thrust_to_weight", "")
        try:
            if "dimensionless" in str(tws):
                pass
        except Exception:
            pass
        raw = engine_out.get("raw_results") or {}
        fos = None
        if isinstance(raw.get("thrust_to_weight_sea_level"), str):
            m = re.search(r"([\d.]+)", raw["thrust_to_weight_sea_level"])
            if m and float(m.group(1)) < 1.0:
                return "INFEASIBLE"
        if engine_out.get("limiting_factors"):
            return "MARGINAL"
        return "FEASIBLE"
    if simulation_type == "structural":
        fos = (engine_out.get("structural_assessment") or {}).get("factor_of_safety")
        if fos is None:
            fos = (engine_out.get("raw_results") or {}).get("factor_of_safety_vs_yield")
        if fos is not None and fos < 1.0:
            return "INFEASIBLE"
        if fos is not None and fos < 1.5:
            return "MARGINAL"
        return "FEASIBLE"
    return "FEASIBLE"


def _claude_interpret_simulation(
    simulation_type: str,
    params: dict[str, Any],
    context: str,
    engine_out: dict[str, Any],
    anthropic_client: Any,
) -> dict[str, Any]:
    payload = json.dumps(
        {"simulation_type": simulation_type, "params": params, "engine": engine_out, "context": context},
        ensure_ascii=False,
        indent=2,
    )[:65_000]
    system = (
        "You interpret physics simulation outputs for Tyler (mission-aware, honest about limits). "
        "Output ONLY valid JSON with keys:\n"
        "- summary (string, 3-5 sentences plain English)\n"
        "- feasibility (one of: FEASIBLE, MARGINAL, INFEASIBLE, THEORETICAL) — align with engine math when possible\n"
        "- limiting_factors (array of strings)\n"
        "- optimization_suggestions (array of strings): parameter changes that would improve feasibility\n"
        "- confidence (HIGH, MEDIUM, or LOW) based on model assumptions\n"
        "- mission_relevance (LOW, MEDIUM, HIGH, CRITICAL)\n"
        "- file_to_intelligence (boolean): true only if mission_relevance is CRITICAL or results are uniquely mission-decisive\n"
        "- assumptions_made (array of strings)\n"
        "- sensitivity (array of strings): which parameters most affect the outcome\n"
        "For THEORETICAL domain, stress speculation and standard physics constraints."
    )
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nSimulation payload:\n{payload}"}],
        )
    except Exception as e:
        return {
            "summary": f"Interpretation failed: {e}",
            "feasibility": _heuristic_feasibility(simulation_type, engine_out),
            "limiting_factors": [],
            "optimization_suggestions": [],
            "confidence": "LOW",
            "mission_relevance": "LOW",
            "file_to_intelligence": False,
            "assumptions_made": list(engine_out.get("assumptions") or []),
            "sensitivity": [],
        }

    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    text = text.strip()
    try:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i : j + 1]
        return json.loads(text)
    except Exception:
        pass
    return {
        "summary": text[:4000] if text else "No interpretation.",
        "feasibility": _heuristic_feasibility(simulation_type, engine_out),
        "limiting_factors": [],
        "optimization_suggestions": [],
        "confidence": "MEDIUM",
        "mission_relevance": "MEDIUM",
        "file_to_intelligence": False,
        "assumptions_made": list(engine_out.get("assumptions") or []),
        "sensitivity": [],
    }


def _cross_refs_physics(files_cabinet: Any | None, query: str) -> dict[str, Any]:
    refs: dict[str, Any] = {"research_intelligence": [], "chemistry_intelligence": []}
    if files_cabinet is None or not query.strip():
        return refs
    terms = list(dict.fromkeys([w for w in re.findall(r"[A-Za-z]{5,}", query.lower())][:5]))
    seen: set[tuple[str, str]] = set()
    for t in terms:
        try:
            hits = files_cabinet.search_files(t)
        except Exception:
            continue
        for h in hits[:8]:
            if not isinstance(h, dict):
                continue
            folder = (h.get("folder") or "").strip()
            name = (h.get("name") or "").strip()
            k = (folder, name)
            if k in seen:
                continue
            seen.add(k)
            if folder == "Research Intelligence":
                refs["research_intelligence"].append({"file": name, "folder": folder})
            elif folder == "Chemistry Intelligence":
                refs["chemistry_intelligence"].append({"file": name, "folder": folder})
    return refs


def maybe_file_physics_simulation(
    full_output: dict[str, Any],
    files_cabinet: Any | None,
    *,
    simulation_type: str,
    context: str,
) -> dict[str, Any]:
    if files_cabinet is None:
        return {"filed": False, "reason": "no files_cabinet"}
    mr = (full_output.get("mission_relevance") or "LOW").strip().upper()
    if mr != "CRITICAL":
        return {"filed": False, "reason": "mission_relevance not CRITICAL"}

    day = _now_utc().strftime("%Y%m%d")
    h = hashlib.sha256(json.dumps(full_output.get("raw_results"), sort_keys=True, default=str).encode()).hexdigest()[:10]
    fname = f"{PHYS_PREFIX}{day}-{h}"
    feas = (full_output.get("feasibility") or "UNKNOWN").strip().upper()
    tech = (full_output.get("technology_area_guess") or simulation_type or "general")[:80]
    tags = [
        "physics_simulation",
        f"type:{simulation_type}",
        f"feasibility:{feas}",
        f"tech:{tech}",
    ]
    body_obj = dict(full_output)
    body_obj["cross_references"] = _cross_refs_physics(files_cabinet, context + " " + simulation_type)
    body = json.dumps(body_obj, ensure_ascii=False, indent=2)
    try:
        files_cabinet.create_file(PHYSICS_SIM_FOLDER, fname, body, tags=tags)
        return {"filed": True, "cabinet_file": fname, "folder": PHYSICS_SIM_FOLDER}
    except ValueError:
        fname2 = f"{PHYS_PREFIX}{day}-{h}-b"
        try:
            files_cabinet.create_file(PHYSICS_SIM_FOLDER, fname2, body, tags=tags)
            return {"filed": True, "cabinet_file": fname2, "folder": PHYSICS_SIM_FOLDER}
        except Exception as e:
            return {"filed": False, "error": str(e)}
    except Exception as e:
        return {"filed": False, "error": str(e)}


def run_physics_simulation(
    simulation_type: str,
    params: dict[str, Any],
    context: str,
    *,
    anthropic_client: Any,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    st = (simulation_type or "").strip().lower().replace("-", "_")
    ctx = (context or "").strip()
    p = dict(params or {})

    engine = _dispatch_simulation(st, p)
    interp = _claude_interpret_simulation(st, p, ctx, engine, anthropic_client)

    feas = (interp.get("feasibility") or _heuristic_feasibility(st, engine)).strip().upper()
    if feas not in ("FEASIBLE", "MARGINAL", "INFEASIBLE", "THEORETICAL"):
        feas = _heuristic_feasibility(st, engine)

    mr = (interp.get("mission_relevance") or "MEDIUM").strip().upper()
    if mr not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        mr = "MEDIUM"

    inputs_used: dict[str, Any] = {"params": p, "units_system": "SI + pint annotations in raw_results strings"}

    out: dict[str, Any] = {
        "simulation_type": st,
        "inputs_used": inputs_used,
        "raw_results": engine.get("raw_results") or engine,
        "feasibility": feas,
        "summary": interp.get("summary") or "",
        "limiting_factors": list(interp.get("limiting_factors") or []),
        "optimization_suggestions": list(interp.get("optimization_suggestions") or []),
        "confidence": (interp.get("confidence") or "MEDIUM").strip().upper(),
        "mission_relevance": mr,
        "file_to_intelligence": bool(interp.get("file_to_intelligence")),
        "assumptions": list(interp.get("assumptions_made") or []) + list(engine.get("assumptions") or []),
        "sensitivity": list(interp.get("sensitivity") or []),
        "engine_detail": {k: v for k, v in engine.items() if k != "raw_results"},
        "intelligence_filing": {"filed": False},
        "technology_area_guess": st,
    }

    if mr == "CRITICAL":
        out["intelligence_filing"] = maybe_file_physics_simulation(
            out, files_cabinet, simulation_type=st, context=ctx
        )
    else:
        out["intelligence_filing"] = {"filed": False, "reason": "auto-file only when mission_relevance is CRITICAL"}

    return out


def extract_simulation_params(user_message: str, anthropic_client: Any) -> dict[str, Any]:
    msg = (user_message or "").strip()
    system = (
        "You extract physics simulation parameters from user text. "
        "Output ONLY valid JSON:\n"
        '{ "simulation_type": one of propulsion|em_field|structural|orbital|energy|theoretical, '
        '"params": { ... numeric SI fields as needed ... }, '
        '"confidence": "HIGH"|"MEDIUM"|"LOW", '
        '"ambiguities": [ string, ... ] }\n'
        "Map synonyms: thrust/rocket→propulsion; orbit/ISS/satellite→orbital; beam/B field/Tesla→em_field; "
        "beam stress/load→structural; battery/power/kW→energy; warp/Casimir/plasma confinement→theoretical with effect_type.\n"
        "Use meters, kg, N, s, W, J, Hz, Tesla, V/m as appropriate."
    )
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1200,
            temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": msg[:12000]}],
        )
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "simulation_type": None,
            "params": {},
            "confidence": "LOW",
            "ambiguities": ["API error"],
        }

    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    text = text.strip()
    try:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i : j + 1]
        data = json.loads(text)
    except Exception:
        return {
            "ok": False,
            "error": "parse_error",
            "simulation_type": None,
            "params": {},
            "confidence": "LOW",
            "ambiguities": ["Could not parse model output"],
        }

    st = str(data.get("simulation_type") or "").strip().lower().replace("-", "_")
    if st == "electromagnetic":
        st = "em_field"
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    return {
        "ok": True,
        "simulation_type": st,
        "params": params,
        "confidence": str(data.get("confidence") or "MEDIUM").upper(),
        "ambiguities": list(data.get("ambiguities") or []),
    }


def detect_physics_simulation_intent(user_message: str) -> bool:
    raw = (user_message or "").strip()
    if len(raw) < 10:
        return False
    if not _SIM_INTENT.search(raw):
        return False
    if not _HAS_NUMBER.search(raw):
        return False
    if not _PHYS_CTX.search(raw):
        return False
    return True


def run_physics_natural(
    user_message: str,
    context: str,
    *,
    anthropic_client: Any,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    ext = extract_simulation_params(user_message, anthropic_client)
    if not ext.get("ok") or not ext.get("simulation_type"):
        return {
            "ok": False,
            "error": ext.get("error") or "extraction failed",
            "extraction": ext,
        }
    sim = run_physics_simulation(
        ext["simulation_type"],
        ext["params"],
        context or user_message,
        anthropic_client=anthropic_client,
        files_cabinet=files_cabinet,
    )
    return {"ok": True, "extraction": ext, "simulation": sim}


def format_physics_chat_block(
    user_message: str,
    *,
    anthropic_client: Any,
    files_cabinet: Any | None,
) -> str:
    if not detect_physics_simulation_intent(user_message):
        return ""
    try:
        out = run_physics_natural(
            user_message,
            user_message[:4000],
            anthropic_client=anthropic_client,
            files_cabinet=files_cabinet,
        )
        if not out.get("ok"):
            return (
                "[Angel physics simulation — extraction error]\n"
                + json.dumps(out, ensure_ascii=False, indent=2, default=str)
            )
        return format_physics_block_for_prompt(out["simulation"])
    except Exception as e:
        return (
            "[Angel physics simulation — error]\n"
            + json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
        )


def format_physics_block_for_prompt(simulation_result: dict[str, Any]) -> str:
    return (
        "[Angel physics simulation — structured results]\n"
        + json.dumps(simulation_result, ensure_ascii=False, indent=2, default=str)
        + "\n\nInstructions: Lead with **feasibility** (" + str(simulation_result.get("feasibility")) + "). "
        "Summarize numbers conversationally; state assumptions; offer to vary parameters and rerun."
    )


def physics_library_status() -> dict[str, Any]:
    out: dict[str, Any] = {
        "domains": list(SIMULATION_TYPES),
        "numpy": getattr(np, "__version__", "?"),
        "scipy": _safe_mod_version("scipy"),
        "sympy": getattr(sp, "__version__", "?"),
        "sympy_rocket_dv_symbolic": _ROCKET_EQ_SANITY,
        "pint": _safe_mod_version("pint"),
        "astropy": _safe_mod_version("astropy") if const is not None else "unavailable",
    }
    return out


def _safe_mod_version(name: str) -> str:
    try:
        import importlib

        m = importlib.import_module(name)
        return getattr(m, "__version__", "?")
    except Exception:
        return "unavailable"
