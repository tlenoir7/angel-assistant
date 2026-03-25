"""
FreeCAD / CadQuery CAD generation for Angel (headless).
Prefers cadquery (Railway-friendly); falls back to FreeCAD Python API when available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Startup probe — Railway/Linux: visible in deploy logs for debugging cadquery + OCC.
_CQ = None
try:
    import cadquery as cq  # type: ignore

    _CQ = cq
    _log.info("Angel CAD: cadquery backend active")
except Exception as e:
    _log.error("Angel CAD: cadquery import failed: %s", e, exc_info=True)

CAD_ROOT = Path(os.getenv("ANGEL_CAD_ROOT", str(Path(tempfile.gettempdir()) / "angel_cad")))
ENGINEERING_DESIGNS_FOLDER = "Engineering Designs"
CAD_FILE_PREFIX = "CAD-"

_BACKEND: str | None = None
_BACKEND_VERSION: str | None = None
_BACKEND_DETAIL: str = ""
_FC = None
_Part = None

_SHAPE_GENERATORS = frozenset(
    {
        "box",
        "cylinder",
        "sphere",
        "cone",
        "torus",
        "airfoil",
        "fuselage",
        "nozzle",
        "pressure_vessel",
        "disc",
        "lenticular",
    }
)


def _detect_backend() -> tuple[str, str, str]:
    global _CQ, _FC, _Part
    # Prefer cadquery (headless Linux / Railway); module import may have set _CQ already.
    if _CQ is not None:
        cq = _CQ
        ver = getattr(cq, "__version__", "unknown")
        return "cadquery", ver, "cadquery (OCP/OCC)"

    try:
        import FreeCAD as App  # type: ignore
        import Part  # type: ignore

        _FC = App
        _Part = Part
        ver = getattr(App, "Version", lambda: ("?",))()
        vstr = ".".join(str(x) for x in ver[:3]) if isinstance(ver, (list, tuple)) else str(ver)
        _log.info("Angel CAD: using FreeCAD backend version %s", vstr)
        return "freecad", vstr, "FreeCAD Part module"
    except Exception as e:
        _log.warning("Angel CAD: FreeCAD unavailable: %s", e)

    return "none", "", "no CAD backend (install cadquery or FreeCAD Python)"


def _ensure_backend_loaded() -> None:
    global _BACKEND, _BACKEND_VERSION, _BACKEND_DETAIL
    if _BACKEND is None:
        _BACKEND, _BACKEND_VERSION, _BACKEND_DETAIL = _detect_backend()


def get_cad_status() -> dict[str, Any]:
    _ensure_backend_loaded()
    gens: list[str] = []
    if _BACKEND == "cadquery":
        gens = sorted(_SHAPE_GENERATORS)
    elif _BACKEND == "freecad":
        gens = ["box", "cylinder"]
    return {
        "backend": _BACKEND or "none",
        "version": _BACKEND_VERSION or "",
        "detail": _BACKEND_DETAIL,
        "generators_available": gens,
        "cad_root": str(CAD_ROOT.resolve()),
    }


def _scale_to_mm(value: float, units: str) -> float:
    u = (units or "mm").strip().lower()
    if u in ("mm", "millimeter", "millimeters"):
        return float(value)
    if u in ("m", "meter", "meters"):
        return float(value) * 1000.0
    if u in ("cm", "centimeter", "centimeters"):
        return float(value) * 10.0
    return float(value)


def _design_dir(session_id: str, design_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", design_name).strip("._") or "design"
    d = CAD_ROOT / re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id or "default") / safe_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _try_export_iges_cq(shape: Any, path: Path) -> bool:
    try:
        from OCP.IGESControl import IGESControl_Writer  # type: ignore
        from OCP.IFSelect import IFSelect_ReturnStatus  # type: ignore
        val = shape.val() if hasattr(shape, "val") else shape
        writer = IGESControl_Writer()
        writer.AddShape(val)
        status = writer.Write(str(path))
        return status == IFSelect_ReturnStatus.IFSelect_RetDone
    except Exception as ex:
        _log.debug("IGES export skipped: %s", ex)
        return False


def _export_shape_cq(shape: Any, base: Path, name: str) -> dict[str, str | None]:
    import cadquery as cq  # type: ignore

    paths: dict[str, str | None] = {"step": None, "stl": None, "iges": None}
    step_p = base / f"{name}.step"
    stl_p = base / f"{name}.stl"
    iges_p = base / f"{name}.iges"
    cq.exporters.export(shape, str(step_p))
    paths["step"] = str(step_p)
    try:
        cq.exporters.export(shape, str(stl_p))
        paths["stl"] = str(stl_p)
    except Exception as ex:
        _log.warning("STL export failed: %s", ex)
    if _try_export_iges_cq(shape, iges_p):
        paths["iges"] = str(iges_p)
    return paths


def _export_freecad_shape(shape: Any, ddir: Path, stem: str) -> dict[str, str | None]:
    paths: dict[str, str | None] = {"step": None, "stl": None, "iges": None}
    step_p = ddir / f"{stem}.step"
    stl_p = ddir / f"{stem}.stl"
    iges_p = ddir / f"{stem}.iges"
    try:
        if hasattr(shape, "exportStep"):
            shape.exportStep(str(step_p))
        else:
            import Import  # type: ignore

            Import.export([shape], str(step_p))
        paths["step"] = str(step_p)
    except Exception as ex:
        _log.warning("FreeCAD STEP export: %s", ex)
    try:
        import Mesh  # type: ignore
        import MeshPart  # type: ignore

        mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.2)
        mesh.write(str(stl_p))
        paths["stl"] = str(stl_p)
    except Exception as ex:
        _log.debug("FreeCAD STL: %s", ex)
    try:
        import Import  # type: ignore

        Import.export([shape], str(iges_p))
        if iges_p.is_file():
            paths["iges"] = str(iges_p)
    except Exception:
        pass
    return paths


def _wrap_cq_result(shape: Any, design_dir: Path, stem: str) -> dict[str, Any]:
    paths = _export_shape_cq(shape, design_dir, stem)
    return {
        "ok": True,
        "backend": "cadquery",
        "cad_object_repr": repr(shape)[:200],
        "paths": paths,
        "design_dir": str(design_dir),
    }


# --- NACA 4-digit -----------------------------------------------------------------

def _naca4_points(code: str, chord_mm: float, n: int = 120) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    c = (code or "0012").strip()
    if not c.isdigit() or len(c) != 4:
        c = "0012"
    m = int(c[0]) / 100.0
    p = int(c[1]) / 10.0
    t = int(c[2:]) / 100.0

    upper_pts: list[tuple[float, float]] = []
    lower_pts: list[tuple[float, float]] = []

    for i in range(n + 1):
        beta = math.pi * i / n
        xc = 0.5 * (1.0 - math.cos(beta))
        yt = (
            5
            * t
            * (
                0.2969 * math.sqrt(xc)
                - 0.1260 * xc
                - 0.3516 * xc**2
                + 0.2843 * xc**3
                - 0.1015 * xc**4
            )
        )
        if p > 0 and xc < p:
            yc = (m / p**2) * (2 * p * xc - xc**2)
            dyc = (2 * m / p**2) * (p - xc)
        elif p > 0:
            yc = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xc - xc**2)
            dyc = (2 * m / (1 - p) ** 2) * (p - xc)
        else:
            yc = 0.0
            dyc = 0.0
        theta = math.atan(dyc)
        x = xc * chord_mm
        upper_pts.append((x, (yc + yt * math.cos(theta)) * chord_mm))
        lower_pts.append((x, (yc - yt * math.cos(theta)) * chord_mm))

    upper = [(float(a), float(b)) for a, b in upper_pts]
    lower = list(reversed([(float(a), float(b)) for a, b in lower_pts]))
    return upper, lower


# --- cadquery generators -----------------------------------------------------------

def _cq_box(length: float, width: float, height: float) -> Any:
    import cadquery as cq  # type: ignore

    return cq.Workplane("XY").box(length, width, height)


def _cq_cylinder(radius: float, height: float) -> Any:
    import cadquery as cq  # type: ignore

    return cq.Workplane("XY").cylinder(height, radius)


def _cq_sphere(radius: float) -> Any:
    import cadquery as cq  # type: ignore

    return cq.Workplane("XY").sphere(radius)


def _cq_cone(r1: float, r2: float, height: float) -> Any:
    import cadquery as cq  # type: ignore

    return (
        cq.Workplane("XY")
        .circle(r1)
        .workplane(offset=height)
        .circle(r2)
        .loft(combine=True)
    )


def _cq_torus(major_r: float, minor_r: float) -> Any:
    import cadquery as cq  # type: ignore

    return (
        cq.Workplane("XZ")
        .center(major_r, 0)
        .circle(minor_r)
        .revolve(360, (0, 0, 0), (0, 0, 1))
    )


def generate_box(
    length: float,
    width: float,
    height: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    L, W, H = (_scale_to_mm(length, units), _scale_to_mm(width, units), _scale_to_mm(height, units))
    ddir = _design_dir(session_id, design_name)
    if _BACKEND == "cadquery":
        sh = _cq_box(L, W, H)
        r = _wrap_cq_result(sh, ddir, "box")
        r["generator"] = "box"
        return r
    if _BACKEND == "freecad" and _Part is not None:
        box = _Part.makeBox(L, W, H)
        paths = _export_freecad_shape(box, ddir, "box")
        return {
            "ok": bool(paths.get("step")),
            "backend": "freecad",
            "generator": "box",
            "paths": paths,
            "design_dir": str(ddir),
            "cad_object_repr": "Part::Box",
        }
    return {"ok": False, "error": "no CAD backend for box", "backend": _BACKEND}


def generate_cylinder(
    radius: float,
    height: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    r_mm, h_mm = _scale_to_mm(radius, units), _scale_to_mm(height, units)
    ddir = _design_dir(session_id, design_name)
    if _BACKEND == "cadquery":
        sh = _cq_cylinder(r_mm, h_mm)
        out = _wrap_cq_result(sh, ddir, "cylinder")
        out["generator"] = "cylinder"
        return out
    if _BACKEND == "freecad" and _Part is not None:
        cyl = _Part.makeCylinder(r_mm, h_mm)
        paths = _export_freecad_shape(cyl, ddir, "cylinder")
        return {
            "ok": bool(paths.get("step")),
            "backend": "freecad",
            "generator": "cylinder",
            "paths": paths,
            "design_dir": str(ddir),
            "cad_object_repr": "Part::Cylinder",
        }
    return {"ok": False, "error": "no CAD backend for cylinder", "backend": _BACKEND}


def generate_sphere(
    radius: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND == "cadquery":
        r_mm = _scale_to_mm(radius, units)
        ddir = _design_dir(session_id, design_name)
        sh = _cq_sphere(r_mm)
        out = _wrap_cq_result(sh, ddir, "sphere")
        out["generator"] = "sphere"
        return out
    return {"ok": False, "error": "cadquery required", "backend": _BACKEND}


def generate_cone(
    radius_base: float,
    radius_top: float,
    height: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND == "cadquery":
        rb, rt, h = (
            _scale_to_mm(radius_base, units),
            _scale_to_mm(radius_top, units),
            _scale_to_mm(height, units),
        )
        ddir = _design_dir(session_id, design_name)
        sh = _cq_cone(rb, rt, h)
        out = _wrap_cq_result(sh, ddir, "cone")
        out["generator"] = "cone"
        return out
    return {"ok": False, "error": "cadquery required", "backend": _BACKEND}


def generate_torus(
    radius_major: float,
    radius_minor: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND == "cadquery":
        rm, rn = _scale_to_mm(radius_major, units), _scale_to_mm(radius_minor, units)
        ddir = _design_dir(session_id, design_name)
        sh = _cq_torus(rm, rn)
        out = _wrap_cq_result(sh, ddir, "torus")
        out["generator"] = "torus"
        return out
    return {"ok": False, "error": "cadquery required", "backend": _BACKEND}


def generate_airfoil(
    naca_code: str,
    chord: float,
    span: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND != "cadquery":
        return {"ok": False, "error": "cadquery required", "backend": _BACKEND}
    import cadquery as cq  # type: ignore

    c_mm, s_mm = _scale_to_mm(chord, units), _scale_to_mm(span, units)
    upper, lower = _naca4_points(naca_code, c_mm)
    loop = upper + lower[1:-1]
    ddir = _design_dir(session_id, design_name)
    wp = cq.Workplane("XY")
    sh = wp.polyline(loop).close().extrude(s_mm)
    out = _wrap_cq_result(sh, ddir, "airfoil")
    out["generator"] = "airfoil"
    out["naca_code"] = naca_code
    return out


def generate_fuselage(
    length: float,
    max_diameter: float,
    nose_ratio: float = 0.15,
    tail_ratio: float = 0.2,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND != "cadquery":
        return {"ok": False, "error": "cadquery required", "backend": _BACKEND}
    import cadquery as cq  # type: ignore

    L = _scale_to_mm(length, units)
    D = _scale_to_mm(max_diameter, units)
    R = D / 2.0
    nose_ratio = max(0.05, min(0.45, float(nose_ratio)))
    tail_ratio = max(0.05, min(0.45, float(tail_ratio)))
    cyl_frac = 1.0 - nose_ratio - tail_ratio
    z0 = 0.0
    pts: list[tuple[float, float]] = []
    n_nose, n_cyl, n_tail = 24, 12, 24
    for i in range(n_nose + 1):
        t = i / n_nose
        z = z0 + t * (L * nose_ratio)
        r = max(R * math.sin(math.pi * t / 2), 0.02 * R)
        pts.append((z, r))
    z_cyl_start = z0 + L * nose_ratio
    for i in range(1, n_cyl + 1):
        t = i / n_cyl
        z = z_cyl_start + t * (L * cyl_frac)
        pts.append((z, R))
    z_tail_start = z_cyl_start + L * cyl_frac
    for i in range(1, n_tail + 1):
        t = i / n_tail
        z = z_tail_start + t * (L * tail_ratio)
        r = R * (1.0 - t**1.2)
        pts.append((z, max(r, 0.01 * R)))

    ddir = _design_dir(session_id, design_name)
    sh = cq.Workplane("XZ").moveTo(*pts[0]).polyline(pts[1:]).close().revolve(360, (0, 0, 0), (0, 0, 1))
    out = _wrap_cq_result(sh, ddir, "fuselage")
    out["generator"] = "fuselage"
    return out


def generate_nozzle(
    throat_radius: float,
    exit_radius: float,
    length: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
    inlet_scale: float = 2.0,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND != "cadquery":
        return {"ok": False, "error": "cadquery required", "backend": _BACKEND}
    import cadquery as cq  # type: ignore

    rt = _scale_to_mm(throat_radius, units)
    re = _scale_to_mm(exit_radius, units)
    L = _scale_to_mm(length, units)
    ri = max(rt * float(inlet_scale), rt * 1.2)
    ddir = _design_dir(session_id, design_name)
    z_conv = L * 0.35
    z_div = L - z_conv
    pts = [
        (0.0, ri),
        (z_conv * 0.5, (ri + rt) / 2),
        (z_conv, rt),
        (z_conv + z_div * 0.6, (rt + re) / 2),
        (L, re),
    ]
    sh = cq.Workplane("XZ").moveTo(*pts[0]).polyline(pts[1:]).close().revolve(360, (0, 0, 0), (0, 0, 1))
    out = _wrap_cq_result(sh, ddir, "nozzle")
    out["generator"] = "nozzle"
    return out


def generate_pressure_vessel(
    outer_radius: float,
    wall_thickness: float,
    length: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND != "cadquery":
        return {"ok": False, "error": "cadquery required", "backend": _BACKEND}
    import cadquery as cq  # type: ignore

    Ro = _scale_to_mm(outer_radius, units)
    wt = max(_scale_to_mm(wall_thickness, units), 0.1)
    Ri = max(Ro - wt, 0.05 * Ro)
    Lc = max(_scale_to_mm(length, units) - 2 * Ro, Ro * 0.5)

    ddir = _design_dir(session_id, design_name)
    cyl = cq.Workplane("XY").cylinder(Lc, Ro).translate((0, 0, -Lc / 2))
    cap1 = cq.Workplane("XY").sphere(Ro).translate((0, 0, Lc / 2))
    cap2 = cq.Workplane("XY").sphere(Ro).translate((0, 0, -Lc / 2))
    outer = cyl.union(cap1).union(cap2)
    inner_cyl = cq.Workplane("XY").cylinder(Lc + 0.2, Ri).translate((0, 0, -Lc / 2))
    inner_cap1 = cq.Workplane("XY").sphere(Ri).translate((0, 0, Lc / 2))
    inner_cap2 = cq.Workplane("XY").sphere(Ri).translate((0, 0, -Lc / 2))
    inner = inner_cyl.union(inner_cap1).union(inner_cap2)
    sh = outer.cut(inner)
    out = _wrap_cq_result(sh, ddir, "pressure_vessel")
    out["generator"] = "pressure_vessel"
    return out


def generate_disc(
    outer_radius: float,
    inner_radius: float,
    thickness: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND != "cadquery":
        return {"ok": False, "error": "cadquery required", "backend": _BACKEND}
    import cadquery as cq  # type: ignore

    ro, ri, t = (
        _scale_to_mm(outer_radius, units),
        _scale_to_mm(inner_radius, units),
        _scale_to_mm(thickness, units),
    )
    if ri >= ro:
        ri = max(0.0, ro * 0.3)
    ddir = _design_dir(session_id, design_name)
    sh = cq.Workplane("XY").circle(ro).circle(ri).extrude(t)
    out = _wrap_cq_result(sh, ddir, "disc")
    out["generator"] = "disc"
    return out


def generate_lenticular(
    diameter: float,
    thickness: float,
    units: str = "mm",
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND != "cadquery":
        return {"ok": False, "error": "cadquery required", "backend": _BACKEND}
    import cadquery as cq  # type: ignore

    D = _scale_to_mm(diameter, units)
    T = _scale_to_mm(thickness, units)
    R = D / 2.0
    half_t = T / 2.0
    n = 48
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        tpar = i / n
        z = -half_t + tpar * T
        rr = max(R * math.sqrt(max(0.0, 1.0 - (z / half_t) ** 2)), 0.02 * R)
        pts.append((z, rr))

    ddir = _design_dir(session_id, design_name)
    sh = cq.Workplane("XZ").moveTo(*pts[0]).polyline(pts[1:]).close().revolve(360, (0, 0, 0), (0, 0, 1))
    out = _wrap_cq_result(sh, ddir, "lenticular")
    out["generator"] = "lenticular"
    return out


def _location_from_spec(spec: dict[str, Any]) -> Any:
    import cadquery as cq  # type: ignore

    x = float(spec.get("x", 0) or 0)
    y = float(spec.get("y", 0) or 0)
    z = float(spec.get("z", 0) or 0)
    rx = float(spec.get("rx", 0) or 0)
    ry = float(spec.get("ry", 0) or 0)
    rz = float(spec.get("rz", 0) or 0)
    return cq.Location(cq.Vector(x, y, z), cq.Vector(rx, ry, rz))


def create_assembly(
    components: list[dict[str, Any]],
    positions: list[dict[str, Any]] | None,
    names: list[str] | None,
    *,
    session_id: str,
    design_name: str,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND != "cadquery":
        return {"ok": False, "error": "cadquery required", "backend": _BACKEND}
    import cadquery as cq  # type: ignore

    positions = positions or [{}] * len(components)
    names = names or [f"part_{i}" for i in range(len(components))]
    ddir = _design_dir(session_id, design_name)
    assy = cq.Assembly()
    for i, comp in enumerate(components):
        gen = (comp.get("generator") or comp.get("shape") or "").strip().lower()
        params = comp.get("params") if isinstance(comp.get("params"), dict) else {}
        sub = f"sub_{i}"
        tmp_name = f"{design_name}_{sub}"
        res = generate_shape(gen, params, session_id=session_id, design_name=tmp_name, context="assembly sub-part")
        if not res.get("ok"):
            return {"ok": False, "error": f"sub-component {i} failed: {res.get('error')}", "partial": res}
        # Reload exported STEP as solid for assembly (stable)
        step_path = res.get("paths", {}).get("step")
        if not step_path or not Path(step_path).is_file():
            return {"ok": False, "error": f"no STEP for sub {i}"}
        obj = cq.importers.importStep(str(step_path))
        nm = names[i] if i < len(names) else f"part_{i}"
        pos = positions[i] if i < len(positions) else {}
        assy.add(obj, name=nm, loc=_location_from_spec(pos))

    stem = "assembly"
    step_p = ddir / f"{stem}.step"
    try:
        if hasattr(assy, "save"):
            assy.save(str(step_p))
        else:
            fused = assy.toCompound()
            cq.exporters.export(fused, str(step_p))
    except Exception as e:
        try:
            fused = assy.toCompound()
            cq.exporters.export(fused, str(step_p))
        except Exception as e2:
            return {"ok": False, "error": f"{e}; {e2}", "backend": "cadquery"}

    paths: dict[str, str | None] = {"step": str(step_p), "stl": None, "iges": None}
    try:
        fused = assy.toCompound()
        cq.exporters.export(fused, str(ddir / f"{stem}.stl"))
        paths["stl"] = str(ddir / f"{stem}.stl")
        _try_export_iges_cq(fused, ddir / f"{stem}.iges")
        if (ddir / f"{stem}.iges").is_file():
            paths["iges"] = str(ddir / f"{stem}.iges")
    except Exception as ex:
        _log.warning("Assembly STL/IGES export: %s", ex)

    return {
        "ok": True,
        "backend": "cadquery",
        "generator": "assembly",
        "paths": paths,
        "design_dir": str(ddir),
        "cad_object_repr": "Assembly",
    }


def generate_shape(
    shape: str,
    params: dict[str, Any],
    *,
    session_id: str,
    design_name: str,
    context: str = "",
) -> dict[str, Any]:
    sh = (shape or "").strip().lower()
    p = dict(params or {})
    units = str(p.get("units") or "mm")
    dn = design_name

    if sh == "box":
        return generate_box(
            float(p["length"]),
            float(p["width"]),
            float(p["height"]),
            units,
            session_id=session_id,
            design_name=dn,
        )
    if sh == "cylinder":
        return generate_cylinder(float(p["radius"]), float(p["height"]), units, session_id=session_id, design_name=dn)
    if sh == "sphere":
        return generate_sphere(float(p["radius"]), units, session_id=session_id, design_name=dn)
    if sh == "cone":
        return generate_cone(
            float(p.get("radius_base", p.get("radius_bottom", 10))),
            float(p.get("radius_top", 0)),
            float(p["height"]),
            units,
            session_id=session_id,
            design_name=dn,
        )
    if sh == "torus":
        return generate_torus(
            float(p["radius_major"]),
            float(p["radius_minor"]),
            units,
            session_id=session_id,
            design_name=dn,
        )
    if sh == "airfoil":
        return generate_airfoil(
            str(p.get("naca_code", "2412")),
            float(p["chord"]),
            float(p["span"]),
            units,
            session_id=session_id,
            design_name=dn,
        )
    if sh == "fuselage":
        return generate_fuselage(
            float(p["length"]),
            float(p["max_diameter"]),
            float(p.get("nose_ratio", 0.15)),
            float(p.get("tail_ratio", 0.2)),
            units,
            session_id=session_id,
            design_name=dn,
        )
    if sh == "nozzle":
        return generate_nozzle(
            float(p["throat_radius"]),
            float(p["exit_radius"]),
            float(p["length"]),
            units,
            session_id=session_id,
            design_name=dn,
        )
    if sh == "pressure_vessel":
        return generate_pressure_vessel(
            float(p["outer_radius"]),
            float(p["wall_thickness"]),
            float(p["length"]),
            units,
            session_id=session_id,
            design_name=dn,
        )
    if sh == "disc":
        return generate_disc(
            float(p["outer_radius"]),
            float(p.get("inner_radius", 0)),
            float(p["thickness"]),
            units,
            session_id=session_id,
            design_name=dn,
        )
    if sh == "lenticular":
        return generate_lenticular(float(p["diameter"]), float(p["thickness"]), units, session_id=session_id, design_name=dn)
    if sh == "assembly":
        comps = p.get("components") or []
        if not isinstance(comps, list):
            return {"ok": False, "error": "assembly requires components[]"}
        return create_assembly(
            comps,
            list(p.get("positions") or []),
            list(p.get("names") or []),
            session_id=session_id,
            design_name=dn,
        )

    return {"ok": False, "error": f"unknown shape {shape!r}"}


_CAD_INTENT = re.compile(
    r"(?i)\b(?:"
    r"design\s+(?:a|an|the)?\s*|generate\s+(?:a\s+)?cad|create\s+(?:a\s+)?cad|"
    r"model\s+this|build\s+specs|draft\s+(?:a\s+)?|draw\s+up|"
    r"what\s+would\s+it\s+look\s+like|generate\s+(?:the\s+)?geometry|"
    r"\bstl\b|\bstep\b|\biges\b|export\s+(?:a\s+)?(?:step|stl)|"
    r"wing\s+profile|fuselage|airfoil|naca|lenticular|flying\s+saucer"
    r")\b"
)
_CAD_DIM = re.compile(
    r"\d+(\.\d+)?\s*(?:mm|cm|m|ft|in|inch|meters?)?|\d+\s*x\s*\d+",
    re.I,
)


def detect_cad_generation_intent(user_message: str) -> bool:
    raw = (user_message or "").strip()
    if len(raw) < 8:
        return False
    if not _CAD_INTENT.search(raw):
        return False
    if not _CAD_DIM.search(raw) and "naca" not in raw.lower():
        return False
    return True


def _physics_hints(physics_bundle: dict[str, Any] | None) -> dict[str, Any]:
    if not physics_bundle or not isinstance(physics_bundle, dict):
        return {}
    hints: dict[str, Any] = {}
    sim = physics_bundle.get("simulation") or {}
    inp = sim.get("inputs_used") or {}
    p = inp.get("params") if isinstance(inp.get("params"), dict) else {}
    for k in ("cross_section_area", "cross_section_area_m2", "mass", "mass_kg", "thrust", "thrust_N"):
        if k in p:
            hints[k] = p[k]
    raw = sim.get("raw_results") or {}
    if isinstance(raw, dict):
        for k in ("cross_section_area", "geometry", "fuel_mass_fraction"):
            if k in raw:
                hints[k] = raw[k]
    eng = sim.get("engine_detail") or {}
    rr = eng.get("raw_results") if isinstance(eng, dict) else {}
    if isinstance(rr, dict):
        for k in ("cross_section_area", "geometry"):
            if k in rr:
                hints[k] = rr[k]
    return hints


def generate_from_brief(
    design_brief: str | dict[str, Any],
    context: str,
    *,
    session_id: str,
    anthropic_client: Any,
    physics_constraints: dict[str, Any] | None = None,
    design_name: str | None = None,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    _ensure_backend_loaded()
    if _BACKEND == "none" or not _BACKEND:
        return {
            "ok": False,
            "error": "No CAD backend (pip install cadquery, or FreeCAD Python for box/cylinder)",
            "cad_status": get_cad_status(),
        }

    ctx = (context or "").strip()
    pb = ""
    if isinstance(design_brief, dict):
        pb = json.dumps(design_brief, indent=2)[:8000]
    else:
        pb = str(design_brief or "")[:8000]

    phys = json.dumps(physics_constraints or {}, indent=2)[:4000]

    system = (
        "You are Angel's CAD planner. Output ONLY valid JSON:\n"
        '{ "shape": "box|cylinder|sphere|cone|torus|airfoil|fuselage|nozzle|pressure_vessel|disc|lenticular|assembly",\n'
        '  "params": { ... },\n'
        '  "design_name": "short_snake_case",\n'
        '  "design_rationale": "string",\n'
        '  "assumptions": ["..."],\n'
        '  "mission_relevance": "LOW|MEDIUM|HIGH|CRITICAL"\n'
        "}\n"
        "Use millimeters in params unless units key specifies otherwise. "
        "If assembly, params.components is a list of {generator, params}. "
        "Honor physics_constraints when sizing (e.g. cross-section area → equivalent diameter)."
    )
    user = f"Context:\n{ctx}\n\nDesign brief:\n{pb}\n\nPhysics constraints JSON:\n{phys}\n"

    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2500,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

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
        plan = json.loads(text)
    except Exception:
        return {"ok": False, "error": "could not parse CAD plan JSON", "raw": text[:2000]}

    sh = str(plan.get("shape") or "").strip().lower()
    params = plan.get("params") if isinstance(plan.get("params"), dict) else {}
    dn = (design_name or plan.get("design_name") or f"design_{datetime.now(timezone.utc).strftime('%H%M%S')}").strip()
    dn = re.sub(r"[^a-zA-Z0-9._-]+", "_", dn)[:80] or "design"

    gen = generate_shape(sh, params, session_id=session_id, design_name=dn, context=ctx)
    if not gen.get("ok"):
        return {
            "ok": False,
            "error": gen.get("error"),
            "plan": plan,
            "cad_status": get_cad_status(),
        }

    ddir = Path(gen["design_dir"])
    brief_path = ddir / f"{dn}_design_brief.json"
    mr = str(plan.get("mission_relevance") or "MEDIUM").strip().upper()
    payload = {
        "plan": plan,
        "paths": gen.get("paths"),
        "backend": gen.get("backend"),
        "context": ctx,
        "physics_constraints": physics_constraints,
    }
    brief_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    out = {
        "ok": True,
        "design_name": dn,
        "shape": sh,
        "files_generated": {k: v for k, v in (gen.get("paths") or {}).items() if v},
        "design_rationale": plan.get("design_rationale") or "",
        "assumptions": list(plan.get("assumptions") or []),
        "mission_relevance": mr if mr in ("LOW", "MEDIUM", "HIGH", "CRITICAL") else "MEDIUM",
        "backend": gen.get("backend"),
        "design_dir": str(ddir),
        "design_brief_json": str(brief_path),
    }

    if mr in ("HIGH", "CRITICAL") and files_cabinet is not None:
        try:
            day = datetime.now(timezone.utc).strftime("%Y%m%d")
            h = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:10]
            fname = f"{CAD_FILE_PREFIX}{day}-{h}"
            body = json.dumps({"title": dn, "mission_relevance": mr, "payload": out}, indent=2, default=str)
            tags = ["engineering_design", f"cad:{sh}", f"relevance:{mr}"]
            files_cabinet.create_file(ENGINEERING_DESIGNS_FOLDER, fname, body, tags=tags)
            out["intelligence_file"] = {"folder": ENGINEERING_DESIGNS_FOLDER, "name": fname}
        except Exception as ex:
            out["intelligence_file"] = {"filed": False, "error": str(ex)}
    else:
        out["intelligence_file"] = {"filed": False, "reason": "below HIGH/CRITICAL or no cabinet"}

    return out


def list_designs(session_id: str) -> list[dict[str, Any]]:
    root = CAD_ROOT / re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id or "default")
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        files = [p.name for p in sub.iterdir() if p.is_file()]
        out.append({"design_name": sub.name, "files": files, "path": str(sub)})
    return out


def resolve_download_path(session_id: str, design_name: str, filename: str) -> Path | None:
    if ".." in design_name or ".." in filename or filename.startswith("/"):
        return None
    if "/" in filename or "\\" in filename:
        return None
    safe_d = re.sub(r"[^a-zA-Z0-9._-]+", "_", design_name)
    if not re.match(r"^[\w.\-]+$", filename):
        return None
    p = CAD_ROOT / re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id or "default") / safe_d / filename
    try:
        p = p.resolve()
        root = (CAD_ROOT / re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id or "default")).resolve()
        if not str(p).startswith(str(root)):
            return None
    except Exception:
        return None
    return p if p.is_file() else None


def resolve_design_dir(session_id: str, design_name: str) -> Path | None:
    """Resolve design directory safely (no creation)."""
    if ".." in design_name:
        return None
    safe_d = re.sub(r"[^a-zA-Z0-9._-]+", "_", design_name).strip("._")
    if not safe_d:
        return None
    root = CAD_ROOT / re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id or "default")
    p = root / safe_d
    try:
        p = p.resolve()
        root = root.resolve()
        if not str(p).startswith(str(root)):
            return None
    except Exception:
        return None
    return p if p.is_dir() else None


def _pick_primary_stl(design_dir: Path, design_name: str) -> Path | None:
    """Choose a best-guess STL for a design (prefers <design_name>.stl)."""
    preferred = design_dir / f"{design_name}.stl"
    if preferred.is_file():
        return preferred
    stls = sorted([p for p in design_dir.iterdir() if p.is_file() and p.suffix.lower() == ".stl"])
    return stls[0] if stls else None


def _mesh_cache_path(design_dir: Path, design_name: str) -> Path:
    return design_dir / f"{design_name}.mesh.json"


def _thumbnail_cache_path(design_dir: Path, design_name: str) -> Path:
    return design_dir / f"{design_name}.thumbnail.png"


def stl_to_mesh_json(stl_path: Path, *, design_name: str) -> dict[str, Any]:
    """
    Convert an STL file into a web-friendly mesh JSON:
    - vertices: [[x,y,z], ...]
    - faces: [i0,i1,i2, i0,i1,i2, ...]
    - normals: [[nx,ny,nz], ...] per face
    - bounds: {min:[x,y,z], max:[x,y,z]}
    """
    from stl import mesh as stlmesh  # type: ignore

    p = Path(stl_path)
    if not p.is_file():
        return {"ok": False, "error": "stl file not found"}

    m = stlmesh.Mesh.from_file(str(p))
    vectors = m.vectors  # (n, 3, 3)
    normals = getattr(m, "normals", None)

    # Build indexed vertex buffer with simple rounding-based dedupe.
    v_index: dict[tuple[float, float, float], int] = {}
    vertices: list[list[float]] = []
    faces: list[int] = []
    for tri in vectors:
        idxs: list[int] = []
        for v in tri:
            key = (round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6))
            i = v_index.get(key)
            if i is None:
                i = len(vertices)
                v_index[key] = i
                vertices.append([float(v[0]), float(v[1]), float(v[2])])
            idxs.append(i)
        faces.extend(idxs[:3])

    # Normals per face
    normals_out: list[list[float]] = []
    if normals is not None:
        try:
            for n in normals:
                normals_out.append([float(n[0]), float(n[1]), float(n[2])])
        except Exception:
            normals_out = []

    # Bounds
    try:
        import numpy as np

        arr = np.array(vertices, dtype=float) if vertices else np.zeros((0, 3), dtype=float)
        if arr.size:
            mn = arr.min(axis=0).tolist()
            mx = arr.max(axis=0).tolist()
        else:
            mn = [0.0, 0.0, 0.0]
            mx = [0.0, 0.0, 0.0]
    except Exception:
        mn = [0.0, 0.0, 0.0]
        mx = [0.0, 0.0, 0.0]

    sz_kb = int(round(p.stat().st_size / 1024.0))
    return {
        "ok": True,
        "design_name": design_name,
        "file_size_kb": sz_kb,
        "vertices": vertices,
        "faces": faces,
        "normals": normals_out,
        "bounds": {"min": mn, "max": mx},
    }


def get_or_create_mesh_json(session_id: str, design_name: str) -> dict[str, Any]:
    ddir = resolve_design_dir(session_id, design_name)
    if ddir is None:
        return {"ok": False, "error": "design not found", "design_name": design_name}
    cache = _mesh_cache_path(ddir, design_name)
    if cache.is_file():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    stl_p = _pick_primary_stl(ddir, design_name)
    if stl_p is None:
        return {"ok": False, "error": "no STL found for design", "design_name": design_name}
    out = stl_to_mesh_json(stl_p, design_name=design_name)
    if out.get("ok"):
        try:
            cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return out


def convert_stl_filename_to_mesh_json(session_id: str, design_name: str, filename: str) -> dict[str, Any]:
    p = resolve_download_path(session_id, design_name, filename)
    if p is None:
        return {"ok": False, "error": "file not found", "design_name": design_name}
    if p.suffix.lower() != ".stl":
        return {"ok": False, "error": "only .stl supported", "design_name": design_name}
    ddir = resolve_design_dir(session_id, design_name)
    if ddir is None:
        return {"ok": False, "error": "design not found", "design_name": design_name}
    out = stl_to_mesh_json(p, design_name=design_name)
    if out.get("ok"):
        try:
            # Cache per design (best-effort) and also per-file.
            _mesh_cache_path(ddir, design_name).write_text(
                json.dumps(out, ensure_ascii=False), encoding="utf-8"
            )
            (ddir / f"{p.stem}.mesh.json").write_text(
                json.dumps(out, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
    return out


def get_or_create_thumbnail_png_bytes(session_id: str, design_name: str) -> tuple[bytes | None, str | None]:
    """
    Render top/front/side thumbnail PNG for design STL.
    Returns (png_bytes, error_str).
    """
    ddir = resolve_design_dir(session_id, design_name)
    if ddir is None:
        return None, "design not found"
    cache = _thumbnail_cache_path(ddir, design_name)
    if cache.is_file():
        try:
            return cache.read_bytes(), None
        except Exception:
            pass
    stl_p = _pick_primary_stl(ddir, design_name)
    if stl_p is None:
        return None, "no STL found for design"

    try:
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from stl import mesh as stlmesh  # type: ignore

        m = stlmesh.Mesh.from_file(str(stl_p))
        tris = m.vectors  # (n, 3, 3)
        all_pts = tris.reshape(-1, 3)
        mn = all_pts.min(axis=0)
        mx = all_pts.max(axis=0)
        center = (mn + mx) / 2.0
        size = float(np.max(mx - mn)) or 1.0

        fig = plt.figure(figsize=(9, 3), dpi=180, facecolor="#0b0b10")
        views = [
            ("Top", 90, -90),
            ("Front", 0, -90),
            ("Side", 0, 0),
        ]
        for i, (title, elev, azim) in enumerate(views, start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            ax.set_facecolor("#0b0b10")
            coll = Poly3DCollection(tris, linewidths=0.0, alpha=1.0)
            coll.set_facecolor("#8b8f97")
            coll.set_edgecolor((0, 0, 0, 0))
            ax.add_collection3d(coll)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, color="#e5e7eb", fontsize=10, pad=6)
            ax.set_xlim(center[0] - size / 2, center[0] + size / 2)
            ax.set_ylim(center[1] - size / 2, center[1] + size / 2)
            ax.set_zlim(center[2] - size / 2, center[2] + size / 2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.set_axis_off()

        import io

        buf = io.BytesIO()
        plt.tight_layout(pad=0.1)
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        png = buf.getvalue()
        try:
            cache.write_bytes(png)
        except Exception:
            pass
        return png, None
    except Exception as e:
        return None, str(e)


def enrich_with_download_urls(result: dict[str, Any], base_url: str) -> dict[str, Any]:
    if not result.get("ok"):
        return result
    base = (base_url or "").rstrip("/")
    dn = result.get("design_name") or ""
    urls: dict[str, str] = {}
    for k, pth in (result.get("files_generated") or {}).items():
        if pth and dn:
            urls[str(k)] = f"{base}/api/cad/download/{dn}/{Path(str(pth)).name}"
    out = dict(result)
    out["download_urls"] = urls
    return out


def format_cad_block_for_prompt(cad_result: dict[str, Any], *, base_url: str = "") -> str:
    lines = [
        "[Angel CAD generation — structured results]",
        json.dumps(cad_result, indent=2, default=str),
        "",
        "Instructions: Summarize what was built, list downloadable STEP/STL paths or URLs, note assumptions. "
        "If backend is unavailable, say cadquery must be installed.",
    ]
    if base_url and cad_result.get("ok") and cad_result.get("design_name"):
        dn = cad_result["design_name"]
        lines.append(f"Download pattern: {base_url}/api/cad/download/{dn}/<filename>")
    return "\n".join(lines)


# Initialize backend at import (logs once)
_ensure_backend_loaded()
