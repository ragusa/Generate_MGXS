"""Legacy Pu-239 core and HDPE shell as a managed OpenMC-only eigenvalue case."""

from generate_mgxs import (
    Case,
    ConcentricCell,
    ConcentricGeometry,
    Material,
    energy_bounds,
)


PU239 = Material(
    logical_name="pu239_material",
    name="Pu-239",
    density_g_cm3=20.0,
    composition=(("Pu239", 1.0),),
)
HDPE = Material(
    logical_name="hdpe_material",
    name="HDPE",
    density_g_cm3=0.954,
    composition=(("H", 2.0 / 3.0), ("C", 1.0 / 3.0)),
    thermal_scattering=("c_H_in_CH2",),
)


GEOMETRY = ConcentricGeometry(
    regions=(
        ConcentricCell("Pu9", PU239, "pu", 1.0),
        ConcentricCell("Hdpe", HDPE, "hdpe", 1.1),
    ),
    height_cm=2.0,
    axial_boundaries=("reflective", "reflective"),
    outer_radial_boundary="reflective",
)


CASE = Case(
    name="pu9_hdpe",
    materials=(PU239, HDPE),
    energy_groups=energy_bounds("LANL30"),
    geometry=GEOMETRY,
    source_bounds_cm=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
    run_mode="eigenvalue",
    particles_per_batch=5_000_000,
    batches=60,
    inactive_batches=10,
    scattering_order=3,
)
