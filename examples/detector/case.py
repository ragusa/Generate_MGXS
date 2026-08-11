"""Legacy He-3/HDPE/Cd/Al detector as a managed OpenMC-only case."""

from generate_mgxs import (
    Case,
    ConcentricCell,
    ConcentricGeometry,
    Material,
    OuterBoxRegion,
)


HE3 = Material(
    logical_name="he3_material",
    name="He3",
    density_g_cm3=8.375e-4,
    composition=(("He3", 1.0),),
)
HDPE = Material(
    logical_name="hdpe_material",
    name="HDPE",
    density_g_cm3=0.954,
    composition=(("H", 2.0 / 3.0), ("C", 1.0 / 3.0)),
    thermal_scattering=("c_H_in_CH2",),
)
CADMIUM = Material(
    logical_name="cadmium_material",
    name="Cadmium",
    density_g_cm3=8.65,
    composition=(("Cd", 1.0),),
)
ALUMINUM = Material(
    logical_name="aluminum_material",
    name="Aluminum",
    density_g_cm3=2.7,
    composition=(("Al", 1.0),),
)


GEOMETRY = ConcentricGeometry(
    regions=(
        ConcentricCell("He3", HE3, "he3", 1.0),
        ConcentricCell("Hdpe", HDPE, "hdpe", 4.0),
        ConcentricCell("Cadmium", CADMIUM, "cad", 4.05),
        ConcentricCell("Alu", ALUMINUM, "alu", 4.3),
    ),
    height_cm=10.0,
    axial_boundaries=("reflective", "reflective"),
    outer_region=OuterBoxRegion("Outer", HDPE, "outer"),
    outer_half_widths_cm=(5.0, 5.0),
    outer_xy_boundaries=("reflective",) * 4,
)


CASE = Case(
    name="detector",
    materials=(HE3, HDPE, CADMIUM, ALUMINUM),
    energy_groups="XMAS-172",
    geometry=GEOMETRY,
    source_kind="uniform_energy",
    source_bounds_cm=(-5.0, -5.0, -5.0, 5.0, 5.0, 5.0),
    source_energy_bounds_ev=(0.0, 20.0e6),
    particles_per_batch=50_000,
    batches=100,
    scattering_order=3,
)
