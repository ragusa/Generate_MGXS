"""Homogeneous high-density polyethylene fixed-source MGXS case."""

from generate_mgxs import Case, Material


# --- Material and energy groups -------------------------------------------

HDPE = Material(
    logical_name="hdpe",
    name="HDPE",
    density_g_cm3=0.955,
    composition=(
        ("H1", 2.0),
        ("C", 1.0),
    ),
    temperature_k=294.0,
    thermal_scattering=("c_H_in_CH2",),
)

# --- Complete case definition ---------------------------------------------

CASE = Case(
    name="hdpe",
    materials=(HDPE,),
    energy_groups="SHEM-361",

    # Group probabilities are derived from this physical source definition.
    source_kind="uniform_energy",

    # Homogeneous 2 x 2 x 100 cm box with reflecting boundaries.
    target_dimensions_cm=(2.0, 2.0, 100.0),
    boundaries=("reflective",) * 6,

    # OpenMC histories = 40 * 50,000 = 2,000,000.
    particles_per_batch=50_000,
    batches=40,

    # OpenSn numerical settings.
    scattering_order=3,
    gmres_tolerance=1.0e-9,
    gmres_max_iterations=300,
    gmres_restart=100,
)
