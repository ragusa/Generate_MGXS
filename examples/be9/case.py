"""The one-material Be-9 reference definition, ready to pass to prepare()."""

from generate_mgxs import Case, Material, energy_bounds


BOUNDS_EV = energy_bounds("WIMS69")
CASE = Case(
    name="be9",
    materials=(Material(
        logical_name="be9",
        name="Be9",
        density_g_cm3=1.85,
        isotopes=(("Be9", 1.0),),
        temperature_k=294.0,
        thermal_scattering=("c_Be",),
    ),),
    energy_bounds_ev=BOUNDS_EV,
    # Group probabilities are derived from this one physical source definition.
    source_kind="uniform_energy",
    target_dimensions_cm=(2.0, 2.0, 100.0),
    mesh_max_width_cm=(1.0, 1.0, 50.0),
    # OpenMC histories = batches * particles_per_batch = 1,000,000.
    particles_per_batch=25_000,
    batches=40,
    scattering_order=3,
    gmres_tolerance=1.0e-10,
    gmres_max_iterations=1200,
    gmres_restart=100,
)
