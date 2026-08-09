"""The one-material Be-9 reference definition, ready to pass to prepare()."""

from generate_mgxs import Case, Material, energy_bounds


# --- Material and energy groups -------------------------------------------
BE9 = Material(
    logical_name="be9",
    name="Be9",
    density_g_cm3=1.85,
    # A mass number denotes an explicit nuclide; bare symbols such as "Fe"
    # would request OpenMC's natural-element expansion.
    composition=(("Be9", 1.0),),
    temperature_k=294.0,
    thermal_scattering=("c_Be",),
)

BOUNDS_EV = energy_bounds("WIMS69")


# --- Complete case definition ---------------------------------------------
CASE = Case(
    name="be9",
    materials=(BE9,),
    energy_bounds_ev=BOUNDS_EV,
    # Group probabilities are derived from this one physical source definition.
    source_kind="uniform_energy",
    target_dimensions_cm=(2.0, 2.0, 100.0),
    # OpenMC histories = batches * particles_per_batch = 1,000,000.
    particles_per_batch=25_000,
    batches=40,
    # OpenMC preserves P0--P3 MGXS; the independent OpenSn verifier uses P0.
    scattering_order=3,
    gmres_tolerance=1.0e-10,
    gmres_max_iterations=1200,
    gmres_restart=100,
)
