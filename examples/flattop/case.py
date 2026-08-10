"""Homogeneous FlatTop natural-uranium k-eigenvalue case definition."""

from generate_mgxs import Case, Material, energy_bounds


# The historical OpenMC notebook supplied these relative atomic amounts with
# bulk density specified separately. Generate_MGXS preserves them verbatim.
FLATTOP = Material(
    logical_name="flattop_nu",
    name="FlatTop_NU",
    density_g_cm3=18.823124,
    composition=(
        ("U234", 2.5759e-06),
        ("U235", 3.4428e-04),
        ("U238", 4.7441e-02),
    ),
    temperature_k=294.0,
)


CASE = Case(
    name="flattop_nu",
    materials=(FLATTOP,),
    energy_groups=energy_bounds("LANL30"),
    run_mode="eigenvalue",
    target_dimensions_cm=(2.0, 2.0, 2.0),
    boundaries=("reflective",) * 6,
    particles_per_batch=20_000,
    batches=520,
    inactive_batches=120,
    scattering_order=7,
    keigen_tolerance=1.0e-8,
    keigen_max_iterations=1000,
)
