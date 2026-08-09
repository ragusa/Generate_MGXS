"""A fixed-source UO2 target embedded in an HDPE moderator."""

from generate_mgxs import Case, Material, energy_bounds


BOUNDS_EV = energy_bounds("LANL30")


# --- Moderator material ---------------------------------------------------
HDPE = Material(
    logical_name="hdpe_moderator",
    name="HDPE moderator",
    density_g_cm3=0.955,
    composition=(("H1", 0.667), ("C12", 0.3294702), ("C13", 0.0035298)),
    thermal_scattering=("c_H_in_CH2",),
    role="moderator",
)


# --- Target material ------------------------------------------------------
UO2 = Material(
    logical_name="uo2_target",
    name="UO2",
    density_g_cm3=10.96,
    composition=(
        ("O16", 0.665047665047665), ("O17", 0.000253000253000253),
        ("O18", 0.001367001367001367), ("U234", 0.00009000009000009),
        ("U235", 0.010124010124010123), ("U236", 0.000046000046000046),
        ("U238", 0.3230723230723231),
    ),
    thermal_scattering=("c_O_in_UO2", "c_U_in_UO2"),
    role="target",
)


# --- Complete moderated case ---------------------------------------------
CASE = Case(
    name="uo2_in_hdpe",
    materials=(HDPE, UO2),  # ordering is deliberately unrelated to block assignment
    energy_bounds_ev=BOUNDS_EV,
    # The continuous OpenMC source and grouped OpenSn source share these parameters.
    source_kind="watt",
    watt_a_mev=0.988,
    watt_b_per_mev=2.249,
    target_dimensions_cm=(0.4, 0.4, 0.4),
    outer_dimensions_cm=(1.5, 1.5, 1.5),
    # The seed's 10,000,000 total histories are distributed over 40 batches.
    particles_per_batch=250_000,
    batches=40,
    # OpenMC preserves P0--P3 MGXS; the independent OpenSn verifier uses P0.
    scattering_order=3,
    gmres_tolerance=1.0e-9,
    gmres_max_iterations=300,
    gmres_restart=100,
)
