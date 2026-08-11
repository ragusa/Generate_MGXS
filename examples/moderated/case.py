"""A fixed-source UO2 target embedded in an HDPE moderator.

This benchmark configuration is subcritical, so a fixed-source calculation is
appropriate. Users adapting the materials or geometry are responsible for
determining whether fixed-source or eigenvalue mode is physically appropriate.
"""

from generate_mgxs import Case, Material, NestedBoxGeometry, energy_bounds


BOUNDS_EV = energy_bounds("LANL70")


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


# --- Explicit target box inside moderator box -----------------------------
GEOMETRY = NestedBoxGeometry(
    target=UO2,
    moderator=HDPE,
    target_dimensions_cm=(0.4, 0.4, 0.4),
    outer_dimensions_cm=(1.5, 1.5, 1.5),
    boundaries=("reflective",) * 6,
)


# --- Complete moderated case ---------------------------------------------
CASE = Case(
    name="uo2_in_hdpe",
    # Material ordering is deliberately unrelated to geometric assignment.
    materials=(HDPE, UO2),
    # LANL70 is custom and therefore remains an explicit ascending boundary tuple.
    energy_groups=BOUNDS_EV,
    # OpenMC samples this physical Watt source inside the target box.
    source_kind="watt",
    watt_a_mev=0.988,
    watt_b_per_mev=2.249,
    geometry=GEOMETRY,
    # The seed's 10,000,000 total histories are distributed over 40 batches.
    particles_per_batch=250_000,
    batches=40,
    # OpenMC preserves P0--P3 MGXS for both material domains.
    scattering_order=3,
)
