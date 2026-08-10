from pathlib import Path

import numpy as np
import pytest

from generate_mgxs import Case, Material, energy_bounds, prepare, source_probabilities
from conftest import material, tiny_case


def test_named_group_structures_are_ascending():
    """All public scientific arrays use increasing physical energy."""
    from openmc.mgxs import GROUP_STRUCTURES

    assert len(energy_bounds("WIMS69")) == 70
    assert len(energy_bounds("LANL30")) == 31
    assert np.all(np.diff(energy_bounds("WIMS69")) > 0.0)
    np.testing.assert_array_equal(
        energy_bounds("SHEM-361"),
        GROUP_STRUCTURES["SHEM-361"],
    )


def test_unknown_group_structure_is_rejected():
    with pytest.raises(ValueError, match="unknown OpenMC energy-group structure"):
        energy_bounds("other")


def test_case_resolves_openmc_standard_name_to_canonical_boundaries():
    """The OpenMC registry, not a package copy, owns standard group edges."""
    from openmc.mgxs import GROUP_STRUCTURES

    case = Case(
        name="shem",
        materials=(material(),),
        energy_groups="SHEM-361",
        source_kind="uniform_energy",
        target_dimensions_cm=(1.0, 1.0, 1.0),
    )

    assert case.energy_group_structure == "SHEM-361"
    np.testing.assert_array_equal(
        case.energy_bounds_ev,
        GROUP_STRUCTURES["SHEM-361"],
    )
    expected_source = np.diff(GROUP_STRUCTURES["SHEM-361"])
    expected_source /= expected_source.sum()
    np.testing.assert_allclose(case.source_probabilities, expected_source)


def test_case_constructor_uses_energy_groups_but_preserves_resolved_bounds():
    import inspect

    parameters = inspect.signature(Case).parameters
    assert "energy_groups" in parameters
    assert "energy_bounds_ev" not in parameters


def test_case_rejects_unknown_openmc_standard_name():
    with pytest.raises(ValueError, match="unknown OpenMC energy-group structure"):
        Case(
            name="unknown",
            materials=(material(),),
            energy_groups="NOT-A-GROUP-STRUCTURE",
            source_kind="uniform_energy",
            target_dimensions_cm=(1.0, 1.0, 1.0),
        )


@pytest.mark.parametrize(
    "bounds",
    [(), (1.0,), (1.0, np.nan), (2.0, 1.0), (1.0, 1.0)],
)
def test_case_rejects_invalid_custom_group_boundaries(bounds):
    with pytest.raises(ValueError, match="two edges|strictly ascending"):
        Case(
            name="bad_bounds",
            materials=(material(),),
            energy_groups=bounds,
            source_kind="uniform_energy",
            target_dimensions_cm=(1.0, 1.0, 1.0),
        )


@pytest.mark.parametrize("bounds", [(1.0, 1.0, 2.0), (1.0, np.nan, 2.0), (2.0, 1.0)])
def test_invalid_energy_boundaries_are_rejected(bounds):
    with pytest.raises(ValueError, match="ascending"):
        source_probabilities(bounds, "uniform_energy")


def test_uniform_source_integrates_nonuniform_groups():
    """Uniform in energy means probability is proportional to group width."""
    result = source_probabilities((1.0, 2.0, 5.0, 9.0), "uniform_energy")

    np.testing.assert_allclose(result, (0.125, 0.375, 0.5))


def test_watt_source_matches_seed_reference():
    """Quadrature must retain the moderated seed spectrum on every platform."""
    result = source_probabilities(energy_bounds("LANL30"), "watt")
    expected = np.array([
        2.6045440419823026e-11, 9.103278474112432e-11, 4.108747530663498e-10,
        1.824710754120624e-9, 8.195137499584237e-9, 3.667342471992377e-8,
        1.6423511732387714e-7, 7.370174419749594e-7, 3.302483520840071e-6,
        1.481566228312341e-5, 6.604507715533968e-5, 2.963545694175255e-4,
        1.3188820574533773e-3, 5.827594055444004e-3, 2.48078848655308e-2,
        3.29671695744622e-2, 6.3153266342927e-2, 1.1176445074503728e-1,
        1.7613857412338474e-1, 1.1143051468913984e-1, 1.1876944273399974e-1,
        1.149826486114795e-1, 9.872045748511424e-2, 1.1387359922545882e-1,
        1.872746210584329e-2, 5.853997003294495e-3, 1.0254536406664307e-3,
        1.8396980682371588e-4, 5.48501988928236e-5, 1.831646888576652e-5,
    ])

    np.testing.assert_allclose(result, expected, rtol=2e-8, atol=1e-18)
    assert sum(result) == pytest.approx(1.0, abs=2e-16)


def test_source_kind_is_validated():
    with pytest.raises(ValueError, match="source kind"):
        source_probabilities((1.0, 2.0), "line")

    with pytest.raises(ValueError, match="source kind"):
        Case(
            name="bad",
            materials=(material(),),
            energy_groups=(1.0, 2.0),
            source_kind="line",
            target_dimensions_cm=(1.0, 1.0, 1.0),
        )


def test_source_has_one_scientific_authority():
    uniform = Case(
        name="uniform",
        materials=(material(),),
        energy_groups=(1.0, 2.0, 5.0),
        source_kind="uniform_energy",
        target_dimensions_cm=(1.0, 1.0, 1.0),
    )

    np.testing.assert_allclose(uniform.source_probabilities, (0.25, 0.75))

    with pytest.raises(ValueError, match="only a grouped"):
        Case(
            name="duplicate", materials=(material(),), energy_groups=(1.0, 2.0),
            source_kind="uniform_energy", source_probabilities=(1.0,),
            target_dimensions_cm=(1.0, 1.0, 1.0),
        )


@pytest.mark.parametrize(
    "probabilities, message",
    [
        ((0.5,), "one finite"),
        ((-0.1, 1.1), "nonnegative"),
        ((0.2, 0.2), "sum to one"),
    ],
)
def test_grouped_source_validation(probabilities, message):
    with pytest.raises(ValueError, match=message):
        Case(
            name="bad_grouped",
            materials=(material(),),
            energy_groups=(1.0, 2.0, 3.0),
            source_kind="grouped",
            source_probabilities=probabilities,
            target_dimensions_cm=(1.0, 1.0, 1.0),
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"density_g_cm3": np.inf}, "density"),
        ({"temperature_k": np.nan}, "temperature"),
        ({"composition": ()}, "composition"),
        ({"composition": (("H1", -0.1), ("H2", 1.1))}, "nonnegative"),
        ({"composition": (("H1", 0.0), ("H2", 0.0))}, "not all zero"),
        ({"composition": (("H1", np.nan),)}, "finite"),
        ({"composition": (("H1", np.inf),)}, "finite"),
    ],
)
def test_material_scientific_validation(kwargs, message):
    values = dict(
        logical_name="x",
        name="x",
        density_g_cm3=1.0,
        composition=(("H1", 1.0),),
    )
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        Material(**values)


def test_explicit_nuclide_composition_is_preserved():
    """An explicit Be9 component must remain a named nuclide for OpenMC."""
    be9 = Material("be9", "Be-9", 1.85, (("Be9", 1.0),))

    assert be9.composition == (("Be9", 1.0),)


def test_natural_element_composition_is_simple():
    """A bare element symbol requests OpenMC's native natural expansion."""
    iron = Material("iron", "natural iron", 7.87, (("Fe", 1.0),))

    assert iron.composition == (("Fe", 1.0),)


def test_natural_element_mixture_uses_one_composition_representation():
    steel = Material(
        "steel",
        "simple Fe-C mixture",
        7.8,
        (("Fe", 0.98), ("C", 0.02)),
    )

    assert steel.composition == (("Fe", 0.98), ("C", 0.02))


def test_relative_atom_amounts_are_preserved_without_normalization():
    """Composition entries are OpenMC relative ao amounts, not a unit-sum vector."""
    amounts = (
        ("U234", 2.5759e-06),
        ("U235", 3.4428e-04),
        ("U238", 4.7441e-02),
    )

    flattop = Material("flattop_nu", "FlatTop_NU", 18.823124, amounts)

    assert flattop.composition == amounts
    assert sum(amount for _, amount in flattop.composition) != pytest.approx(1.0)


@pytest.mark.parametrize("identifier", ["iron", "Fe-56", "56Fe", "Fe!", "H0"])
def test_invalid_composition_identifiers_are_rejected(identifier):
    """Reject noncanonical spelling without embedding a periodic-table database."""
    with pytest.raises(ValueError, match="composition identifier"):
        Material("bad", "bad", 1.0, ((identifier, 1.0),))


def test_one_and_two_material_mapping_are_physical(one_case, two_case):
    """Physical roles, never tuple position, determine OpenMC geometry domains."""
    assert [x.logical_name for x in one_case.materials] == ["one"]
    assert one_case.geometry_type == "homogeneous"

    # Supply moderator first: roles, not tuple positions, still establish geometry.
    assert [x.role for x in two_case.materials] == ["moderator", "target"]
    assert two_case.geometry_type == "moderated_target"


def test_material_constructor_has_no_solver_ids():
    import inspect

    parameters = inspect.signature(Material).parameters
    assert "openmc_id" not in parameters
    assert "opensn_block" not in parameters


def test_case_exposes_only_genuine_opensn_convergence_controls():
    """Fixed verification mesh/quadrature choices are not misleading Case knobs."""
    import inspect

    parameters = inspect.signature(Case).parameters
    for removed in ("mesh_max_width_cm", "num_polar", "num_azimuthal"):
        assert removed not in parameters
    # This still governs the moments generated and retained by OpenMC MGXS.
    assert "scattering_order" in parameters


def test_geometry_and_solver_controls_are_validated():
    common = dict(
        name="case", materials=(material(),), energy_groups=(1.0, 2.0),
        source_kind="uniform_energy", target_dimensions_cm=(1.0, 1.0, 1.0),
    )
    for field, value, message in (
        ("target_dimensions_cm", (1.0, np.inf, 1.0), "target dimensions"),
        ("particles_per_batch", True, "particles_per_batch"),
        ("batches", 0, "batches"),
        ("scattering_order", -1, "scattering_order"),
        ("gmres_tolerance", np.nan, "gmres_tolerance"),
        ("gmres_max_iterations", 0, "gmres_max_iterations"),
        ("gmres_restart", 0, "gmres_restart"),
        ("keigen_tolerance", np.inf, "keigen_tolerance"),
        ("keigen_max_iterations", 0, "keigen_max_iterations"),
    ):
        with pytest.raises(ValueError, match=message):
            Case(**(common | {field: value}))


def test_case_run_modes_preserve_fixed_source_and_validate_eigenvalue():
    common = dict(
        name="mode",
        materials=(material(),),
        energy_groups=(1.0, 2.0),
        target_dimensions_cm=(1.0, 1.0, 1.0),
    )
    fixed = Case(**common, source_kind="uniform_energy")
    eigenvalue = Case(
        **common,
        run_mode="eigenvalue",
        inactive_batches=2,
        batches=5,
    )

    assert fixed.run_mode == "fixed_source"
    assert fixed.source_definition == {"kind": "uniform_energy"}
    assert eigenvalue.run_mode == "eigenvalue"
    assert eigenvalue.source_kind is None
    assert eigenvalue.source_definition is None
    with pytest.raises(ValueError, match="no external source"):
        _ = eigenvalue.source_probabilities

    with pytest.raises(ValueError, match="require source kind"):
        Case(**common)
    with pytest.raises(ValueError, match="do not accept source_kind"):
        Case(**common, run_mode="eigenvalue", source_kind="uniform_energy")
    with pytest.raises(ValueError, match="inactive_batches=0"):
        Case(**common, source_kind="uniform_energy", inactive_batches=1)
    with pytest.raises(ValueError, match="less than batches"):
        Case(**common, run_mode="eigenvalue", batches=3, inactive_batches=3)
    with pytest.raises(ValueError, match="inactive_batches"):
        Case(**common, run_mode="eigenvalue", inactive_batches=True)
    with pytest.raises(ValueError, match="one homogeneous material"):
        Case(
            name="multi_eigenvalue",
            materials=(
                material("target", "target"),
                material("moderator", "moderator"),
            ),
            energy_groups=(1.0, 2.0),
            target_dimensions_cm=(1.0, 1.0, 1.0),
            outer_dimensions_cm=(2.0, 2.0, 2.0),
            run_mode="eigenvalue",
        )


def test_moderated_outer_dimensions_strictly_enclose_target():
    with pytest.raises(ValueError, match="strictly exceed"):
        Case(
            name="touching",
            materials=(
                material("target", "target"),
                material("mod", "moderator"),
            ),
            energy_groups=(1.0, 2.0),
            source_kind="uniform_energy",
            target_dimensions_cm=(1.0, 1.0, 1.0),
            outer_dimensions_cm=(2.0, 1.0, 2.0),
        )


def test_history_semantics_match_reference_case_definitions():
    from examples.be9.case import CASE as be9
    from examples.moderated.case import CASE as moderated

    assert (be9.batches, be9.particles_per_batch, be9.total_histories) == (
        40,
        25_000,
        1_000_000,
    )
    assert (moderated.batches, moderated.particles_per_batch, moderated.total_histories) == (
        40,
        250_000,
        10_000_000,
    )


def test_source_volume_is_target_volume(two_case):
    assert two_case.source_volume_cm3 == pytest.approx(0.064)
