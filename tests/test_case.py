from pathlib import Path

import numpy as np
import pytest

from generate_mgxs import Case, Material, energy_bounds, prepare, source_probabilities
from conftest import material, tiny_case


def test_named_group_structures_are_ascending():
    """All public scientific arrays use increasing physical energy."""
    assert len(energy_bounds("WIMS69")) == 70
    assert len(energy_bounds("LANL30")) == 31
    assert np.all(np.diff(energy_bounds("WIMS69")) > 0.0)


def test_unknown_group_structure_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        energy_bounds("other")


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
            energy_bounds_ev=(1.0, 2.0),
            source_kind="line",
            target_dimensions_cm=(1.0, 1.0, 1.0),
        )


def test_source_has_one_scientific_authority():
    uniform = Case(
        name="uniform",
        materials=(material(),),
        energy_bounds_ev=(1.0, 2.0, 5.0),
        source_kind="uniform_energy",
        target_dimensions_cm=(1.0, 1.0, 1.0),
    )

    np.testing.assert_allclose(uniform.source_probabilities, (0.25, 0.75))

    with pytest.raises(ValueError, match="only a grouped"):
        Case(
            name="duplicate", materials=(material(),), energy_bounds_ev=(1.0, 2.0),
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
            energy_bounds_ev=(1.0, 2.0, 3.0),
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
        ({"composition": (("H1", 0.5),)}, "sum to one"),
        ({"composition": (("H1", -0.1), ("H2", 1.1))}, "nonnegative"),
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


@pytest.mark.parametrize("identifier", ["iron", "Fe-56", "56Fe", "Fe!", "H0"])
def test_invalid_composition_identifiers_are_rejected(identifier):
    """Reject noncanonical spelling without embedding a periodic-table database."""
    with pytest.raises(ValueError, match="composition identifier"):
        Material("bad", "bad", 1.0, ((identifier, 1.0),))


def test_one_and_two_material_mapping_are_physical(one_case, two_case):
    """Physical roles, never tuple position, determine OpenSn blocks."""
    assert [(x.logical_name, x.opensn_block) for x in one_case.materials] == [("one", 0)]
    assert one_case.geometry_type == "homogeneous"

    # Supply moderator first: roles, not tuple positions, still establish blocks.
    assert [x.role for x in two_case.materials] == ["moderator", "target"]
    assert {x.logical_name: x.opensn_block for x in two_case.materials} == {
        "moderator": 1, "target": 0,
    }


def test_material_constructor_has_no_solver_ids():
    import inspect

    parameters = inspect.signature(Material).parameters
    assert "openmc_id" not in parameters
    assert "opensn_block" not in parameters


def test_geometry_and_solver_controls_are_validated():
    common = dict(
        name="case", materials=(material(),), energy_bounds_ev=(1.0, 2.0),
        source_kind="uniform_energy", target_dimensions_cm=(1.0, 1.0, 1.0),
    )
    for field, value, message in (
        ("target_dimensions_cm", (1.0, np.inf, 1.0), "target dimensions"),
        ("mesh_max_width_cm", (1.0, 0.0, 1.0), "mesh widths"),
        ("particles_per_batch", True, "particles_per_batch"),
        ("batches", 0, "batches"),
        ("num_polar", 0, "num_polar"),
        ("num_azimuthal", True, "num_azimuthal"),
        ("scattering_order", -1, "scattering_order"),
        ("gmres_tolerance", np.nan, "gmres_tolerance"),
        ("gmres_max_iterations", 0, "gmres_max_iterations"),
        ("gmres_restart", 0, "gmres_restart"),
    ):
        with pytest.raises(ValueError, match=message):
            Case(**(common | {field: value}))


def test_moderated_outer_dimensions_strictly_enclose_target():
    with pytest.raises(ValueError, match="strictly exceed"):
        Case(
            name="touching",
            materials=(
                material("target", "target"),
                material("mod", "moderator"),
            ),
            energy_bounds_ev=(1.0, 2.0),
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
