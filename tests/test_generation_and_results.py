import ast
from importlib import resources
import json
from pathlib import Path

import numpy as np
import pytest

from generate_mgxs import (
    Case,
    Material,
    Spectrum,
    load_openmc_domain_spectra,
    load_openmc_result,
    load_opensn_result,
    prepare,
)
from conftest import EVIDENCE, material


def test_generation_is_deterministic(one_case, tmp_path):
    """The same scientific case must produce byte-identical independent inputs."""
    first = prepare(one_case, tmp_path / "first")
    second = prepare(one_case, tmp_path / "second")

    assert (first / "openmc/model.py").read_text() == (
        second / "openmc/model.py"
    ).read_text()
    assert (first / "opensn/input.py").read_text() == (
        second / "opensn/input.py"
    ).read_text()


def test_solver_templates_are_readable_package_data():
    """Both plain templates must be available through installed package resources."""
    template_root = resources.files("generate_mgxs").joinpath("templates")

    openmc_template = template_root.joinpath("openmc_model.py.template").read_text()
    opensn_template = template_root.joinpath("opensn_input.py.template").read_text()

    assert "def build_model" in openmc_template
    assert "material.add_element" in openmc_template
    assert "SteadyStateSourceSolver" in opensn_template
    assert "SOURCE_PROBABILITIES_ASCENDING[::-1]" in opensn_template


def test_openmc_input_is_scientifically_readable(one_case, tmp_path):
    text = (prepare(one_case, tmp_path / "run") / "openmc/model.py").read_text()

    for fact in (
        "H1",
        "density_g_cm3",
        "temperature_k",
        "target_dimensions_cm",
        "PHYSICAL_SOURCE",
        "ENERGY_BOUNDS_EV",
        "particles_per_batch",
        "batches",
        "scattering_order",
    ):
        assert fact in text

    assert "openmc.mgxs.Library" in text
    assert text.index("OPENMC_HISTORY_SETTINGS") < text.index("ENERGY_BOUNDS_EV =")


def test_generated_openmc_uses_exact_solver_mgxs_contract(one_case, tmp_path):
    """Only fields consumed by canonical loading and OpenSn belong in mgxs.h5."""
    text = (prepare(one_case, tmp_path / "run") / "openmc/model.py").read_text()
    mgxs_types = _literal_assignment(text, "MGXS_TYPES")

    assert mgxs_types == [
        "total",
        "absorption",
        "consistent scatter matrix",
        "consistent nu-scatter matrix",
        "fission",
        "nu-fission",
        "chi",
    ]
    assert 'library.scatter_format = "legendre"' in text
    assert 'library.legendre_order = CONFIG["scattering_order"]' in text
    assert "library.correction = None" in text

    # Exact element checks avoid confusing the consistent names with the old
    # unqualified scatter/nu-scatter MGXS variants.
    assert not {
        "capture",
        "chi-prompt",
        "reduced absorption",
        "scatter matrix",
        "nu-scatter matrix",
        "multiplicity matrix",
        "nu-fission matrix",
    }.intersection(mgxs_types)


def test_named_openmc_groups_remain_named_in_generated_model(tmp_path):
    """SHEM boundaries are derived from one native OpenMC groups object."""
    from openmc.mgxs import GROUP_STRUCTURES

    case = Case(
        name="shem",
        materials=(material(),),
        energy_groups="SHEM-361",
        source_kind="uniform_energy",
        target_dimensions_cm=(1.0, 1.0, 1.0),
    )
    run = prepare(case, tmp_path / "shem")
    model_text = (run / "openmc/model.py").read_text()
    opensn_text = (run / "opensn/input.py").read_text()
    metadata = json.loads((run / "_metadata/run.json").read_text())

    assert 'ENERGY_GROUP_STRUCTURE = "SHEM-361"' in model_text
    assert 'EnergyGroups(group_edges="SHEM-361")' in model_text
    assert "ENERGY_BOUNDS_EV = groups.group_edges" in model_text
    assert "ENERGY_BOUNDS_EV = np.asarray(" not in model_text
    assert "np.diff(ENERGY_BOUNDS_EV)" in model_text
    assert "library.energy_groups = groups" in model_text
    assert '"energy_bounds": ENERGY_BOUNDS_EV.tolist()' in model_text
    assert metadata["case"]["energy_group_structure"] == "SHEM-361"
    np.testing.assert_array_equal(
        metadata["case"]["energy_bounds_ev"],
        GROUP_STRUCTURES["SHEM-361"],
    )

    # OpenSn has no named-group registry, so it receives the resolved values
    # derived from OpenMC during preparation.
    np.testing.assert_array_equal(
        _array_literal_assignment(
            opensn_text,
            "ENERGY_BOUNDS_EV_ASCENDING",
        ),
        GROUP_STRUCTURES["SHEM-361"],
    )


def test_custom_groups_remain_explicit_in_generated_model(one_case, tmp_path):
    """Custom edges stay visible and feed the same OpenMC groups object."""
    model_text = (
        prepare(one_case, tmp_path / "custom") / "openmc/model.py"
    ).read_text()

    assert "ENERGY_GROUP_STRUCTURE = None" in model_text
    assert "ENERGY_BOUNDS_EV = np.asarray(" in model_text
    assert "EnergyGroups(group_edges=ENERGY_BOUNDS_EV)" in model_text
    generated = _array_literal_assignment(model_text, "ENERGY_BOUNDS_EV")
    np.testing.assert_array_equal(generated, one_case.energy_bounds_ev)


def test_opensn_input_is_scientifically_readable(one_case, tmp_path):
    text = (prepare(one_case, tmp_path / "run") / "opensn/input.py").read_text()

    for fact in (
        "MGXS_HDF5",
        "logical_name",
        "cube_side_cm",
        "cells_per_axis",
        "cell_count",
        "volume_cm3",
        "n_polar",
        "n_azimuthal",
        "angular_directions",
        "scattering_order",
        "gmres_tolerance",
        "gmres_restart",
    ):
        assert fact in text

    assert "LoadFromOpenMC" in text
    assert "SOURCE_PROBABILITIES_ASCENDING[::-1]" in text
    assert '/ VERIFICATION["volume_cm3"]' in text
    assert 'source_strength_high_to_low.sum() * VERIFICATION["volume_cm3"]' in text
    assert "flux_high_to_low[::-1]" in text
    assert text.index("OPENSN_NUMERICAL_SETTINGS") < text.index("ENERGY_BOUNDS_EV_ASCENDING =")


def _literal_assignment(text, name):
    """Read a generated top-level scientific literal without executing solver code."""
    tree = ast.parse(text)
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value)

    raise AssertionError(f"missing generated assignment {name}")


def _array_literal_assignment(text, name):
    """Read the literal passed to a generated top-level np.asarray call."""
    tree = ast.parse(text)
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value.args[0])

    raise AssertionError(f"missing generated array assignment {name}")


def test_openmc_and_opensn_sources_share_one_generated_authority(tmp_path):
    """Both solver inputs must serialize the same Watt parameters, never a copied vector."""
    bounds = (1.0e-5, 1.0e6, 2.0e7)
    case = Case(
        name="watt",
        materials=(material(),),
        energy_groups=bounds,
        source_kind="watt",
        watt_a_mev=0.988,
        watt_b_per_mev=2.249,
        target_dimensions_cm=(1.0, 1.0, 1.0),
    )

    run = prepare(case, tmp_path / "watt")
    texts = [
        (run / relative).read_text()
        for relative in ("openmc/model.py", "opensn/input.py")
    ]

    for text in texts:
        assert _literal_assignment(text, "PHYSICAL_SOURCE") == case.source_definition
        assert "SOURCE_PROBABILITIES_ASCENDING = source_probabilities()" in text
        # Derived probabilities are code, not a second serialized scientific input.
        assert repr(case.source_probabilities) not in text


def test_generated_openmc_marks_nuclides_and_natural_elements(tmp_path):
    """Generated materials must visibly delegate natural expansion to OpenMC."""
    material_definition = Material(
        "steel",
        "simple steel",
        7.8,
        (("Fe", 0.98), ("C12", 0.02)),
    )
    case = Case(
        name="steel",
        materials=(material_definition,),
        energy_groups=(1.0, 2.0),
        source_kind="uniform_energy",
        target_dimensions_cm=(1.0, 1.0, 1.0),
    )

    text = (prepare(case, tmp_path / "steel") / "openmc/model.py").read_text()
    materials = _literal_assignment(text, "MATERIALS")
    components = materials[0]["composition"]

    assert components[0] == {
        "identifier": "Fe",
        "kind": "element",
        "atom_amount": 0.98,
    }
    assert components[1] == {
        "identifier": "C12",
        "kind": "nuclide",
        "atom_amount": 0.02,
    }
    assert "material.add_element" in text
    assert "material.add_nuclide" in text


def test_flattop_eigenvalue_generation_preserves_its_scientific_definition(
    tmp_path,
):
    """FlatTop generation retains notebook inputs without inventing a source."""
    import runpy

    from examples.flattop.case import CASE

    run = prepare(CASE, tmp_path / "flattop")
    model_path = run / "openmc/model.py"
    model_text = model_path.read_text()
    opensn_text = (run / "opensn/input.py").read_text()
    metadata = json.loads((run / "_metadata/run.json").read_text())["case"]
    materials = _literal_assignment(model_text, "MATERIALS")
    history = _literal_assignment(model_text, "OPENMC_HISTORY_SETTINGS")
    geometry = _literal_assignment(model_text, "GEOMETRY")

    expected_amounts = [2.5759e-06, 3.4428e-04, 4.7441e-02]
    assert [item["atom_amount"] for item in materials[0]["composition"]] == (
        expected_amounts
    )
    assert [item["atom_amount"] for item in metadata["materials"][0]["composition"]] == (
        expected_amounts
    )
    assert "source" not in metadata
    assert "source_probabilities" not in metadata
    assert history == {
        "run_mode": "eigenvalue",
        "particles_per_batch": 20_000,
        "batches": 520,
        "inactive_batches": 120,
        "total_histories": 10_400_000,
    }
    assert geometry["target_dimensions_cm"] == (2.0, 2.0, 2.0)
    assert set(geometry["boundaries"].values()) == {"reflective"}
    assert _literal_assignment(model_text, "PHYSICAL_SOURCE") is None
    assert 'constraints={"fissionable": True}' in model_text
    assert "only_fissionable" not in model_text
    assert 'PowerIterationKEigenSolver' in opensn_text
    assert '"scattering_order": 0' in opensn_text
    assert '"angular_directions": 8' in opensn_text

    # Constructing the generated native model is a fast smoke test of the
    # actual OpenMC Settings API; no particle transport is started.
    generated = runpy.run_path(model_path)
    settings = generated["MODEL"].settings
    library = generated["MGXS_LIBRARY"]
    initial_source = settings.source[0]
    assert settings.run_mode == "eigenvalue"
    assert (settings.batches, settings.inactive, settings.particles) == (
        520,
        120,
        20_000,
    )
    assert settings.temperature["method"] == "interpolation"
    assert initial_source.constraints["fissionable"] is True
    assert initial_source.energy is None
    assert library.scatter_format == "legendre"
    assert library.legendre_order == 7
    assert library.correction is None


def test_moderated_example_uses_lanl70_box_inside_box_geometry(tmp_path):
    """The UO2 target is an inner box surrounded by the outer HDPE box."""
    import runpy

    from examples.moderated.case import CASE

    run = prepare(CASE, tmp_path / "moderated", solvers=("openmc",))
    generated = runpy.run_path(run / "openmc/model.py")
    model = generated["MODEL"]

    assert generated["GEOMETRY"]["type"] == "moderated_target"
    assert generated["ENERGY_BOUNDS_EV"].size == 71

    cells = list(model.geometry.get_all_cells().values())
    assert {cell.fill.name for cell in cells} == {"UO2", "HDPE moderator"}

    surfaces = list(model.geometry.get_all_surfaces().values())
    for surface_type, coordinate in (
        ("XPlane", "x0"),
        ("YPlane", "y0"),
        ("ZPlane", "z0"),
    ):
        planes = sorted(
            (
                (getattr(surface, coordinate), surface.boundary_type)
                for surface in surfaces
                if surface.__class__.__name__ == surface_type
            ),
            key=lambda item: item[0],
        )
        assert planes == [
            (-0.75, "reflective"),
            (-0.2, "transmission"),
            (0.2, "transmission"),
            (0.75, "reflective"),
        ]


def test_generated_inputs_are_run_relative_and_have_no_ascii_handoff(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "run")
    text = (
        (run / "openmc/model.py").read_text()
        + (run / "opensn/input.py").read_text()
    )

    assert "/home/" not in text
    assert ".cxs" not in text.lower()
    assert "MGXS_HDF5 = RUN_DIRECTORY / \"openmc\" / \"mgxs.h5\"" in text


def test_two_material_domains_are_verified_independently(two_case, tmp_path):
    """Input order cannot couple target and moderator in the verification solve."""
    text = (prepare(two_case, tmp_path / "run") / "opensn/input.py").read_text()
    materials = _literal_assignment(text, "MATERIALS")

    assert [item["logical_name"] for item in materials] == ["moderator", "target"]
    assert "opensn_block" not in text
    assert "grid.SetUniformBlockID(0)" in text
    assert 'xs_map=[{"block_ids": [0], "xs": cross_sections}]' in text
    assert 'for material in CONFIG["materials"]' in text
    assert 'str(MGXS_HDF5), logical_name, material["temperature_k"]' in text


def test_fixed_opensn_verifier_is_independent_of_case_size_and_group_count(
    tmp_path
):
    """OpenSn cost is fixed at eight cells/eight directions, even for SHEM-361."""
    from examples.be9.case import CASE as be9

    hdpe = Material(
        "hdpe",
        "HDPE",
        0.955,
        (("H1", 0.667), ("C12", 0.333)),
        thermal_scattering=("c_H_in_CH2",),
    )
    hdpe_shem_like = Case(
        name="hdpe_shem_like",
        materials=(hdpe,),
        energy_groups=tuple(np.linspace(0.0, 2.0e7, 362)),
        source_kind="uniform_energy",
        target_dimensions_cm=(20.0, 30.0, 400.0),
        scattering_order=3,
    )

    for case in (be9, hdpe_shem_like):
        text = (prepare(case, tmp_path / case.name) / "opensn/input.py").read_text()
        verification = _literal_assignment(text, "VERIFICATION")
        boundaries = _literal_assignment(text, "REFLECTING_BOUNDARIES")

        assert verification == {
            "cube_side_cm": 2.0,
            "cells_per_axis": 2,
            "cell_count": 8,
            "volume_cm3": 8.0,
            "n_polar": 2,
            "n_azimuthal": 4,
            "angular_directions": 8,
            "scattering_order": 0,
        }
        assert len(boundaries) == 6
        assert {item["type"] for item in boundaries} == {"reflecting"}
        assert "target_dimensions_cm" not in text
        assert "outer_dimensions_cm" not in text
        assert "mesh_max_width_cm" not in text

    shem_text = (
        tmp_path / hdpe_shem_like.name / "opensn" / "input.py"
    ).read_text()
    generated_bounds = _array_literal_assignment(
        shem_text, "ENERGY_BOUNDS_EV_ASCENDING"
    )
    assert len(generated_bounds) - 1 == 361
    assert generated_bounds[0] == 0.0


def test_prepare_writes_only_nonempty_scientific_directories(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "run")
    generated_files = {
        path.relative_to(run).as_posix()
        for path in run.rglob("*")
        if path.is_file()
    }

    assert generated_files == {
        "_metadata/run.json", "openmc/model.py", "opensn/input.py",
    }

    (run / "notes.md").write_text("my notes")
    prepare(one_case, run)

    assert (run / "notes.md").read_text() == "my notes"


def test_prepare_openmc_only_writes_no_opensn_input(one_case, tmp_path):
    """OpenMC-only generation must not leave an empty or misleading OpenSn tree."""
    run_path = prepare(one_case, tmp_path / "run", solvers=("openmc",))
    generated_files = {
        path.relative_to(run_path).as_posix()
        for path in run_path.rglob("*")
        if path.is_file()
    }
    metadata = json.loads((run_path / "_metadata/run.json").read_text())

    assert generated_files == {"_metadata/run.json", "openmc/model.py"}
    assert not (run_path / "opensn").exists()
    assert metadata["solvers"] == ["openmc"]
    assert [item["path"] for item in metadata["artifacts"]] == ["openmc/model.py"]


@pytest.mark.parametrize(
    ("solvers", "message"),
    [
        ((), "at least one"),
        (("unknown",), "unknown solver"),
        (("opensn",), "requires OpenMC"),
        (("openmc", "openmc"), "duplicated"),
        (("openmc", 1), "must be strings"),
        ("openmc", "iterable of solver names"),
    ],
)
def test_prepare_rejects_invalid_solver_selections(one_case, tmp_path, solvers, message):
    with pytest.raises(ValueError, match=message):
        prepare(one_case, tmp_path / "run", solvers=solvers)


def test_minimal_metadata_has_case_and_input_hashes(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "run")
    metadata = json.loads((run / "_metadata/run.json").read_text())

    assert metadata["case"]["name"] == one_case.name
    assert metadata["case"]["source_volume_cm3"] == 1.0
    assert metadata["case"]["particles_per_batch"] == 10
    assert metadata["case"]["total_histories"] == 20
    assert metadata["solvers"] == ["openmc", "opensn"]
    assert {item["path"] for item in metadata["artifacts"]} == {
        "openmc/model.py",
        "opensn/input.py",
    }


def test_load_openmc_seed_preserves_standard_deviation():
    """Statepoint processing must retain flux uncertainty and ascending groups."""
    result = load_openmc_result(EVIDENCE / "be9_openmc_result.json")
    assert result.values.shape == result.std_dev.shape == (69,)
    assert np.all(result.std_dev > 0.0)
    assert np.all(np.diff(result.energy_bounds_ev) > 0.0)


def test_load_all_openmc_domains_in_declared_order_without_material_aliasing(
    tmp_path,
):
    """Run metadata restores declaration order for pre-domain_order results."""
    run = tmp_path / "run"
    result_path = run / "openmc" / "openmc_result.json"
    metadata_path = run / "_metadata" / "run.json"
    result_path.parent.mkdir(parents=True)
    metadata_path.parent.mkdir(parents=True)

    records = {
        "alu": {"flux": [4.0, 1.0], "std_dev": [0.4, 0.1]},
        "cad": {"flux": [3.0, 2.0], "std_dev": [0.3, 0.2]},
        "hdpe": {"flux": [2.0, 3.0], "std_dev": [0.2, 0.3]},
        "he3": {"flux": [1.0, 4.0], "std_dev": [0.1, 0.4]},
        "outer": {"flux": [5.0, 0.5], "std_dev": [0.5, 0.05]},
    }
    result_path.write_text(json.dumps({
        "energy_bounds": [1.0, 2.0, 5.0],
        "flux": records["he3"]["flux"],
        "std_dev": records["he3"]["std_dev"],
        "logical_domain": "he3",
        "domains": records,
    }, sort_keys=True))
    metadata_path.write_text(json.dumps({
        "case": {
            "mgxs_domains": [
                {
                    "xsdata_name": name,
                    "material_logical_name": material,
                }
                for name, material in (
                    ("he3", "he3_material"),
                    ("hdpe", "hdpe_material"),
                    ("cad", "cadmium_material"),
                    ("alu", "aluminum_material"),
                    ("outer", "hdpe_material"),
                )
            ]
        }
    }))

    domains = load_openmc_domain_spectra(run)

    assert list(domains) == ["he3", "hdpe", "cad", "alu", "outer"]
    assert domains["hdpe"] is not domains["outer"]
    for name, spectrum in domains.items():
        assert spectrum.logical_domain == name
        np.testing.assert_array_equal(spectrum.energy_bounds_ev, [1.0, 2.0, 5.0])
        np.testing.assert_array_equal(spectrum.values, records[name]["flux"])
        np.testing.assert_array_equal(spectrum.std_dev, records[name]["std_dev"])

    # The original primary-only API remains unchanged.
    primary = load_openmc_result(run)
    assert isinstance(primary, Spectrum)
    assert primary.logical_domain == "he3"
    np.testing.assert_array_equal(primary.values, records["he3"]["flux"])


def test_generated_openmc_result_records_domain_order(one_case, tmp_path):
    model = (prepare(one_case, tmp_path / "run") / "openmc/model.py").read_text()

    assert '"domain_order": [item["xsdata_name"] for item in MGXS_DOMAINS]' in model


def test_load_openmc_eigenvalue_result_preserves_keff_and_raw_flux(tmp_path):
    """Eigenvalue loading adds k data without normalizing the raw OpenMC tally."""
    path = tmp_path / "openmc_result.json"
    path.write_text(json.dumps({
        "run_mode": "eigenvalue",
        "energy_bounds": [1.0, 2.0, 5.0],
        "flux": [12.0, 3.0],
        "std_dev": [0.6, 0.2],
        "logical_domain": "fuel",
        "k_eff": 1.01234,
        "k_eff_std_dev": 0.00021,
    }))

    result = load_openmc_result(path)

    assert result.k_eff == pytest.approx(1.01234)
    assert result.k_eff_std_dev == pytest.approx(0.00021)
    np.testing.assert_array_equal(result.spectrum.values, [12.0, 3.0])
    np.testing.assert_array_equal(result.spectrum.std_dev, [0.6, 0.2])
    np.testing.assert_allclose(result.spectrum.normalized, [0.8, 0.2])


def test_load_converged_opensn_seed():
    """A stored OpenSn result remains usable only with explicit convergence evidence."""
    result = load_opensn_result(EVIDENCE / "be9_opensn_result.json")
    assert result.converged
    assert result.iterations == 578
    assert result.residual == pytest.approx(9.239371e-11)
    assert result.spectrum.values.sum() == pytest.approx(1578.8194676115886)


def test_load_opensn_rejects_unknown_convergence(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "energy_bounds": [1.0, 2.0], "flux": [1.0],
        "solver": {"converged": None, "iterations": None, "residual": None},
    }))
    with pytest.raises(RuntimeError, match="explicitly"):
        load_opensn_result(path)


def test_be9_normalized_l1_regression():
    """All three solvers retain the accepted normalized Be-9 spectra."""
    openmc = load_openmc_result(EVIDENCE / "be9_openmc_result.json")
    opensn = load_opensn_result(EVIDENCE / "be9_opensn_result.json").spectrum
    direct_data = json.loads((EVIDENCE / "be9_direct_result.json").read_text())
    direct = Spectrum(
        np.asarray(direct_data["energy_bounds"]),
        np.asarray(direct_data["flux"]),
    )

    openmc_l1 = np.linalg.norm(openmc.normalized - direct.normalized, 1)
    opensn_l1 = np.linalg.norm(opensn.normalized - direct.normalized, 1)

    assert openmc_l1 == pytest.approx(1.5125390210620065e-4, rel=2e-12)
    assert opensn_l1 == pytest.approx(6.345963571253652e-11, rel=2e-6)


def test_spectrum_rejects_wrong_boundary_count():
    with pytest.raises(ValueError, match=r"G \+ 1"):
        Spectrum(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_examples_show_the_complete_workflows():
    root = Path(__file__).parents[1]
    be9 = (
        (root / "examples/be9/case.py").read_text()
        + (root / "examples/be9/run.py").read_text()
    )
    moderated = (
        (root / "examples/moderated/case.py").read_text()
        + (root / "examples/moderated/run.py").read_text()
    )
    flattop = (
        (root / "examples/flattop/case.py").read_text()
        + (root / "examples/flattop/run.py").read_text()
    )

    for name in ("prepare", "run_openmc", "load_mgxs", "solve_infinite_medium", "run_opensn"):
        assert name in be9
    for name in ("uo2_target", "hdpe_moderator", "run_openmc", "run_opensn"):
        assert name in moderated
    assert "/home/ragusa/" not in be9 + moderated
    assert "OPENMC_CROSS_SECTIONS" in be9 + moderated
    assert "OPENSN_CONSOLE" in be9 + moderated
    assert "run_path = prepare" in be9 + moderated
    assert "plot_mgxs" in be9 + moderated
    for name in (
        "run_mode=\"eigenvalue\"",
        "inactive_batches=120",
        "solve_infinite_medium_eigenvalue",
        "OpenMC k_eff",
        "Direct  k_eff",
        "OpenSn  k_eff",
    ):
        assert name in flattop
    assert "/home/ragusa/" not in flattop


@pytest.mark.parametrize(
    "example",
    ("be9", "hdpe", "flattop", "detector", "pu9_hdpe"),
)
def test_per_case_example_runner_uses_its_run_directory_directly(example):
    runner = Path(__file__).parents[1] / "examples" / example / "run.py"
    tree = ast.parse(runner.read_text())
    prepare_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "prepare"
    ]

    assert len(prepare_calls) == 1
    directory = prepare_calls[0].args[1]
    assert isinstance(directory, ast.Call)
    assert isinstance(directory.func, ast.Name)
    assert directory.func.id == "Path"
    assert len(directory.args) == 1
    assert isinstance(directory.args[0], ast.Constant)
    assert directory.args[0].value == "run"


def test_prepare_does_not_execute_subprocesses(one_case, tmp_path, monkeypatch):
    """Preparation is generation only, which is essential for external scheduling."""
    import subprocess

    def unexpected(*args, **kwargs):
        raise AssertionError("prepare() attempted subprocess execution")

    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)

    run = prepare(one_case, tmp_path / "prepared")

    assert (run / "openmc/model.py").is_file()
    assert (run / "opensn/input.py").is_file()


def test_multiple_cases_prepare_as_independent_directories(tmp_path, monkeypatch):
    """A bulk-style loop needs no shared campaign object or mutable central state."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("solver executed during preparation")
        ),
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("solver executed during preparation")
        ),
    )
    cases = [
        Case(
            name=f"material_{index:03d}", materials=(material(f"domain_{index}"),),
            energy_groups=(1.0, 2.0, 4.0 + index), source_kind="uniform_energy",
            target_dimensions_cm=(float(index), 1.0, 1.0),
        )
        for index in (1, 2, 3)
    ]
    runs = [
        prepare(case, tmp_path / case.name, solvers=("openmc",))
        for case in cases
    ]

    assert len({run.resolve() for run in runs}) == len(cases)
    for case, run in zip(cases, runs):
        openmc_text = (run / "openmc/model.py").read_text()

        assert repr(case.name) in openmc_text
        assert not (run / "opensn").exists()
        assert (run / "_metadata/run.json").is_file()
        # A later shell/SLURM job receives a complete standalone Python model,
        # not a reference to the in-memory Case used during this loop.
        compile(openmc_text, str(run / "openmc/model.py"), "exec")

    assert len({(run / "openmc/model.py").read_text() for run in runs}) == len(cases)
