import runpy

import pytest

from generate_mgxs import (
    Case,
    ConcentricCell,
    ConcentricGeometry,
    Material,
    prepare,
)


def _material(logical_name):
    return Material(
        logical_name=logical_name,
        name=logical_name,
        density_g_cm3=1.0,
        composition=(("H1", 1.0),),
    )


def _geometry(*, radii=(1.0, 2.0), outer_region=None, half_widths=None):
    inner = _material("inner_material")
    shell = _material("shell_material")
    regions = tuple(
        ConcentricCell(name, material, xsdata_name, radius)
        for name, material, xsdata_name, radius in zip(
            ("Inner", "Shell"),
            (inner, shell),
            ("inner", "shell"),
            radii,
        )
    )
    return ConcentricGeometry(
        regions=regions,
        height_cm=4.0,
        axial_boundaries=("reflective", "reflective"),
        outer_radial_boundary=None if outer_region else "reflective",
        outer_region=outer_region,
        outer_half_widths_cm=half_widths,
        outer_xy_boundaries=("reflective",) * 4 if outer_region else None,
    )


def _case(geometry, *, run_mode="fixed_source"):
    materials = []
    for region in geometry.regions + ((geometry.outer_region,) if geometry.outer_region else ()):
        if region.material not in materials:
            materials.append(region.material)
    values = dict(
        name="concentric",
        materials=materials,
        energy_groups=(0.0, 1.0e6, 20.0e6),
        geometry=geometry,
        source_bounds_cm=(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
        particles_per_batch=10,
        batches=3,
        scattering_order=0,
    )
    if run_mode == "fixed_source":
        values["source_kind"] = "uniform_energy"
    else:
        values.update(run_mode="eigenvalue", inactive_batches=1)
    return Case(**values)


def _halfspaces(region):
    if hasattr(region, "surface"):
        return [(region.surface, region.side)]
    return [item for child in region for item in _halfspaces(child)]


@pytest.fixture(scope="module")
def statepoint_selector(tmp_path_factory):
    case = Case(
        name="statepoint_selection",
        materials=(_material("statepoint_material"),),
        energy_groups=(1.0, 2.0),
        source_kind="uniform_energy",
        target_dimensions_cm=(1.0, 1.0, 1.0),
    )
    run = prepare(
        case,
        tmp_path_factory.mktemp("statepoint-model"),
        solvers=("openmc",),
    )
    generated = runpy.run_path(run / "openmc/model.py")
    return generated["_select_statepoint"]


def _use_statepoint_directory(selector, directory):
    selector.__globals__["OPENMC_DIRECTORY"] = directory
    return selector


def test_concentric_geometry_validates_only_the_required_shape():
    inner = _material("inner")
    outer = ConcentricCell("Outer", inner, "outer")

    with pytest.raises(ValueError, match="positive"):
        ConcentricCell("Bad", inner, "bad", 0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        _geometry(radii=(2.0, 1.0))
    with pytest.raises(ValueError, match="height"):
        ConcentricGeometry(
            regions=(ConcentricCell("Inner", inner, "inner", 1.0),),
            height_cm=0.0,
            outer_radial_boundary="reflective",
        )
    with pytest.raises(ValueError, match="duplicate.*XS"):
        ConcentricGeometry(
            regions=(
                ConcentricCell("A", inner, "same", 1.0),
                ConcentricCell("B", inner, "same", 2.0),
            ),
            height_cm=2.0,
            outer_radial_boundary="reflective",
        )
    with pytest.raises(ValueError, match="half-width"):
        ConcentricGeometry(
            regions=(ConcentricCell("Inner", inner, "inner", 1.0),),
            height_cm=2.0,
            outer_region=outer,
            outer_half_widths_cm=(1.0, 2.0),
        )
    with pytest.raises(ValueError, match="together"):
        ConcentricGeometry(
            regions=(ConcentricCell("Inner", inner, "inner", 1.0),),
            height_cm=2.0,
            outer_region=outer,
        )
    with pytest.raises(ValueError, match="must be used by the geometry"):
        Case(
            name="mismatch",
            materials=(inner, _material("unused")),
            energy_groups=(1.0, 2.0),
            source_kind="uniform_energy",
            geometry=ConcentricGeometry(
                regions=(ConcentricCell("Inner", inner, "inner", 1.0),),
                height_cm=2.0,
                outer_radial_boundary="reflective",
            ),
            source_bounds_cm=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        )


def test_concentric_geometry_is_run_mode_independent():
    geometry = _geometry()

    fixed = _case(geometry)
    eigenvalue = _case(geometry, run_mode="eigenvalue")

    assert fixed.geometry is eigenvalue.geometry is geometry
    assert fixed.geometry_type == eigenvalue.geometry_type == "concentric"
    assert fixed.source_bounds_cm == eigenvalue.source_bounds_cm


def test_prepare_rejects_opensn_for_concentric_geometry_before_writing(tmp_path):
    destination = tmp_path / "unsupported"

    with pytest.raises(ValueError, match="OpenSn.*concentric"):
        prepare(_case(_geometry()), destination)

    assert not destination.exists()


def test_explicit_statepoint_override_is_preserved(statepoint_selector, tmp_path):
    explicit = tmp_path / "named-statepoint.h5"
    explicit.touch()
    selector = _use_statepoint_directory(statepoint_selector, tmp_path)

    assert selector(explicit) == explicit


def test_single_statepoint_is_discovered(statepoint_selector, tmp_path):
    expected = tmp_path / "statepoint.7.h5"
    expected.touch()
    selector = _use_statepoint_directory(statepoint_selector, tmp_path)

    assert selector() == expected


def test_greatest_statepoint_batch_is_selected_numerically(
    statepoint_selector, tmp_path
):
    for batch in (2, 30, 100):
        (tmp_path / f"statepoint.{batch}.h5").touch()
    selector = _use_statepoint_directory(statepoint_selector, tmp_path)

    assert selector().name == "statepoint.100.h5"


def test_statepoint_batch_ten_sorts_after_nine(statepoint_selector, tmp_path):
    (tmp_path / "statepoint.9.h5").touch()
    (tmp_path / "statepoint.10.h5").touch()
    selector = _use_statepoint_directory(statepoint_selector, tmp_path)

    assert selector().name == "statepoint.10.h5"


def test_missing_statepoints_fail_clearly(statepoint_selector, tmp_path):
    selector = _use_statepoint_directory(statepoint_selector, tmp_path)

    with pytest.raises(FileNotFoundError, match="no valid OpenMC statepoint"):
        selector()


def test_malformed_statepoint_names_are_ignored(statepoint_selector, tmp_path):
    for name in (
        "statepoint.final.h5",
        "statepoint.-8.h5",
        "statepoint.12.h5.backup",
        "statepoint.20.1.h5",
    ):
        (tmp_path / name).touch()
    expected = tmp_path / "statepoint.6.h5"
    expected.touch()
    selector = _use_statepoint_directory(statepoint_selector, tmp_path)

    assert selector() == expected


def test_detector_native_openmc_geometry_and_cell_domain_mapping(tmp_path):
    from examples.detector.case import CASE, HDPE

    run = prepare(CASE, tmp_path / "detector", solvers=("openmc",))
    generated = runpy.run_path(run / "openmc/model.py")
    model = generated["MODEL"]
    library = generated["MGXS_LIBRARY"]
    domains = generated["MGXS_DOMAINS"]
    cells = list(model.geometry.get_all_cells().values())

    assert [cell.name for cell in cells] == ["He3", "Hdpe", "Cadmium", "Alu", "Outer"]
    assert [item["xsdata_name"] for item in domains] == [
        "he3", "hdpe", "cad", "alu", "outer"
    ]
    assert library.domain_type == "cell"
    assert library.domains == cells
    assert len(cells) == 5
    assert cells[1].fill is cells[4].fill
    assert cells[1].fill.name == HDPE.name
    assert model.settings.source[0].energy.a == 0.0
    assert model.settings.source[0].energy.b == 20.0e6
    assert tuple(model.settings.source[0].space.lower_left) == (-5.0, -5.0, -5.0)
    assert tuple(model.settings.source[0].space.upper_right) == (5.0, 5.0, 5.0)
    assert model.settings.trigger_active is None
    assert library.tally_trigger is None

    surfaces = model.geometry.get_all_surfaces()
    cylinders = sorted(
        surface.r for surface in surfaces.values() if surface.__class__.__name__ == "ZCylinder"
    )
    assert cylinders == [1.0, 4.0, 4.05, 4.3]
    planes = list(surfaces.values())
    assert sorted(
        surface.x0 for surface in planes if surface.__class__.__name__ == "XPlane"
    ) == [-5.0, 5.0]
    assert sorted(
        surface.y0 for surface in planes if surface.__class__.__name__ == "YPlane"
    ) == [-5.0, 5.0]
    assert sorted(
        surface.z0 for surface in planes if surface.__class__.__name__ == "ZPlane"
    ) == [-5.0, 5.0]
    assert {surface.boundary_type for surface in planes if surface.__class__.__name__ != "ZCylinder"} == {
        "reflective"
    }
    region_surfaces = [
        [
            (surface.__class__.__name__, getattr(surface, "r", None), side)
            for surface, side in _halfspaces(cell.region)
        ]
        for cell in cells
    ]
    assert region_surfaces[0] == [
        ("ZCylinder", 1.0, "-"),
        ("ZPlane", None, "+"),
        ("ZPlane", None, "-"),
    ]
    assert region_surfaces[1] == [
        ("ZCylinder", 1.0, "+"),
        ("ZCylinder", 4.0, "-"),
        ("ZPlane", None, "+"),
        ("ZPlane", None, "-"),
    ]
    assert region_surfaces[-1][-1] == ("ZCylinder", 4.3, "+")
    assert [name for name, _, _ in region_surfaces[-1][:-1]] == [
        "XPlane", "XPlane", "YPlane", "YPlane", "ZPlane", "ZPlane"
    ]

    captured = {}

    class FakeLibrary:
        def create_mg_library(self, *, xsdata_names):
            captured["names"] = xsdata_names
            return "mean-library"

    assert generated["create_mean_mg_library"](FakeLibrary()) == "mean-library"
    assert captured["names"] == ["he3", "hdpe", "cad", "alu", "outer"]


def test_pu9_hdpe_native_openmc_geometry_and_eigenvalue_settings(tmp_path):
    from examples.pu9_hdpe.case import CASE

    run = prepare(CASE, tmp_path / "pu9_hdpe", solvers=("openmc",))
    generated = runpy.run_path(run / "openmc/model.py")
    model = generated["MODEL"]
    library = generated["MGXS_LIBRARY"]
    cells = list(model.geometry.get_all_cells().values())
    surfaces = list(model.geometry.get_all_surfaces().values())

    assert [cell.name for cell in cells] == ["Pu9", "Hdpe"]
    assert [cell.name for cell in library.domains] == ["Pu9", "Hdpe"]
    assert [item["xsdata_name"] for item in generated["MGXS_DOMAINS"]] == ["pu", "hdpe"]
    cylinders = sorted(
        (surface for surface in surfaces if surface.__class__.__name__ == "ZCylinder"),
        key=lambda surface: surface.r,
    )
    assert [(surface.r, surface.boundary_type) for surface in cylinders] == [
        (1.0, "transmission"),
        (1.1, "reflective"),
    ]
    zplanes = sorted(
        (surface for surface in surfaces if surface.__class__.__name__ == "ZPlane"),
        key=lambda surface: surface.z0,
    )
    assert [(surface.z0, surface.boundary_type) for surface in zplanes] == [
        (-1.0, "reflective"),
        (1.0, "reflective"),
    ]
    assert model.settings.run_mode == "eigenvalue"
    assert (model.settings.batches, model.settings.inactive, model.settings.particles) == (
        60,
        10,
        5_000_000,
    )
    assert model.settings.source[0].constraints["fissionable"] is True
    assert tuple(model.settings.source[0].space.lower_left) == (-2.0, -2.0, -2.0)
    assert tuple(model.settings.source[0].space.upper_right) == (2.0, 2.0, 2.0)


def test_homogeneous_and_flattop_geometry_paths_remain_material_domain(tmp_path):
    from examples.flattop.case import CASE as flattop

    simple = Case(
        name="simple",
        materials=(_material("simple"),),
        energy_groups=(1.0, 2.0),
        source_kind="uniform_energy",
        target_dimensions_cm=(2.0, 2.0, 2.0),
    )

    for name, case in (("simple", simple), ("flattop", flattop)):
        run = prepare(case, tmp_path / name, solvers=("openmc",))
        generated = runpy.run_path(run / "openmc/model.py")
        assert generated["MGXS_LIBRARY"].domain_type == "material"
        assert generated["GEOMETRY"]["type"] == "homogeneous"
