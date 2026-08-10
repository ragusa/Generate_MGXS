import json

import h5py
import numpy as np
import pytest

from generate_mgxs import MGXS, load_mgxs, solve_infinite_medium
from conftest import EVIDENCE, write_tiny_mgxs


BE9_HDF5 = EVIDENCE / "be9_mgxs.h5"


def test_load_be9_hdf5_contract():
    """OpenMC HDF5 becomes ascending arrays with [moment, g_in, g_out] scatter."""
    xs = load_mgxs(BE9_HDF5, "be9", 294.0)

    assert xs.energy_bounds_ev.shape == (70,)
    assert xs.total.shape == xs.absorption.shape == (69,)
    assert xs.scatter.shape == (4, 69, 69)
    assert xs.logical_domain == "be9"
    assert xs.fission is xs.nu_fission is xs.chi is None


def test_factorized_fission_hdf5_contract_needs_no_production_matrix(tmp_path):
    """Fission production is represented by nu-fission and chi, not a matrix."""
    path = tmp_path / "fissionable.h5"
    write_tiny_mgxs(path)
    with h5py.File(path, "r+") as h5:
        domain = h5["one"]
        domain.attrs["fissionable"] = True
        temperature = domain["294K"]
        temperature.create_dataset("fission", data=[0.2, 0.1])
        temperature.create_dataset("nu-fission", data=[0.5, 0.25])
        temperature.create_dataset("chi", data=[0.6, 0.2])

    xs = load_mgxs(path, "one", 294.0)

    # OpenMC HDF5 is high-to-low, while package arrays are ascending energy.
    np.testing.assert_array_equal(xs.fission, [0.1, 0.2])
    np.testing.assert_array_equal(xs.nu_fission, [0.25, 0.5])
    np.testing.assert_array_equal(xs.chi, [0.25, 0.75])
    np.testing.assert_array_equal(
        xs.nu_fission[:, np.newaxis] * xs.chi[np.newaxis, :],
        [[0.0625, 0.1875], [0.125, 0.375]],
    )
    with h5py.File(path, "r") as h5:
        assert "nu-fission matrix" not in h5["one/294K"]


def test_be9_scatter_orientation_reproduces_direct_oracle():
    """Reversing either scatter axis incorrectly would destroy this reference solve."""
    xs = load_mgxs(BE9_HDF5, "be9", 294.0)
    source = np.diff(xs.energy_bounds_ev) / np.ptp(xs.energy_bounds_ev)
    solution = solve_infinite_medium(xs, source, 400.0)
    expected = json.loads((EVIDENCE / "be9_direct_result.json").read_text())

    np.testing.assert_allclose(solution.spectrum.values, expected["flux"], rtol=2e-14)
    assert solution.spectrum.values.sum() == pytest.approx(1578.819468067599, rel=2e-14)


def test_missing_domain_is_clear():
    with pytest.raises(KeyError, match="absent"):
        load_mgxs(BE9_HDF5, "missing")


def test_missing_temperature_is_clear():
    with pytest.raises(KeyError, match="temperature"):
        load_mgxs(BE9_HDF5, "be9", 600.0)


def test_invalid_group_structure_is_rejected(tmp_path):
    path = tmp_path / "bad.h5"
    write_tiny_mgxs(path)
    with h5py.File(path, "r+") as h5:
        h5.attrs["group structure"] = [1.0, 1.0, 2.0]

    with pytest.raises(ValueError, match="ascending"):
        load_mgxs(path, "one")


def test_uncertainty_sidecar_is_preserved(tmp_path):
    """MGXS means and their statistical uncertainty remain separate data products."""
    path = tmp_path / "openmc/mgxs.h5"
    write_tiny_mgxs(path)
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    sidecar = {"domains": {"one": {"quantities": {
        "total": {"std_dev": [0.01, 0.02]},
        "absorption": {"std_dev": [0.03, 0.04]},
        "scatter": {"std_dev": [[[0.1, 0.0], [0.0, 0.2]]]},
    }}}}
    (diagnostics / "mgxs_uncertainty.json").write_text(json.dumps(sidecar))

    xs = load_mgxs(path, "one")

    assert set(xs.uncertainty) == {"total", "absorption", "scatter"}
    np.testing.assert_array_equal(xs.uncertainty["total"], [0.01, 0.02])


def test_invalid_uncertainty_shape_is_rejected(tmp_path):
    path = tmp_path / "mgxs.h5"
    write_tiny_mgxs(path)
    sidecar = tmp_path / "uncertainty.json"
    sidecar.write_text(json.dumps({
        "domains": {"one": {"quantities": {"total": {"std_dev": [0.1]}}}}
    }))

    with pytest.raises(ValueError, match="uncertainty"):
        load_mgxs(path, "one", uncertainty_path=sidecar)


def test_raw_chi_uncertainty_is_not_reused_for_normalized_chi(tmp_path):
    """Chi normalization changes uncertainty, so raw sigma cannot be relabeled."""
    path = tmp_path / "mgxs.h5"
    write_tiny_mgxs(path)
    with h5py.File(path, "r+") as h5:
        h5["one"].attrs["fissionable"] = True
        h5["one/294K"].create_dataset("fission", data=[0.2, 0.1])
        h5["one/294K"].create_dataset("nu-fission", data=[0.5, 0.25])
        h5["one/294K"].create_dataset("chi", data=[0.6, 0.2])
    sidecar = tmp_path / "uncertainty.json"
    sidecar.write_text(json.dumps({
        "domains": {"one": {
            "chi": {"normalized_uncertainty": None},
            "quantities": {"chi_raw": {"std_dev": [0.03, 0.01]}},
        }}
    }))

    xs = load_mgxs(path, "one", uncertainty_path=sidecar)

    np.testing.assert_array_equal(xs.uncertainty["chi_raw"], [0.03, 0.01])
    assert "chi" not in xs.uncertainty


def test_chi_raw_and_normalization_metadata(tmp_path):
    """Preserve raw chi and record how its normalized vector was closed to unity."""
    path = tmp_path / "mgxs.h5"
    write_tiny_mgxs(path)
    with h5py.File(path, "r+") as h5:
        domain = h5["one"]
        domain.attrs["fissionable"] = True
        temperature = domain["294K"]
        temperature.create_dataset("fission", data=[0.2, 0.1])
        temperature.create_dataset("nu-fission", data=[0.5, 0.25])
        temperature.create_dataset("chi", data=[0.6, 0.2])

    xs = load_mgxs(path, "one")

    np.testing.assert_array_equal(xs.chi_raw, [0.2, 0.6])
    assert xs.chi_raw_sum == pytest.approx(0.8)
    assert xs.chi.sum() == 1.0
    assert xs.chi_normalization_factor == pytest.approx(1.25)
    assert xs.chi_closure_correction[0] == 1


def test_invalid_chi_sum_is_rejected(tmp_path):
    path = tmp_path / "mgxs.h5"
    write_tiny_mgxs(path)
    with h5py.File(path, "r+") as h5:
        h5["one"].attrs["fissionable"] = True
        h5["one/294K"].create_dataset("fission", data=[0.2, 0.1])
        h5["one/294K"].create_dataset("nu-fission", data=[0.5, 0.25])
        h5["one/294K"].create_dataset("chi", data=[0.0, 0.0])

    with pytest.raises(ValueError, match="raw chi"):
        load_mgxs(path, "one")


def test_fissionability_and_dataset_presence_must_agree(tmp_path):
    path = tmp_path / "mgxs.h5"
    write_tiny_mgxs(path)
    with h5py.File(path, "r+") as h5:
        h5["one"].attrs["fissionable"] = True

    with pytest.raises(ValueError, match="missing"):
        load_mgxs(path, "one")


def test_direct_solver_uses_incoming_outgoing_orientation():
    """Rows are incident groups, so the balance operator requires scatter P0.T."""
    scatter = np.zeros((1, 2, 2))
    scatter[0, 0, 1] = 0.5
    xs = MGXS(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0]),
        np.array([1.5, 2.5]),
        scatter,
        "asymmetric",
        294.0,
    )

    solution = solve_infinite_medium(xs, [0.75, 0.25], 4.0)
    expected_density = np.linalg.solve(
        np.diag(xs.total) - scatter[0].T,
        np.array([0.75, 0.25]) / 4.0,
    )

    np.testing.assert_allclose(solution.flux_density, expected_density)
    np.testing.assert_allclose(solution.spectrum.values, expected_density * 4.0)
    assert solution.residual < 1e-14
    assert abs(solution.balance) < 1e-14


def test_direct_solver_volume_scaling():
    """Source density scales by volume while the integrated spectrum does not."""
    xs = MGXS(
        np.array([1.0, 2.0]),
        np.array([2.0]),
        np.array([2.0]),
        np.zeros((1, 1, 1)),
        "one",
        294.0,
    )

    a = solve_infinite_medium(xs, [1.0], 1.0)
    b = solve_infinite_medium(xs, [1.0], 10.0)

    np.testing.assert_allclose(a.flux_density, b.flux_density * 10.0)
    np.testing.assert_allclose(a.spectrum.values, b.spectrum.values)


def test_direct_solver_rejects_invalid_source():
    xs = load_mgxs(BE9_HDF5, "be9")

    with pytest.raises(ValueError, match="normalized"):
        solve_infinite_medium(xs, np.zeros(69), 400.0)


def test_direct_solver_rejects_fissionable_mgxs():
    """The current direct equation has no fission-production term."""
    xs = MGXS(
        np.array([1.0, 2.0]),
        np.array([2.0]),
        np.array([1.0]),
        np.zeros((1, 1, 1)),
        "fissionable",
        294.0,
        fission=np.array([0.1]),
        nu_fission=np.array([0.2]),
        chi=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="non-fissionable homogeneous"):
        solve_infinite_medium(xs, [1.0], 1.0)
