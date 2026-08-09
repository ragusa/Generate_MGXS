import matplotlib
import warnings

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from generate_mgxs import MGXS, Spectrum, plot_mgxs, plot_spectra


def synthetic_mgxs(*, fissionable=False, zero_lower_bound=False):
    """Create asymmetric two-group data so plotting orientation is observable."""
    energy_bounds = (
        np.array([0.0, 1.0, 1.0e8])
        if zero_lower_bound
        else np.array([1.0, 10.0, 100.0])
    )
    scatter = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.1, -0.2], [0.3, -0.4]],
        ]
    )
    optional = {}
    if fissionable:
        optional = {
            "fission": np.array([0.05, 0.1]),
            "nu_fission": np.array([0.12, 0.25]),
            "chi": np.array([0.8, 0.2]),
        }

    return MGXS(
        energy_bounds,
        np.array([0.8, 1.1]),
        np.array([0.2, 0.3]),
        scatter,
        "fuel domain" if fissionable else "be9",
        294.0,
        **optional,
    )


def comparison_spectra():
    """Return three distinct ascending-energy spectra with OpenMC uncertainty."""
    bounds = np.array([1.0, np.e, np.e**3])
    openmc = Spectrum(bounds, [2.2, 5.8], std_dev=[0.2, 0.6])
    opensn = Spectrum(bounds, [0.9, 3.1])
    direct = Spectrum(bounds, [1.0, 3.0])

    return openmc, opensn, direct


def test_plot_spectra_writes_three_diagnostics_with_solver_markers_and_uncertainty(
    tmp_path,
):
    openmc, opensn, direct = comparison_spectra()

    figures = plot_spectra(
        openmc,
        opensn,
        direct,
        output_directory=tmp_path,
    )

    assert set(figures) == {
        "group_spectrum",
        "flux_per_lethargy",
        "relative_differences",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "group_spectrum.png",
        "flux_per_lethargy.png",
        "relative_differences.png",
    }

    group_axis = figures["group_spectrum"].axes[0]
    assert [line.get_marker() for line in group_axis.lines] == ["o", "s", "^"]
    assert group_axis.get_xscale() == "log"

    band = next(
        patch
        for patch in group_axis.patches
        if patch.get_label() == "OpenMC ±1σ (covariance ignored)"
    )
    sigma = openmc.std_dev / np.sum(openmc.values)
    np.testing.assert_allclose(band.get_data().values, openmc.normalized + sigma)
    np.testing.assert_allclose(band.get_data().baseline, openmc.normalized - sigma)

    plt.close("all")


def test_plot_spectra_lethargy_and_relative_difference_definitions():
    openmc, opensn, direct = comparison_spectra()

    figures = plot_spectra(openmc, opensn, direct)

    lethargy_axis = figures["flux_per_lethargy"].axes[0]
    widths = np.log(direct.energy_bounds_ev[1:] / direct.energy_bounds_ev[:-1])
    for label, spectrum in (
        ("OpenMC", openmc),
        ("OpenSn", opensn),
        ("Direct", direct),
    ):
        patch = next(item for item in lethargy_axis.patches if item.get_label() == label)
        np.testing.assert_allclose(patch.get_data().values, spectrum.normalized / widths)
    assert lethargy_axis.get_xscale() == lethargy_axis.get_yscale() == "log"
    assert [line.get_marker() for line in lethargy_axis.lines] == ["o", "s", "^"]

    relative_axis = figures["relative_differences"].axes[0]
    direct_values = direct.normalized
    floor = max(1.0e-14, 1.0e-6 * np.max(np.abs(direct_values)))
    denominator = np.maximum(np.abs(direct_values), floor)
    for label, spectrum in (("OpenMC", openmc), ("OpenSn", opensn)):
        patch = next(
            item
            for item in relative_axis.patches
            if item.get_label() == f"{label} vs Direct"
        )
        expected = (spectrum.normalized - direct_values) / denominator
        np.testing.assert_allclose(patch.get_data().values, expected)
    assert not any("Direct vs Direct" in patch.get_label() for patch in relative_axis.patches)
    assert [line.get_marker() for line in relative_axis.lines[:2]] == ["o", "s"]

    plt.close("all")


def test_plot_spectra_requires_identical_energy_bounds():
    openmc, opensn, direct = comparison_spectra()
    opensn = Spectrum([1.0, 2.0, np.e**3], opensn.values)

    with pytest.raises(ValueError, match="identical energy bounds"):
        plot_spectra(openmc, opensn, direct)


def test_plot_spectra_accepts_openmc_and_direct_without_opensn():
    openmc, _, direct = comparison_spectra()

    figures = plot_spectra(
        openmc,
        None,
        direct,
        include=("openmc", "direct"),
    )

    assert set(figures) == {
        "group_spectrum",
        "flux_per_lethargy",
        "relative_differences",
    }
    group_labels = {
        patch.get_label()
        for patch in figures["group_spectrum"].axes[0].patches
    }
    assert "OpenMC" in group_labels
    assert "Direct" in group_labels
    assert "OpenSn" not in group_labels
    relative_labels = {
        patch.get_label()
        for patch in figures["relative_differences"].axes[0].patches
    }
    assert relative_labels == {"OpenMC vs Direct"}

    plt.close("all")


def test_plot_spectra_accepts_opensn_and_direct_without_openmc():
    _, opensn, direct = comparison_spectra()

    figures = plot_spectra(
        None,
        opensn,
        direct,
        include=("opensn", "direct"),
    )

    group_labels = {
        patch.get_label()
        for patch in figures["group_spectrum"].axes[0].patches
    }
    assert group_labels == {"OpenSn", "Direct"}
    relative_labels = {
        patch.get_label()
        for patch in figures["relative_differences"].axes[0].patches
    }
    assert relative_labels == {"OpenSn vs Direct"}

    plt.close("all")


def test_plot_spectra_accepts_openmc_only():
    openmc, _, _ = comparison_spectra()

    figures = plot_spectra(
        openmc,
        None,
        None,
        include=("openmc",),
    )

    assert set(figures) == {"group_spectrum", "flux_per_lethargy"}
    group_labels = {
        patch.get_label()
        for patch in figures["group_spectrum"].axes[0].patches
    }
    assert "OpenMC" in group_labels
    assert "OpenSn" not in group_labels
    assert "Direct" not in group_labels

    plt.close("all")


def test_plot_spectra_rejects_requested_missing_result():
    openmc, _, direct = comparison_spectra()

    with pytest.raises(ValueError, match="OpenSn result was requested but is None"):
        plot_spectra(
            openmc,
            None,
            direct,
            include=("openmc", "opensn", "direct"),
        )


def test_zero_energy_spectrum_uses_plotting_edge_without_mutating_physics():
    physical_bounds = np.array([0.0, 1.0, 1.0e8])
    openmc = Spectrum(physical_bounds, [1.0, 2.0], std_dev=[0.1, 0.2])
    opensn = Spectrum(physical_bounds, [1.1, 1.9])
    direct = Spectrum(physical_bounds, [0.9, 2.1])
    original_bounds = [
        spectrum.energy_bounds_ev.copy()
        for spectrum in (openmc, opensn, direct)
    ]

    with pytest.warns(UserWarning, match="using 1e-5 eV for logarithmic plotting only"):
        figures = plot_spectra(openmc, opensn, direct)

    group_axis = figures["group_spectrum"].axes[0]
    plotted_edges = group_axis.patches[0].get_data().edges
    np.testing.assert_array_equal(plotted_edges, [1.0e-5, 1.0, 1.0e8])
    assert group_axis.lines[0].get_xdata()[0] == pytest.approx(np.sqrt(1.0e-5))
    assert group_axis.get_xlim()[0] <= 1.0e-5
    assert group_axis.get_xlim()[1] >= 1.0e8

    for spectrum, original in zip((openmc, opensn, direct), original_bounds):
        np.testing.assert_array_equal(spectrum.energy_bounds_ev, original)
        assert spectrum.energy_bounds_ev[0] == 0.0

    plt.close("all")


def test_zero_energy_matrix_spans_full_range_and_preserves_orientation():
    mgxs = synthetic_mgxs(fissionable=True, zero_lower_bound=True)
    original_bounds = mgxs.energy_bounds_ev.copy()
    original_scatter = mgxs.scatter.copy()
    original_nu_fission = mgxs.nu_fission.copy()
    original_chi = mgxs.chi.copy()

    with pytest.warns(UserWarning, match="using 1e-5 eV for logarithmic plotting only"):
        figures = plot_mgxs(mgxs)

    vector_edges = figures["cross_sections"].axes[0].patches[0].get_data().edges
    plotting_bounds = np.array([1.0e-5, 1.0, 1.0e8])
    np.testing.assert_array_equal(vector_edges, plotting_bounds)

    expected_matrices = {
        "scatter_p0": original_scatter[0],
        "fission_matrix": original_nu_fission[:, None] * original_chi[None, :],
    }
    for name, scientific_matrix in expected_matrices.items():
        matrix_axis = figures[name].axes[0]
        mesh = matrix_axis.collections[0]
        coordinates = mesh.get_coordinates()

        np.testing.assert_array_equal(coordinates[0, :, 0], plotting_bounds)
        np.testing.assert_array_equal(coordinates[:, 0, 1], plotting_bounds)
        np.testing.assert_array_equal(
            mesh.get_array().reshape(2, 2),
            scientific_matrix.T,
        )
        assert matrix_axis.get_xlim() == pytest.approx((1.0e8, 1.0e-5))
        assert matrix_axis.get_ylim() == pytest.approx((1.0e-5, 1.0e8))

    np.testing.assert_array_equal(mgxs.energy_bounds_ev, original_bounds)
    np.testing.assert_array_equal(mgxs.scatter, original_scatter)
    np.testing.assert_array_equal(mgxs.nu_fission, original_nu_fission)
    np.testing.assert_array_equal(mgxs.chi, original_chi)
    assert mgxs.energy_bounds_ev[0] == 0.0

    plt.close("all")


def test_positive_energy_boundaries_are_unchanged_and_do_not_warn():
    openmc, opensn, direct = comparison_spectra()
    mgxs = synthetic_mgxs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spectrum_figures = plot_spectra(openmc, opensn, direct)
        mgxs_figures = plot_mgxs(mgxs)

    assert not any("Lowest energy boundary" in str(item.message) for item in caught)
    spectrum_edges = (
        spectrum_figures["group_spectrum"].axes[0].patches[0].get_data().edges
    )
    mgxs_edges = mgxs_figures["cross_sections"].axes[0].patches[0].get_data().edges
    np.testing.assert_array_equal(spectrum_edges, direct.energy_bounds_ev)
    np.testing.assert_array_equal(mgxs_edges, mgxs.energy_bounds_ev)

    plt.close("all")


def test_nonfissionable_plot_uses_energy_edges_and_requested_scatter_orientation():
    """Plotting transposes the view, never canonical [g_in, g_out] data."""
    mgxs = synthetic_mgxs()
    original = {
        "total": mgxs.total.copy(),
        "absorption": mgxs.absorption.copy(),
        "scatter": mgxs.scatter.copy(),
    }

    figures = plot_mgxs(mgxs, scatter_moments=(1,))

    assert set(figures) == {"cross_sections", "scatter_p1"}
    vector_axis = figures["cross_sections"].axes[0]
    plotted_edges = np.unique(vector_axis.patches[0].get_path().vertices[:, 0])
    np.testing.assert_array_equal(plotted_edges, mgxs.energy_bounds_ev)

    matrix_axis = figures["scatter_p1"].axes[0]
    mesh = matrix_axis.collections[0]
    np.testing.assert_array_equal(mesh.get_array().reshape(2, 2), mgxs.scatter[1].T)
    coordinates = mesh.get_coordinates()
    np.testing.assert_array_equal(coordinates[0, :, 0], mgxs.energy_bounds_ev)
    np.testing.assert_array_equal(coordinates[:, 0, 1], mgxs.energy_bounds_ev)
    assert matrix_axis.get_xlabel() == "Incoming energy [eV]"
    assert matrix_axis.get_ylabel() == "Outgoing energy [eV]"
    assert matrix_axis.get_xlim()[0] > matrix_axis.get_xlim()[1]
    assert matrix_axis.get_ylim()[0] < matrix_axis.get_ylim()[1]

    for name, values in original.items():
        np.testing.assert_array_equal(getattr(mgxs, name), values)

    plt.close("all")


def test_fissionable_plot_returns_and_saves_all_applicable_figures(tmp_path):
    mgxs = synthetic_mgxs(fissionable=True)
    nu_fission = mgxs.nu_fission.copy()
    chi = mgxs.chi.copy()

    figures = plot_mgxs(mgxs, output_directory=tmp_path)

    assert set(figures) == {
        "cross_sections",
        "chi",
        "scatter_p0",
        "fission_matrix",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "fuel_domain_cross_sections.png",
        "fuel_domain_chi.png",
        "fuel_domain_scatter_p0.png",
        "fuel_domain_fission_matrix.png",
    }

    # F[g_in, g_out] is derived from incident-group nu-fission and the
    # destination-group solver-ready chi distribution.
    mesh = figures["fission_matrix"].axes[0].collections[0]
    expected = nu_fission[:, None] * chi[None, :]
    np.testing.assert_array_equal(mesh.get_array().reshape(2, 2), expected.T)
    matrix_axis = figures["fission_matrix"].axes[0]
    assert matrix_axis.get_xlim()[0] > matrix_axis.get_xlim()[1]
    assert matrix_axis.get_ylim()[0] < matrix_axis.get_ylim()[1]
    np.testing.assert_array_equal(mgxs.nu_fission, nu_fission)
    np.testing.assert_array_equal(mgxs.chi, chi)

    plt.close("all")


def test_plot_mgxs_validates_moments_and_show_is_explicit(monkeypatch):
    mgxs = synthetic_mgxs()
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    figures = plot_mgxs(mgxs, scatter_moments=(), show=True)

    assert set(figures) == {"cross_sections"}
    assert shown == [True]

    with pytest.raises(ValueError, match="invalid scattering moment"):
        plot_mgxs(mgxs, scatter_moments=(2,))

    plt.close("all")
