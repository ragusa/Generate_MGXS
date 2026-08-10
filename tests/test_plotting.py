import matplotlib
import warnings

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pytest

from generate_mgxs import (
    MGXS,
    Spectrum,
    plot_mgxs,
    plot_openmc_domain_spectra,
    plot_spectra,
)


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


def labelled_line(axis, label):
    """Return one primary solver curve, excluding marker-only overlays."""
    return next(line for line in axis.lines if line.get_label() == label)


def assert_stepwise_band(axis, lower, upper):
    """Check the y levels represented by a stepwise uncertainty collection."""
    band = next(
        collection
        for collection in axis.collections
        if collection.get_label() == "OpenMC ±1σ (covariance ignored)"
    )
    actual_levels = np.unique(band.get_paths()[0].vertices[:, 1])
    expected_levels = np.unique(
        np.concatenate(
            (
                np.insert(lower, 0, lower[0]),
                np.insert(upper, 0, upper[0]),
            )
        )
    )
    np.testing.assert_allclose(actual_levels, expected_levels)


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
        "flux_spectrum",
        "lethargy_spectrum",
        "relative_differences",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "flux_spectrum.png",
        "lethargy_spectrum.png",
        "relative_differences.png",
    }

    energy_axis = figures["flux_spectrum"].axes[0]
    marker_lines = [line for line in energy_axis.lines if line.get_label() == "_nolegend_"]
    assert [line.get_marker() for line in marker_lines] == ["o", "s", "^"]
    assert energy_axis.get_xscale() == energy_axis.get_yscale() == "log"
    energy_legend = energy_axis.get_legend()
    assert [handle.get_marker() for handle in energy_legend.legend_handles[:3]] == [
        "o",
        "s",
        "^",
    ]

    widths = np.diff(openmc.energy_bounds_ev)
    midpoints = openmc.energy_bounds_ev[:-1] + 0.5 * widths
    normalized_sigma = openmc.std_dev / np.sum(openmc.values)
    energy_sigma = normalized_sigma / widths
    energy_values = openmc.normalized / widths
    assert_stepwise_band(
        energy_axis,
        energy_values - energy_sigma,
        energy_values + energy_sigma,
    )

    lethargy_axis = figures["lethargy_spectrum"].axes[0]
    lethargy_legend = lethargy_axis.get_legend()
    assert [handle.get_marker() for handle in lethargy_legend.legend_handles[:3]] == [
        "o",
        "s",
        "^",
    ]
    lethargy_values = midpoints * energy_values
    lethargy_sigma = midpoints * energy_sigma
    assert_stepwise_band(
        lethargy_axis,
        lethargy_values - lethargy_sigma,
        lethargy_values + lethargy_sigma,
    )

    plt.close("all")


def test_plot_spectra_energy_lethargy_and_relative_difference_definitions():
    openmc, opensn, direct = comparison_spectra()

    figures = plot_spectra(openmc, opensn, direct)

    widths = np.diff(direct.energy_bounds_ev)
    midpoints = direct.energy_bounds_ev[:-1] + 0.5 * widths
    energy_axis = figures["flux_spectrum"].axes[0]
    lethargy_axis = figures["lethargy_spectrum"].axes[0]
    for label, spectrum in (
        ("OpenMC", openmc),
        ("OpenSn", opensn),
        ("Direct", direct),
    ):
        normalized = spectrum.normalized
        energy_values = normalized / widths
        energy_line = labelled_line(energy_axis, label)
        lethargy_line = labelled_line(lethargy_axis, label)
        expected_energy_plot = np.insert(energy_values, 0, energy_values[0])
        expected_lethargy = midpoints * energy_values
        expected_lethargy_plot = np.insert(
            expected_lethargy,
            0,
            expected_lethargy[0],
        )

        assert np.sum(normalized) == pytest.approx(1.0)
        np.testing.assert_array_equal(energy_line.get_xdata(), direct.energy_bounds_ev)
        np.testing.assert_allclose(energy_line.get_ydata(), expected_energy_plot)
        assert energy_line.get_ydata()[0] == energy_line.get_ydata()[1]
        assert energy_line.get_drawstyle() == "steps"
        assert np.sum(energy_values * widths) == pytest.approx(1.0)
        np.testing.assert_array_equal(lethargy_line.get_xdata(), direct.energy_bounds_ev)
        np.testing.assert_allclose(
            lethargy_line.get_ydata(),
            expected_lethargy_plot,
        )
        assert lethargy_line.get_ydata()[0] == lethargy_line.get_ydata()[1]
        assert lethargy_line.get_drawstyle() == "steps"

    assert energy_axis.get_xscale() == energy_axis.get_yscale() == "log"
    assert lethargy_axis.get_xscale() == "log"
    assert lethargy_axis.get_yscale() == "linear"
    marker_lines = [
        line for line in lethargy_axis.lines if line.get_label() == "_nolegend_"
    ]
    assert [line.get_marker() for line in marker_lines] == ["o", "s", "^"]

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
        "flux_spectrum",
        "lethargy_spectrum",
        "relative_differences",
    }
    group_labels = {
        line.get_label()
        for line in figures["flux_spectrum"].axes[0].lines
        if line.get_label() != "_nolegend_"
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
        line.get_label()
        for line in figures["flux_spectrum"].axes[0].lines
        if line.get_label() != "_nolegend_"
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

    assert set(figures) == {"flux_spectrum", "lethargy_spectrum"}
    group_labels = {
        line.get_label()
        for line in figures["flux_spectrum"].axes[0].lines
        if line.get_label() != "_nolegend_"
    }
    assert "OpenMC" in group_labels
    assert "OpenSn" not in group_labels
    assert "Direct" not in group_labels

    plt.close("all")


def test_plot_openmc_domains_normalizes_each_shape_and_uses_line_steps(tmp_path):
    bounds = np.array([1.0, 2.0, 4.0])
    domain_spectra = {
        "he3": Spectrum(bounds, [2.0, 0.0], std_dev=[0.2, 0.0], logical_domain="he3"),
        "hdpe": Spectrum(bounds, [2.0, 6.0], std_dev=[0.2, 0.6], logical_domain="hdpe"),
        "cad": Spectrum(bounds, [1.0, 1.0], std_dev=[0.1, 0.1], logical_domain="cad"),
        "alu": Spectrum(bounds, [9.0, 1.0], std_dev=[0.9, 0.1], logical_domain="alu"),
        "outer": Spectrum(bounds, [20.0, 60.0], std_dev=[2.0, 6.0], logical_domain="outer"),
    }
    labels = {
        "he3": "He3",
        "hdpe": "Hdpe",
        "cad": "Cadmium",
        "alu": "Alu",
        "outer": "Outer",
    }
    originals = {
        name: (
            spectrum.energy_bounds_ev.copy(),
            spectrum.values.copy(),
            spectrum.std_dev.copy(),
        )
        for name, spectrum in domain_spectra.items()
    }

    figures = plot_openmc_domain_spectra(
        domain_spectra,
        labels=labels,
        output_directory=tmp_path,
    )

    assert set(figures) == {
        "openmc_domain_flux_spectra",
        "openmc_domain_lethargy_spectra",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "openmc_domain_flux_spectra.png",
        "openmc_domain_lethargy_spectra.png",
    }

    widths = np.diff(bounds)
    midpoints = bounds[:-1] + 0.5 * widths
    energy_axis = figures["openmc_domain_flux_spectra"].axes[0]
    lethargy_axis = figures["openmc_domain_lethargy_spectra"].axes[0]
    assert energy_axis.get_xscale() == energy_axis.get_yscale() == "log"
    assert lethargy_axis.get_xscale() == "log"
    assert lethargy_axis.get_yscale() == "linear"
    assert [line.get_label() for line in energy_axis.lines] == list(labels.values())
    assert not energy_axis.patches
    assert not lethargy_axis.patches

    for name, spectrum in domain_spectra.items():
        label = labels[name]
        energy_line = labelled_line(energy_axis, label)
        lethargy_line = labelled_line(lethargy_axis, label)
        energy_values = spectrum.normalized / widths
        expected_energy = np.append(energy_values, energy_values[-1])
        expected_energy[expected_energy <= 0.0] = np.nan
        expected_lethargy = np.append(
            midpoints * energy_values,
            (midpoints * energy_values)[-1],
        )

        assert np.sum(spectrum.normalized) == pytest.approx(1.0)
        assert energy_line.get_drawstyle() == "steps-post"
        assert lethargy_line.get_drawstyle() == "steps-post"
        np.testing.assert_array_equal(energy_line.get_xdata(), bounds)
        np.testing.assert_allclose(
            energy_line.get_ydata(),
            expected_energy,
            equal_nan=True,
        )
        np.testing.assert_allclose(lethargy_line.get_ydata(), expected_lethargy)

    # Proportional raw tallies overlay after independent shape normalization,
    # while their distinct domain identities remain in the legend.
    np.testing.assert_allclose(
        labelled_line(energy_axis, "Hdpe").get_ydata(),
        labelled_line(energy_axis, "Outer").get_ydata(),
    )
    for name, spectrum in domain_spectra.items():
        original_bounds, original_values, original_std = originals[name]
        np.testing.assert_array_equal(spectrum.energy_bounds_ev, original_bounds)
        np.testing.assert_array_equal(spectrum.values, original_values)
        np.testing.assert_array_equal(spectrum.std_dev, original_std)

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


def test_fine_group_spectrum_markers_are_thinned_deterministically():
    groups = 361
    bounds = np.geomspace(1.0e-5, 1.0e7, groups + 1)
    values = np.linspace(1.0, 2.0, groups)
    openmc = Spectrum(bounds, values)
    opensn = Spectrum(bounds, values * 1.01)
    direct = Spectrum(bounds, values * 0.99)

    figures = plot_spectra(openmc, opensn, direct)

    expected_stride = groups // 20
    for name in ("flux_spectrum", "lethargy_spectrum"):
        axis = figures[name].axes[0]
        marker_lines = [
            line for line in axis.lines if line.get_label() == "_nolegend_"
        ]
        primary_lines = [
            line for line in axis.lines if line.get_label() != "_nolegend_"
        ]

        assert len(marker_lines) == len(primary_lines) == 3
        assert all(line.get_markevery() == expected_stride for line in marker_lines)
        assert all(line.get_markersize() == pytest.approx(3.25) for line in marker_lines)
        assert all(line.get_xdata().size == groups + 1 for line in primary_lines)

    plt.close("all")


def test_zero_energy_spectrum_uses_plotting_edge_without_mutating_physics():
    physical_bounds = np.array([0.0, 1.0, 1.0e8])
    openmc = Spectrum(physical_bounds, [1.0, 2.0], std_dev=[0.1, 0.2])
    opensn = Spectrum(physical_bounds, [1.1, 1.9])
    direct = Spectrum(physical_bounds, [0.9, 2.1])
    original_bounds = [
        spectrum.energy_bounds_ev.copy()
        for spectrum in (openmc, opensn, direct)
    ]
    original_values = [
        spectrum.values.copy()
        for spectrum in (openmc, opensn, direct)
    ]

    with pytest.warns(UserWarning, match="using 1e-5 eV for logarithmic plotting only"):
        figures = plot_spectra(openmc, opensn, direct)

    energy_axis = figures["flux_spectrum"].axes[0]
    openmc_energy_line = labelled_line(energy_axis, "OpenMC")
    plotted_edges = openmc_energy_line.get_xdata()
    np.testing.assert_array_equal(plotted_edges, [1.0e-5, 1.0, 1.0e8])
    marker_line = next(
        line for line in energy_axis.lines if line.get_label() == "_nolegend_"
    )
    assert marker_line.get_xdata()[0] == pytest.approx(0.5)
    assert energy_axis.get_xlim()[0] <= 1.0e-5
    assert energy_axis.get_xlim()[1] >= 1.0e8

    physical_widths = np.diff(physical_bounds)
    physical_midpoints = physical_bounds[:-1] + 0.5 * physical_widths
    expected_energy = openmc.normalized / physical_widths
    expected_lethargy = physical_midpoints * expected_energy
    np.testing.assert_allclose(
        openmc_energy_line.get_ydata(),
        np.insert(expected_energy, 0, expected_energy[0]),
    )
    openmc_lethargy_line = labelled_line(
        figures["lethargy_spectrum"].axes[0],
        "OpenMC",
    )
    np.testing.assert_allclose(
        openmc_lethargy_line.get_ydata(),
        np.insert(expected_lethargy, 0, expected_lethargy[0]),
    )

    for spectrum, bounds, values in zip(
        (openmc, opensn, direct),
        original_bounds,
        original_values,
    ):
        np.testing.assert_array_equal(spectrum.energy_bounds_ev, bounds)
        np.testing.assert_array_equal(spectrum.values, values)
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

    vector_edges = labelled_line(
        figures["cross_sections"].axes[0],
        "Total",
    ).get_xdata()
    energy_plotting_bounds = np.array([1.0e-5, 1.0, 1.0e8])
    np.testing.assert_array_equal(vector_edges, energy_plotting_bounds)

    expected_matrices = {
        "scatter_p0": original_scatter[0],
        "fission_matrix": original_nu_fission[:, None] * original_chi[None, :],
    }
    for name, scientific_matrix in expected_matrices.items():
        matrix_axis = figures[name].axes[0]
        mesh = matrix_axis.collections[0]
        coordinates = mesh.get_coordinates()
        displayed_matrix = mesh.get_array().reshape(2, 2)

        np.testing.assert_array_equal(coordinates[0, :, 0], [0.5, 1.5, 2.5])
        np.testing.assert_array_equal(coordinates[:, 0, 1], [0.5, 1.5, 2.5])
        np.testing.assert_array_equal(
            displayed_matrix,
            scientific_matrix[::-1, ::-1].T,
        )
        assert displayed_matrix[0, 0] == scientific_matrix[-1, -1]
        assert displayed_matrix[-1, -1] == scientific_matrix[0, 0]
        assert matrix_axis.get_xlim() == pytest.approx((0.5, 2.5))
        assert matrix_axis.get_ylim() == pytest.approx((2.5, 0.5))

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
    spectrum_edges = labelled_line(
        spectrum_figures["flux_spectrum"].axes[0],
        "OpenMC",
    ).get_xdata()
    mgxs_edges = labelled_line(
        mgxs_figures["cross_sections"].axes[0],
        "Total",
    ).get_xdata()
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
    for label, values in (("Total", mgxs.total), ("Absorption", mgxs.absorption)):
        line = labelled_line(vector_axis, label)
        np.testing.assert_array_equal(line.get_xdata(), mgxs.energy_bounds_ev)
        np.testing.assert_array_equal(
            line.get_ydata(),
            np.insert(values, 0, values[0]),
        )
        assert line.get_ydata()[0] == line.get_ydata()[1]
        assert line.get_drawstyle() == "steps"
        assert line.get_linewidth() == pytest.approx(1.0)
    assert vector_axis.get_xscale() == "log"
    assert vector_axis.get_yscale() == "log"

    matrix_axis = figures["scatter_p1"].axes[0]
    mesh = matrix_axis.collections[0]
    assert not isinstance(mesh.norm, LogNorm)
    displayed = mesh.get_array().reshape(2, 2)
    expected = mgxs.scatter[1][::-1, ::-1].T
    np.testing.assert_array_equal(displayed, expected)
    assert displayed[0, 0] == mgxs.scatter[1, -1, -1]
    assert displayed[-1, -1] == mgxs.scatter[1, 0, 0]

    coordinates = mesh.get_coordinates()
    np.testing.assert_array_equal(coordinates[0, :, 0], [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(coordinates[:, 0, 1], [0.5, 1.5, 2.5])
    assert matrix_axis.get_xlabel() == "Incoming group"
    assert matrix_axis.get_ylabel() == "Outgoing group"
    assert matrix_axis.get_xscale() == matrix_axis.get_yscale() == "linear"
    assert matrix_axis.get_xlim() == pytest.approx((0.5, 2.5))
    assert matrix_axis.get_ylim() == pytest.approx((2.5, 0.5))
    np.testing.assert_array_equal(matrix_axis.get_xticks(), [1, 2])
    np.testing.assert_array_equal(matrix_axis.get_yticks(), [1, 2])

    for name, values in original.items():
        np.testing.assert_array_equal(getattr(mgxs, name), values)

    plt.close("all")


def test_fissionable_mgxs_vectors_use_explicit_step_line_data_without_mutation():
    """All cross-section vectors start horizontally at their first group value."""
    mgxs = synthetic_mgxs(fissionable=True)
    expected = {
        "Total": mgxs.total.copy(),
        "Absorption": mgxs.absorption.copy(),
        "Fission": mgxs.fission.copy(),
        "Nu-fission": mgxs.nu_fission.copy(),
    }
    original_bounds = mgxs.energy_bounds_ev.copy()

    figures = plot_mgxs(mgxs)
    axis = figures["cross_sections"].axes[0]

    assert {line.get_label() for line in axis.lines} == set(expected)
    for label, values in expected.items():
        line = labelled_line(axis, label)
        np.testing.assert_array_equal(line.get_xdata(), original_bounds)
        np.testing.assert_array_equal(
            line.get_ydata(),
            np.insert(values, 0, values[0]),
        )
        assert line.get_ydata()[0] == line.get_ydata()[1]
        assert line.get_drawstyle() == "steps"
        assert line.get_linewidth() == pytest.approx(1.0)

    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert any(line.get_visible() for line in axis.get_xgridlines())
    assert any(line.get_visible() for line in axis.get_ygridlines())
    np.testing.assert_array_equal(mgxs.energy_bounds_ev, original_bounds)
    np.testing.assert_array_equal(mgxs.total, expected["Total"])
    np.testing.assert_array_equal(mgxs.absorption, expected["Absorption"])
    np.testing.assert_array_equal(mgxs.fission, expected["Fission"])
    np.testing.assert_array_equal(mgxs.nu_fission, expected["Nu-fission"])

    plt.close("all")


def test_cross_section_log_y_masks_zeros_without_mutation():
    mgxs = synthetic_mgxs(fissionable=True)
    mgxs.total[0] = 0.0
    mgxs.absorption[1] = 0.0
    mgxs.fission[0] = 0.0
    mgxs.nu_fission[1] = 0.0
    mgxs.chi[1] = 0.0
    original = {
        name: getattr(mgxs, name).copy()
        for name in ("total", "absorption", "fission", "nu_fission", "chi")
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figures = plot_mgxs(mgxs)

    cross_sections = figures["cross_sections"].axes[0]
    assert cross_sections.get_xscale() == cross_sections.get_yscale() == "log"
    for label, name in (
        ("Total", "total"),
        ("Absorption", "absorption"),
        ("Fission", "fission"),
        ("Nu-fission", "nu_fission"),
    ):
        plotted = labelled_line(cross_sections, label).get_ydata()
        expected = np.insert(original[name], 0, original[name][0])
        positive = expected > 0.0
        np.testing.assert_array_equal(
            plotted[positive],
            expected[positive],
        )
        assert np.all(np.isnan(plotted[~positive]))

    chi_axis = figures["chi"].axes[0]
    assert chi_axis.get_xscale() == "log"
    assert chi_axis.get_yscale() == "linear"
    assert any(line.get_visible() for line in chi_axis.get_xgridlines())
    assert any(line.get_visible() for line in chi_axis.get_ygridlines())
    assert len(chi_axis.patches) == 0
    chi_line = chi_axis.lines[0]
    np.testing.assert_array_equal(chi_line.get_xdata(), mgxs.energy_bounds_ev)
    np.testing.assert_array_equal(
        chi_line.get_ydata(),
        np.append(original["chi"], original["chi"][-1]),
    )
    assert chi_line.get_drawstyle() == "steps-post"
    assert not any(
        issubclass(item.category, RuntimeWarning)
        or "non-positive" in str(item.message).lower()
        or "no positive" in str(item.message).lower()
        for item in caught
    )

    for name, values in original.items():
        np.testing.assert_array_equal(getattr(mgxs, name), values)

    plt.close("all")


def test_p0_scattering_uses_log_color_and_masks_zero_without_mutation():
    scientific_matrix = np.array(
        [
            [1.0e-12, 0.0, 1.0e-9],
            [1.0e-6, 1.0e-3, 0.0],
            [1.0e2, 1.0, 1.0e-1],
        ]
    )
    mgxs = MGXS(
        np.array([1.0, 10.0, 100.0, 1000.0]),
        np.ones(3),
        np.ones(3),
        scientific_matrix[None, :, :],
        "log_scatter",
        294.0,
    )
    original_scatter = mgxs.scatter.copy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figures = plot_mgxs(mgxs)

    matrix_axis = figures["scatter_p0"].axes[0]
    mesh = matrix_axis.collections[0]
    displayed = mesh.get_array().reshape(3, 3)
    expected = scientific_matrix[::-1, ::-1].T

    assert isinstance(mesh.norm, LogNorm)
    assert mesh.norm.vmin == pytest.approx(1.0e-12)
    assert mesh.norm.vmax == pytest.approx(1.0e2)
    np.testing.assert_array_equal(np.ma.getdata(displayed), expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(displayed), expected <= 0.0)
    assert displayed[0, 0] == scientific_matrix[-1, -1]
    assert displayed[-1, -1] == scientific_matrix[0, 0]

    assert matrix_axis.get_xscale() == matrix_axis.get_yscale() == "linear"
    assert matrix_axis.get_xlim() == pytest.approx((0.5, 3.5))
    assert matrix_axis.get_ylim() == pytest.approx((3.5, 0.5))
    assert any(line.get_visible() for line in matrix_axis.get_xgridlines())
    assert any(line.get_visible() for line in matrix_axis.get_ygridlines())
    assert "(log scale)" in figures["scatter_p0"].axes[1].get_ylabel()
    assert not any(
        issubclass(item.category, RuntimeWarning)
        or "divide by zero" in str(item.message).lower()
        for item in caught
    )
    np.testing.assert_array_equal(mgxs.scatter, original_scatter)

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
    displayed = mesh.get_array().reshape(2, 2)
    np.testing.assert_array_equal(displayed, expected[::-1, ::-1].T)
    assert displayed[0, 0] == expected[-1, -1]
    assert displayed[-1, -1] == expected[0, 0]
    matrix_axis = figures["fission_matrix"].axes[0]
    chi_axis = figures["chi"].axes[0]
    assert chi_axis.get_xscale() == "log"
    assert chi_axis.get_yscale() == "linear"
    assert matrix_axis.get_xlim() == pytest.approx((0.5, 2.5))
    assert matrix_axis.get_ylim() == pytest.approx((2.5, 0.5))
    assert matrix_axis.get_xscale() == matrix_axis.get_yscale() == "linear"
    np.testing.assert_array_equal(mgxs.nu_fission, nu_fission)
    np.testing.assert_array_equal(mgxs.chi, chi)

    plt.close("all")


def test_large_group_matrix_uses_readable_integer_tick_subset():
    groups = 361
    mgxs = MGXS(
        np.arange(1.0, groups + 2.0),
        np.ones(groups),
        np.ones(groups),
        np.zeros((1, groups, groups)),
        "shem_like",
        294.0,
    )

    figures = plot_mgxs(mgxs)

    matrix_axis = figures["scatter_p0"].axes[0]
    x_ticks = matrix_axis.get_xticks()
    y_ticks = matrix_axis.get_yticks()
    assert 2 < len(x_ticks) <= 9
    np.testing.assert_array_equal(x_ticks, y_ticks)
    assert x_ticks[0] == 1
    assert x_ticks[-1] == groups
    assert np.all(x_ticks == x_ticks.astype(int))

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
