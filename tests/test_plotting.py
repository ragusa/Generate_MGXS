import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from generate_mgxs import MGXS, plot_mgxs


def synthetic_mgxs(*, fissionable=False):
    """Create asymmetric two-group data so plotting orientation is observable."""
    energy_bounds = np.array([1.0, 10.0, 100.0])
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
