"""Small spectrum-comparison plotting helper."""

from __future__ import annotations


def plot_spectra(*spectra, labels=None, path=None, normalize=True):
    """Plot stepwise spectra against ascending energy boundaries."""
    import matplotlib.pyplot as plt

    labels = labels or [
        spectrum.logical_domain or f"spectrum {index + 1}"
        for index, spectrum in enumerate(spectra)
    ]

    figure, axis = plt.subplots()

    # stairs() consumes one more boundary than values, exactly matching the
    # package's group-integrated spectrum representation.
    for spectrum, label in zip(spectra, labels):
        values = spectrum.normalized if normalize else spectrum.values
        axis.stairs(values, spectrum.energy_bounds_ev, label=label)

    axis.set_xscale("log")
    axis.set_xlabel("Energy (eV)")
    axis.set_ylabel("Normalized group flux" if normalize else "Group-integrated flux")
    axis.legend()
    figure.tight_layout()

    if path is not None:
        figure.savefig(path)

    return figure, axis
