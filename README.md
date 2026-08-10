# Generate MGXS

`generate_mgxs` is a small, explicit bridge between OpenMC and OpenSn for
managed multigroup calculations. It supports fixed-source and k-eigenvalue
operation, homogeneous OpenMC/OpenSn verification cases, and a deliberately
limited OpenMC-only concentric-cylinder geometry. Python-facing energy arrays
always run from low to high physical energy. OpenMC MGXS arrays and OpenSn group
numbering are converted at their boundaries only.

Define a `Case`, then prepare an independent run directory:

```python
from pathlib import Path

from generate_mgxs import prepare
from my_case import CASE

run_path = prepare(CASE, Path("results/material_001"))
```

By default, `prepare()` writes `_metadata/run.json`, `openmc/model.py`, and
`opensn/input.py`; it never starts a subprocess. This makes a plain generation
loop sufficient for one or one hundred cases, with no campaign state.

Energy groups may be a canonical OpenMC structure name or explicit custom
boundaries:

```python
named_case = Case(
    ...,
    energy_groups="SHEM-361",
)

custom_case = Case(
    ...,
    energy_groups=LANL30_BOUNDS_EV,
)
```

Named structures are validated and resolved lazily from
`openmc.mgxs.GROUP_STRUCTURES`; this repository does not copy OpenMC's standard
boundary tables. In both forms, `case.energy_bounds_ev` contains the resolved
ascending numerical boundaries used by source integration, OpenSn, direct
solutions, and result comparison.

For OpenMC-only MGXS production, select only the OpenMC input:

```python
run_path = prepare(
    CASE,
    Path("results/material_001"),
    solvers=("openmc",),
)
```

The managed cylindrical form consists only of ordered radial cell records, a
finite axial height, boundary conditions, and an optional outer rectangular
prism. Each cell declares its XSdata name explicitly, so two cells may share a
physical material without sharing an MGXS dataset:

```python
geometry = ConcentricGeometry(
    regions=(
        ConcentricCell("Inner", material_a, "inner", 1.0),
        ConcentricCell("Shell", material_b, "shell", 2.0),
    ),
    height_cm=4.0,
    outer_radial_boundary="reflective",
)
```

Concentric cases must be prepared with `solvers=("openmc",)`; requesting an
OpenSn input fails during preparation because OpenSn does not recreate this
cylindrical CSG model.

The resulting directory contains `openmc/model.py` and `_metadata/run.json`,
with no OpenSn directory. This is useful for large material libraries,
parameter sweeps, external shell scripts, SLURM array jobs, and any workflow
that only needs OpenMC MGXS production:

```python
for case in cases:
    run_path = prepare(
        case,
        results_root / case.name,
        solvers=("openmc",),
    )
```

Each directory is independently executable later. Set portable environment
variables in the interactive shell, batch script, or SLURM job:

```bash
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
export OPENSN_CONSOLE=/path/to/opensn-console
export OPENSN_MPIEXEC=/path/to/opensn-mpiexec  # only needed for MPI
```

Material composition uses one simple sequence for explicit nuclides and
natural elements:

```python
be9 = Material("be9", "Be-9", 1.85, (("Be9", 1.0),))
iron = Material("iron", "natural iron", 7.87, (("Fe", 1.0),))
steel = Material("steel", "Fe-C", 7.8, (("Fe", 0.98), ("C", 0.02)))
uranium = Material(
    "fuel", "uranium", 18.823124,
    (("U234", 2.5759e-6), ("U235", 3.4428e-4), ("U238", 4.7441e-2)),
)
```

A mass number selects an explicit nuclide; a bare symbol delegates natural
isotope expansion to OpenMC. The package carries no periodic-table or natural-
abundance database. Composition values are nonnegative relative atomic amounts;
they need not sum to one and are passed unchanged to OpenMC with `percent_type="ao"`.

For `run_dir=results/material_001`, the direct serial command contract is:

```bash
(cd "$run_dir/openmc" && python model.py run)
(cd "$run_dir/openmc" && python model.py process)
"$OPENSN_CONSOLE" -i "$run_dir/opensn/input.py"
```

The OpenMC commands need an OpenMC-capable Python environment and
`OPENMC_CROSS_SECTIONS`. The processing command reads
`openmc/statepoint.<batches>.h5` and produces `openmc/mgxs.h5`,
`openmc/openmc_result.json`, and `diagnostics/mgxs_uncertainty.json`. OpenSn
then reads the run-relative `openmc/mgxs.h5` and writes
`opensn/opensn_result.json`. For two ranks, use:

```bash
"$OPENSN_MPIEXEC" -n 2 "$OPENSN_CONSOLE" -i "$run_dir/opensn/input.py"
```

Python execution helpers remain optional conveniences:

```python
from generate_mgxs import run_openmc, run_opensn

run_openmc(run_path, cross_sections="/path/to/cross_sections.xml")
run_opensn(run_path, executable="/path/to/opensn-console")
```

MGXS plotting is separate scientific postprocessing and does not execute a
solver or read an OpenMC statepoint:

```python
from generate_mgxs import load_mgxs, plot_mgxs

mgxs = load_mgxs(run_path / "openmc/mgxs.h5", "be9", 294.0)
figures = plot_mgxs(
    mgxs,
    output_directory=run_path / "plots",
    scatter_moments=(0,),
)
```

The returned dictionary contains the applicable cross-section, chi,
scattering-moment, and derived fission-production figures. Energy axes use the
package's ascending-eV convention; plotting never requires OpenSn.

Complete, commented Be-9, UO2-in-HDPE, homogeneous FlatTop, detector, and
Pu9+HDPE definitions are in `examples/`. The detector and Pu9+HDPE cases use
the OpenMC-only concentric geometry. FlatTop uses `run_mode="eigenvalue"`; its
direct comparison is:

```python
direct = solve_infinite_medium_eigenvalue(mgxs)
print(openmc_result.k_eff, direct.k_eff, opensn_result.k_eff)
```

Eigenvectors are compared through their independently normalized spectrum
shapes; raw OpenMC tally values and uncertainty remain available on
`openmc_result.spectrum`.

Their run scripts use `OPENMC_CROSS_SECTIONS` and `OPENSN_CONSOLE`; no
machine-specific path is embedded. The generated Python files are readable
solver inputs, not wrappers around hidden configuration or in-memory `Case`
objects. OpenMC MGXS HDF5 is the only supported cross-section handoff.

## OpenSn verification scope

Generated OpenSn input is intentionally a verification calculation, not a
second model of the OpenMC geometry. Each material domain in openmc/mgxs.h5
is loaded and solved independently in a homogeneous, all-reflecting 2 cm cube
with 2 cells per axis (8 cells total). The fixed
GLCProductQuadrature3DXYZ(2, 4) supplies 8 directions, and P0 scattering is
sufficient for the isotropic infinite-homogeneous scalar-flux comparison.
Fixed-source cases use the existing steady-state source solver. A homogeneous
eigenvalue case instead uses `PowerIterationKEigenSolver` with no external
volumetric source and reports normalized flux shape, k-effective, power
iterations, sweeps, balance, and final relative k change.

OpenMC target/moderator dimensions and boundary choices therefore do not
control OpenSn cost. scattering_order remains a Case setting because it
controls the MGXS moments produced by OpenMC; the OpenSn verifier deliberately
uses only P0. The only case-specific OpenSn controls are GMRES tolerance,
maximum iterations, and restart.
