# Generate MGXS

`generate_mgxs` is a small, explicit bridge between OpenMC and OpenSn for
managed multigroup calculations. It prepares readable solver inputs, runs the
solvers only when explicitly requested, loads their compact results, provides
homogeneous direct checks, and plots spectra and multigroup cross sections.

The package supports:

- fixed-source and k-eigenvalue OpenMC calculations;
- homogeneous one-material and explicit nested target/moderator box models;
- an intentionally limited OpenMC-only concentric-cylinder geometry;
- OpenSn homogeneous verification calculations;
- direct homogeneous fixed-source and rank-one eigenvalue checks; and
- per-domain MGXS and spectrum plotting.

Python-facing energy arrays always run from low to high physical energy.
OpenMC MGXS arrays and OpenSn group numbering are converted only at their
boundaries. Scattering data use `scatter[moment, g_in, g_out]` ordering.

## Requirements

The Python package requires Python 3.10 or newer, NumPy, and h5py. Install the
repository in editable mode with:

```bash
python -m pip install -e .
```

Plotting and test dependencies are optional:

```bash
python -m pip install -e ".[plot]"
python -m pip install -e ".[plot,test]"
```

OpenMC is an external scientific runtime rather than a package dependency.
Resolving a named OpenMC energy-group structure and executing a generated
OpenMC model need an OpenMC-capable Python environment. Preparing a case with
explicit energy boundaries is deterministic file generation and does not start
or import OpenMC. OpenMC transport also needs a compatible
`cross_sections.xml` nuclear-data library.

OpenSn is likewise external. Generated OpenSn inputs must be run with an
installed `opensn-console` wrapper. OpenMC and OpenSn are separate processes;
do not import OpenMC and `pyopensn` into the same Python process or activate an
OpenMC environment around an OpenSn process that supplies its own runtime.

## Define and prepare a case

Define a `Case`, then prepare an independent run directory:

```python
from pathlib import Path

from generate_mgxs import Case, Material, prepare

material = Material(
    "be9",
    "Be-9",
    1.85,
    (("Be9", 1.0),),
)
case = Case(
    name="be9",
    materials=(material,),
    energy_groups=(1.0e-5, 1.0e6, 2.0e7),
    target_dimensions_cm=(2.0, 2.0, 2.0),
    source_kind="uniform_energy",
)

run_path = prepare(case, Path("results/be9"))
```

`prepare()` never starts a subprocess. For a homogeneous one-material case, it
writes independent OpenMC and OpenSn inputs plus provenance metadata by
default. OpenMC-only preparation is explicit:

```python
run_path = prepare(
    case,
    Path("results/be9"),
    solvers=("openmc",),
)
```

Prepared directories do not depend on the original in-memory `Case` and can be
submitted later through a shell script, scheduler, or Python helper. There is
no campaign object or shared mutable run state.

## Energy groups

Energy groups may be a canonical OpenMC structure name or explicit ascending
boundaries in eV:

```python
named_case = Case(
    ...,
    energy_groups="SHEM-361",
)

custom_case = Case(
    ...,
    energy_groups=(1.0e-5, 0.625, 20.0e6),
)
```

Named structures are resolved lazily from `openmc.mgxs.GROUP_STRUCTURES`; the
repository does not copy OpenMC's standard tables. The custom `WIMS69`,
`LANL30`, and `LANL70` definitions are available through `energy_bounds()` and
are passed to `Case` as explicit boundaries:

```python
from generate_mgxs import energy_bounds

lanl70_case = Case(
    ...,
    energy_groups=energy_bounds("LANL70"),
)
```

In every case, `case.energy_bounds_ev` contains the resolved ascending
numerical boundaries used by source integration, OpenMC tallying, direct
solutions, result comparison, and homogeneous OpenSn verification when
requested.

## Materials and geometry

Material composition is a sequence of explicit nuclides or natural elements:

```python
be9 = Material("be9", "Be-9", 1.85, (("Be9", 1.0),))
iron = Material("iron", "natural iron", 7.87, (("Fe", 1.0),))
steel = Material("steel", "Fe-C", 7.8, (("Fe", 0.98), ("C", 0.02)))
uranium = Material(
    "fuel",
    "uranium",
    18.823124,
    (("U234", 2.5759e-6), ("U235", 3.4428e-4), ("U238", 4.7441e-2)),
)
```

A mass number selects an explicit nuclide; a bare symbol delegates natural
isotope expansion to OpenMC. Composition values are nonnegative relative atomic
amounts. They need not sum to one and are passed unchanged to OpenMC with
`percent_type="ao"`.

Every nonhomogeneous case declares its geometry explicitly. A target box inside
a larger moderator box uses `NestedBoxGeometry`:

```python
from generate_mgxs import NestedBoxGeometry

geometry = NestedBoxGeometry(
    target=target_material,
    moderator=moderator_material,
    target_dimensions_cm=(0.4, 0.4, 0.4),
    outer_dimensions_cm=(1.5, 1.5, 1.5),
    boundaries=("reflective",) * 6,
)
```

The two materials use `role="target"` and `role="moderator"`, respectively,
and both are listed in the enclosing `Case.materials`. Dimensions and boundary
conditions belong to the explicit geometry rather than to `Case`.

The managed concentric geometry consists of ordered radial cells, a finite
axial height, boundary conditions, and an optional surrounding rectangular
prism. Every cell declares its logical XSdata name, so cells can share a
physical material without aliasing their MGXS datasets:

```python
from generate_mgxs import ConcentricCell, ConcentricGeometry

geometry = ConcentricGeometry(
    regions=(
        ConcentricCell("Inner", material_a, "inner", 1.0),
        ConcentricCell("Shell", material_b, "shell", 2.0),
    ),
    height_cm=4.0,
    outer_radial_boundary="reflective",
)
```

All nonhomogeneous cases are OpenMC-only and must use
`solvers=("openmc",)`. OpenSn and the direct solvers are intentionally limited
to homogeneous one-material verification problems.

## Run the generated workflow

Set runtime locations in the shell or scheduler environment:

```bash
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
export OPENSN_CONSOLE=/path/to/opensn-console
```

Generated OpenMC input is directly executable:

```bash
cd results/be9/openmc
python model.py run
python model.py process
```

Run OpenSn separately from its generated directory:

```bash
cd results/be9/opensn
"$OPENSN_CONSOLE" -i input.py
```

The equivalent optional Python helpers are:

```python
from generate_mgxs import run_openmc, run_opensn

run_openmc(
    run_path,
    cross_sections="/path/to/cross_sections.xml",
    operation="all",
)
run_opensn(
    run_path,
    executable="/path/to/opensn-console",
)
```

For MPI OpenSn execution, supply both `ranks` and `mpi_executable` to
`run_opensn()`, or invoke the installed MPI wrapper manually.

## Run-directory products

A completed two-solver run normally contains:

```text
run/
├── _metadata/run.json
├── diagnostics/mgxs_uncertainty.json
├── logs/
│   ├── openmc_run.stdout
│   ├── openmc_run.stderr
│   ├── openmc_process.stdout
│   ├── openmc_process.stderr
│   ├── opensn.stdout
│   └── opensn.stderr
├── openmc/
│   ├── model.py
│   ├── statepoint.<batches>.h5
│   ├── mgxs.h5
│   └── openmc_result.json
└── opensn/
    ├── input.py
    └── opensn_result.json
```

`model.py process` reads the OpenMC statepoint and produces `mgxs.h5`, the
compact OpenMC spectrum/result JSON, and the uncertainty diagnostics. OpenSn
then reads the run-relative `openmc/mgxs.h5` and writes a result only after
explicit convergence evidence is available.

## Results and plotting

Load one logical MGXS domain and generate all applicable plots:

```python
from generate_mgxs import load_mgxs, plot_mgxs

mgxs = load_mgxs(run_path / "openmc/mgxs.h5", "be9", 294.0)
figures = plot_mgxs(
    mgxs,
    output_directory=run_path / "plots",
    scatter_moments=(0,),
)
```

Depending on fissionability, the returned figures include macroscopic
cross-section curves, chi, scattering matrices, and a derived fission-production
matrix. MGXS plots use logarithmic physical-energy axes where appropriate and
include grids for readability.

`plot_spectra()` compares only explicitly selected available solutions. Do not
substitute dummy spectra for a solver that was not run:

```python
from generate_mgxs import load_openmc_result, load_opensn_result, plot_spectra

openmc = load_openmc_result(run_path)
opensn = load_opensn_result(run_path)
plot_spectra(
    openmc,
    opensn.spectrum,
    None,
    include=("openmc", "opensn"),
    output_directory=run_path / "plots",
)
```

For OpenMC multi-domain cases, use `load_openmc_domain_spectra()` and
`plot_openmc_domain_spectra()` to compare independently normalized domain
shapes in declared geometry order.

Plotting is postprocessing only: it does not run a solver, process a statepoint,
or regenerate MGXS data.

## Direct-solver scope

`solve_infinite_medium()` is a verification solve for non-fissionable,
homogeneous fixed-source MGXS. `solve_infinite_medium_eigenvalue()` implements
the corresponding homogeneous factorized-fission eigenvalue check. Neither is
a general geometry solver.

The direct eigenvalue solver currently requires a nonsingular full-group loss
operator. The 30-group FlatTop result has two structurally disconnected zero
groups, so its direct call remains visible but commented out in
`examples/flattop/run.py`. OpenMC and OpenSn eigenvalue comparison remains
active. Removing inactive groups and reconstructing their zero flux is a
possible future extension, not current behavior.

## Example runners

The `examples/` directory contains complete case definitions and intentionally
simple orchestration scripts:

| Example | Geometry/mode | Active workflow |
| --- | --- | --- |
| `be9` | Homogeneous fixed source | OpenMC, direct, OpenSn, plots |
| `hdpe` | Homogeneous fixed source | OpenMC, direct, OpenSn, plots |
| `moderated` | LANL70 UO2 box inside an HDPE box | OpenMC-only, per-domain MGXS plots |
| `flattop` | Homogeneous eigenvalue | OpenMC, OpenSn, plots; direct call commented out |
| `detector` | Concentric fixed source | OpenMC-only, per-domain and all-domain plots |
| `pu9_hdpe` | Concentric eigenvalue | OpenMC-only, per-domain plots |

Each `run.py` is hand-maintained source, not generated code. The runners are
deliberately fail-fast and easy to edit: an exception from a stage stops that
script. `prepare()` generates only solver inputs and metadata inside the chosen
run directory.

Run an example from its own directory after setting the required environment:

```bash
cd examples/be9
python run.py
```

## OpenSn verification scope

Generated OpenSn input verifies one homogeneous material in an all-reflecting
2 cm cube with two cells per axis (eight cells total). A fixed product
quadrature supplies eight directions, and P0 scattering is used for the
isotropic infinite-homogeneous scalar-flux check. OpenSn input generation is
rejected for nested-box, concentric, and other nonhomogeneous cases.

Fixed-source cases use a steady-state source solver. Homogeneous eigenvalue
cases use `PowerIterationKEigenSolver` with no external volumetric source and
report normalized flux shape, k-effective, power iterations, sweeps, balance,
and final relative k change.

The homogeneous OpenMC box dimensions and boundary choices do not control the
fixed OpenSn verification cube. `scattering_order` remains a `Case` setting
because it controls MGXS moments produced by OpenMC; the OpenSn verifier
deliberately uses only P0. Case-specific OpenSn controls are GMRES tolerance,
maximum iterations, restart, and the eigenvalue convergence controls.

## Tests

Most tests are fast and use generated fixtures or mocked subprocesses. The
suite imports OpenMC to verify its named group structures, so run it from an
OpenMC-capable Python environment. Run everything except the external-runtime
execution module with:

```bash
python -m pytest --ignore=tests/test_execution.py
```

Run all tests with:

```bash
python -m pytest
```

The external-runtime paths used by `tests/test_execution.py` are configured by
environment variables:

```bash
export OPENMC_PYTHON=/path/to/openmc/python  # defaults to the pytest interpreter
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
export OPENSN_CONSOLE=/path/to/opensn-console
export OPENSN_MPIEXEC=/path/to/opensn-mpiexec
export OPENSN_FISSION_MGXS=/path/to/fissionable_fixture.h5
```

Tests skip when their required resources are not configured or unavailable.
The production examples are not launched by the unit-test suite.
