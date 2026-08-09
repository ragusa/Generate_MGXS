# Generate MGXS

`generate_mgxs` is a small, explicit bridge between OpenMC and OpenSn for
fixed-source multigroup calculations. Python-facing energy arrays always run
from low to high physical energy. OpenMC MGXS arrays and OpenSn group numbering
are converted at their boundaries only.

Define a `Case`, then prepare an independent run directory:

```python
from generate_mgxs import prepare
from my_case import CASE

run_directory = prepare(CASE, "results/material_001")
```

`prepare()` only writes `_metadata/run.json`, `openmc/model.py`, and
`opensn/input.py`; it never starts a subprocess. This makes a plain generation
loop sufficient for one or one hundred cases, with no campaign state:

```python
for case in cases:
    prepare(case, results_root / case.name)
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
```

A mass number selects an explicit nuclide; a bare symbol delegates natural
isotope expansion to OpenMC. The package carries no periodic-table or natural-
abundance database.

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

run_openmc(run_directory, cross_sections="/path/to/cross_sections.xml")
run_opensn(run_directory, executable="/path/to/opensn-console")
```

Complete, commented Be-9 and UO2-in-HDPE definitions are in `examples/`.
Their run scripts use `OPENMC_CROSS_SECTIONS` and `OPENSN_CONSOLE`; no
machine-specific path is embedded. The generated Python files are readable
solver inputs, not wrappers around hidden configuration or in-memory `Case`
objects. OpenMC MGXS HDF5 is the only supported cross-section handoff.
