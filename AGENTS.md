# Generate_MGXS agent instructions

## Execution environment

Use the existing Ubuntu 22.04 WSL installation as Linux user `ragusa`.
Do not use Windows Python, Windows OpenMC, or Windows-side path conversions.

The main repository is:

```text
/home/ragusa/repo/Generate_MGXS
```

External case runs and their potentially large outputs live under:

```text
/home/ragusa/work/Generate_MGXS/cases
```

If the controlling shell starts on Windows, enter WSL with:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc '<Linux command>'
```

Do not repeatedly try alternative Windows launch mechanisms.

## OpenMC environment

Use this exact interpreter without activating or modifying the Conda
environment:

```text
/home/ragusa/miniforge3/envs/openmc-env/bin/python
```

Set this environment before running repository tests, examples, or OpenMC
case runners:

```bash
export PYTHONPATH=/home/ragusa/repo/Generate_MGXS${PYTHONPATH:+:$PYTHONPATH}
export OPENMC_CROSS_SECTIONS=/home/ragusa/xs/endfb-viii.0-hdf5/cross_sections.xml
export OPENSN_CONSOLE=/home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console
export OMP_NUM_THREADS=40
```

Do not set `MGXS_PROCESSES=1`.

## Tests in the main repository

Run focused plotting and generation tests with:

```bash
cd /home/ragusa/repo/Generate_MGXS
/home/ragusa/miniforge3/envs/openmc-env/bin/python -m pytest \
    tests/test_plotting.py \
    tests/test_generation_and_results.py
```

Run the repository suite without solver execution tests with:

```bash
cd /home/ragusa/repo/Generate_MGXS
/home/ragusa/miniforge3/envs/openmc-env/bin/python -m pytest \
    --ignore=tests/test_execution.py
```

Run execution tests only when solver execution is explicitly in scope.

## Running examples and work cases

Repository examples can be run directly from their example directory:

```bash
cd /home/ragusa/repo/Generate_MGXS/examples/<case>
/home/ragusa/miniforge3/envs/openmc-env/bin/python run.py
```

Prefer the external work area for production or exploratory runs whose outputs
should remain outside the Git repository:

```bash
cd /home/ragusa/work/Generate_MGXS/cases/<case>
/home/ragusa/miniforge3/envs/openmc-env/bin/python run.py
```

A case `run.py` is a full workflow. Depending on the case, it can prepare
inputs, run OpenMC, process results, run another solver, and generate plots.
Inspect the runner before executing it and do not assume it is plot-only.

The generated OpenMC calculation can also be run explicitly from its generated
directory:

```bash
cd <run-directory>/openmc
/home/ragusa/miniforge3/envs/openmc-env/bin/python model.py run
/home/ragusa/miniforge3/envs/openmc-env/bin/python model.py process
```

## Plot-only tasks

When asked to regenerate plots from existing results, do not execute the case
`run.py`, `prepare()`, `run_openmc()`, `model.py run`, or `model.py process`.
Load the existing `openmc/openmc_result.json` and `openmc/mgxs.h5`, then invoke
only the applicable plotting functions.

For multi-domain cases such as `detector`, regenerate both the per-domain MGXS
plots and the all-domain flux plots using `load_openmc_domain_spectra()` and
`plot_openmc_domain_spectra()`.

## OpenSn environment

OpenMC and OpenSn must run as separate processes with separate environments.
Never import OpenMC and `pyopensn` in the same Python process.

Do not activate the OpenMC Conda environment for OpenSn and do not source an
OpenSn activation script. Run the installed wrapper directly; it establishes
its own child-local runtime environment:

```bash
cd <run-directory>/opensn
/home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console -i input.py
```

To preserve logs for a manual OpenSn run:

```bash
cd <run-directory>/opensn
/home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console -i input.py \
    > ../logs/opensn_manual.stdout \
    2> ../logs/opensn_manual.stderr
```

The expected result is `opensn/opensn_result.json` under the run directory.

## Monitoring and logs

Useful OpenMC checks from a case directory are:

```bash
tail -f run/logs/openmc_run.stdout
ps -eo pid,ppid,etime,nlwp,%cpu,%mem,cmd \
    | grep -E '[o]penmc|[m]odel.py|[p]ython run.py'
ps -o pid,nlwp,etime,%cpu,%mem,cmd -C openmc
ls -lh run/openmc/statepoint.*.h5
```

Important OpenMC logs are:

```text
run/logs/openmc_run.stdout
run/logs/openmc_run.stderr
run/logs/openmc_process.stdout
run/logs/openmc_process.stderr
```

Useful OpenSn checks are:

```bash
ps -eo pid,ppid,etime,%cpu,%mem,cmd | grep -E '[o]pensn'
tail -f run/logs/opensn.stdout
tail -f run/logs/opensn_manual.stdout
```

## Numerical and output safeguards

Do not change group structures, sources, tolerances, particle counts, batch
counts, or other numerical settings unless explicitly authorized. Preserve
existing output directories instead of deleting them.

## Work-area note: FlatTop snapshot (2026-08-10)

The external FlatTop case currently uses 300,000 particles per batch, 520
batches, and 120 inactive batches. Its 156-million-history OpenMC calculation
completed with `k_eff = 0.43224 +/- 0.00004`.

The full 30-group direct eigenvalue solve is singular because groups 0 and 1
have structurally zero rows and columns. The remaining 28-group subsystem is
full rank and gives `k_eff = 0.43222843706920405`. A manual OpenSn run converged
to `k_eff = 0.4322284`; OpenSn sees the same empty groups as groups 28 and 29
because its group ordering is reversed.

Relevant work artifacts are under:

```text
/home/ragusa/work/Generate_MGXS/cases/flattop/run
```

Before relying on this snapshot, verify the current work-area files and output
timestamps. The likely repository fix is for the direct eigenvalue solver to
remove structurally disconnected zero rows and columns, solve the active
subsystem, and reconstruct zero flux in excluded groups.
