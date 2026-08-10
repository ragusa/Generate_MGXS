Use the existing Ubuntu 22.04 WSL installation under Linux user `ragusa`. Do not use Windows Python, Windows OpenMC, or Windows-side path conversions.

Repository:
  /home/ragusa/repo/Generate_MGXS

Work area:
  /home/ragusa/work/Generate_MGXS/cases

OpenMC and OpenSn must run as separate processes with separate environments. Never import OpenMC and pyopensn in the same Python process.

OPENMC
======

Use this exact interpreter:
  /home/ragusa/miniforge3/envs/openmc-env/bin/python

Required environment:
  export OPENMC_CROSS_SECTIONS=/home/ragusa/xs/endfb-viii.0-hdf5/cross_sections.xml
  export PYTHONPATH=/home/ragusa/repo/Generate_MGXS${PYTHONPATH:+:$PYTHONPATH}
  export OMP_NUM_THREADS=40
  export OPENSN_CONSOLE=/home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console

Do not modify or activate the Conda environment. Do not set MGXS_PROCESSES=1.

To run a case:
  cd /home/ragusa/work/Generate_MGXS/cases/<case>
  /home/ragusa/miniforge3/envs/openmc-env/bin/python run.py

For example, FlatTop:
  cd /home/ragusa/work/Generate_MGXS/cases/flattop
  export OPENMC_CROSS_SECTIONS=/home/ragusa/xs/endfb-viii.0-hdf5/cross_sections.xml
  export PYTHONPATH=/home/ragusa/repo/Generate_MGXS${PYTHONPATH:+:$PYTHONPATH}
  export OMP_NUM_THREADS=40
  export OPENSN_CONSOLE=/home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console
  /home/ragusa/miniforge3/envs/openmc-env/bin/python run.py

If the controlling shell starts on Windows, enter WSL with:
  wsl -d Ubuntu-22.04 -- bash -lc '<Linux command>'

Do not repeatedly try alternative Windows launch mechanisms.

OpenMC monitoring:
  tail -f run/logs/openmc_run.stdout

  ps -eo pid,ppid,etime,nlwp,%cpu,%mem,cmd \
    | grep -E '[o]penmc|[m]odel.py|[p]ython run.py'

  ps -o pid,nlwp,etime,%cpu,%mem,cmd -C openmc

  ls -lh run/openmc/statepoint.*.h5

Important OpenMC logs:
  run/logs/openmc_run.stdout
  run/logs/openmc_run.stderr
  run/logs/openmc_process.stdout
  run/logs/openmc_process.stderr

OPENSN
======

Use the installed wrapper directly:
  /home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console

Do not activate the OpenMC Conda environment for OpenSn.
Do not source an OpenSn activation script.
The wrapper sets its own child-local runtime environment.

To run a generated OpenSn input manually:
  cd /home/ragusa/work/Generate_MGXS/cases/<case>/run/opensn
  /home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console -i input.py

To preserve manual OpenSn logs:
  cd /home/ragusa/work/Generate_MGXS/cases/<case>/run/opensn
  /home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console -i input.py \
    > ../logs/opensn_manual.stdout \
    2> ../logs/opensn_manual.stderr

OpenSn monitoring:
  ps -eo pid,ppid,etime,%cpu,%mem,cmd | grep -E '[o]pensn'
  tail -f run/logs/opensn.stdout
  tail -f run/logs/opensn_manual.stdout

Expected result:
  run/opensn/opensn_result.json

CURRENT FLATTOP STATE
=====================

The work-area file:
  /home/ragusa/work/Generate_MGXS/cases/flattop/case.py

currently contains:
  particles_per_batch=300_000
  batches=520
  inactive_batches=120

This calculation completed successfully:
  Total histories: 156 million
  Active histories: 120 million
  OpenMC elapsed time: approximately 3345 seconds
  OpenMC combined k_eff: 0.43224 +/- 0.00004
  Statepoint: run/openmc/statepoint.520.h5
  MGXS: run/openmc/mgxs.h5

However, `run.py` exits at the direct eigenvalue solver before it can invoke OpenSn or plotting:
  ValueError: eigenvalue loss operator is singular

This is now well diagnosed:
  Direct loss matrix size: 30 x 30
  Rank: 28
  Exactly zero OpenMC groups: 0 and 1
  Rows and columns 0 and 1 are entirely zero
  The two smallest singular values are exactly zero

The remaining active subsystem is healthy:
  Active matrix size: 28 x 28
  Active matrix rank: 28
  Condition number: 34.35
  Reduced direct k_eff: 0.43222843706920405
  Reduced direct residual: 1.87e-18

The generated OpenSn calculation was run manually and succeeded:
  OpenSn k_eff: 0.4322284
  Power iterations: 2
  Final k_eff change: 2.487563e-12
  Sweeps: 108
  Balance: 8.003553e-11
  Exit code: 0

OpenSn sees the same zero groups as groups 28 and 29 because its energy-group ordering is reversed. It emits near-zero transport warnings, applies its built-in handling, and converges.

Relevant logs:
  /home/ragusa/work/Generate_MGXS/cases/flattop/run/logs/openmc_run.stdout
  /home/ragusa/work/Generate_MGXS/cases/flattop/run/logs/opensn_manual.stdout

Relevant outputs:
  /home/ragusa/work/Generate_MGXS/cases/flattop/run/openmc/statepoint.520.h5
  /home/ragusa/work/Generate_MGXS/cases/flattop/run/openmc/mgxs.h5
  /home/ragusa/work/Generate_MGXS/cases/flattop/run/openmc/openmc_result.json
  /home/ragusa/work/Generate_MGXS/cases/flattop/run/opensn/opensn_result.json

Conclusion:
  Increasing OpenMC histories does not eliminate the final two structurally empty groups.
  The full direct solver still fails because it attempts to solve the unreduced 30-group matrix.
  OpenSn handles the degenerate groups and converges.
  The smallest likely repository fix is for the direct eigenvalue solver to remove structurally disconnected zero rows/columns, solve the active subsystem, and reconstruct zero flux in the excluded groups.

Do not change the group structure, source, tolerances, or other numerical settings unless explicitly authorized. Preserve existing output directories instead of deleting them.