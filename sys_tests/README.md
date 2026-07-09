# Systematic Contamination Testing (`sys_tests/`)

This subdirectory contains specialized scripts for testing transverse systematic contamination effects on power-spectrum multipoles. The workflow decouples expensive clean baseline computation from fast contamination variants.

## Architecture

```
sys_tests/
├── compute_clean_baseline.py          # Compute clean multipoles once (reusable)
├── compute_contaminant_multipoles.py  # Add contamination, reuse clean baseline
├── config_clean_baseline.yaml         # Config for clean baseline (YAML)
├── config_contaminant_powerlaw.yaml   # Config for power-law contamination
├── config_contaminant_deltafn.yaml    # Config for delta-function contamination
├── cache/                             # Centralized baseline storage
└── README.md                          # This file
```


## current runs:

# default config file, first realization
"$PYBIN" compute_clean_baseline.py --nmock 10 --verbose --run-mode all

# through salloc/srun
salloc --exclusive --nodes=1 --cpus-per-task=128 --time=01:00:00
srun -n 128 "$PYBIN" compute_clean_baseline.py --run-mode all --verbose
exit

# Generate 10 contaminant-only realizations
"$PYBIN" compute_contaminant_only_multipoles.py \
  --base-config z0.1_0.4_nbar3e-04_nmesh256 \
  --sys-type power_law \
  --sys-amp 0.01 \
  --n-realizations 10

"$PYBIN" compute_contaminant_only_multipoles.py \
  --base-config z0.4_0.8_nbar3e-04_nmesh256 \
  --sys-type delta_function \
  --sys-ell-contam 6 \
  --sys-amp 0.01 \
  --n-realizations 10

# Plot the contaminant-only power spectra
"$PYBIN" plot_multipoles.py \
  --config-str z0.1_0.4_nbar3e-04_nmesh128 \
  --mode contaminant_only_powerlaw_alpha-2.0_amp0.0100

# Plot the contaminant-only power spectra
"$PYBIN" plot_multipoles.py \
  --config-str z0.1_0.4_nbar3e-04_nmesh512 \
  --mode clean_baseline

# compare different mu binning on all configs
cd /global/homes/r/rmfeder/desi_sys && source ~/.desi_bashrc && "$PYBIN" sys_tests/compute_pkmu_binning.py --config-label z0.1_0.4_nbar3e-04_nmesh256 --output-root sys_tests/results/pkmu_binning/z0.1_0.4_nbar3e-04_nmesh256/ --verbose

## Configuration: YAML + CLI Overrides

Rather than long command-line arguments, configuration is specified in **YAML files** with optional **CLI overrides**:

```bash
# Use default config file
"$PYBIN" compute_clean_baseline.py --verbose

# Override specific values from config
"$PYBIN" compute_clean_baseline.py --config my_config.yaml --nmock 20 --verbose

# Override without config file (falls back to defaults)
"$PYBIN" compute_clean_baseline.py --nmesh 256 --zmin 0.2 --zmax 0.5
```

### Config File Format

**`config_clean_baseline.yaml`** — Hardcoded defaults for clean baseline:
```yaml
redshift:
  zmin: 0.1
  zmax: 0.4

mesh:
  nmesh: 512

multipoles:
  ells: [0, 2, 4, 6, 8, 10, 12, 14, 16]

k_range:
  k_min: 0.006
  k_max: 0.2
  delta_k: 0.01

mocks:
  nmock: 5
  seed: 42
```

Edit these files to change defaults for your use case, then use `--config` to load custom versions.

## Workflow

### Step 1: Compute Clean Baseline (Once)

For a given redshift range and mesh resolution, compute the uncontaminated multipoles once:

```bash
cd sys_tests
source ~/.desi_bashrc

# Using default config (no target_nbar specified)
"$PYBIN" compute_clean_baseline.py --nmock 10 --verbose

# OR with custom config and target number density
"$PYBIN" compute_clean_baseline.py \
  --config my_config.yaml \
  --nmock 10 \
  --target-nbar 3e-4 \
  --verbose
```

**Outputs:**
- `cache/clean_baseline_z0.1_0.4.npz` — multipole data
- `cache/clean_baseline_z0.1_0.4.yaml` — reproducibility metadata
- `cache/rerun_clean_baseline_z0.1_0.4.sh` — re-run script

### Step 2: Compute Contamination Variants (Many)

Supports **two systematic types**: power-law and delta-function.

#### Option A: Power-Law Contamination

Power-law systematic: $P_{\text{sys}} \propto k^{\alpha}$ in a range of multipoles.

```bash
# Using default power-law config
"$PYBIN" compute_contaminant_multipoles.py --config config_contaminant_powerlaw.yaml --sys-amp 0.01 --verbose

# Override config parameters with target density
"$PYBIN" compute_contaminant_multipoles.py \
  --sys-type power_law \
  --sys-amp 0.01 \
  --sys-alpha -2.0 \
  --sys-ell-min 6 \
  --sys-ell-max 64 \
  --target-nbar 3e-4 \
  --nmock 10 \
  --verbose

# Different amplitude (same code path)
"$PYBIN" compute_contaminant_multipoles.py --sys-amp 0.05 --target-nbar 3e-4 --verbose
```

**Outputs:**
- `cache/contamination_additive_powerlaw_alpha-2.0_amp0.0100.npz`
- `cache/contamination_additive_powerlaw_alpha-2.0_amp0.0100.yaml`
- `cache/rerun_contamination_additive_powerlaw_alpha-2.0_amp0.0100.sh`

#### Option B: Delta-Function Contamination

Delta-function spike at specific multipole(s):

```bash
# Using default delta-function config
"$PYBIN" compute_contaminant_multipoles.py --config config_contaminant_deltafn.yaml --sys-amp 0.01 --verbose

# Override with CLI
"$PYBIN" compute_contaminant_multipoles.py \
  --sys-type delta_function \
  --sys-ell-contam 6 \
  --sys-amp 0.01 \
  --nmock 10 \
  --verbose

# Multiple ell values
"$PYBIN" compute_contaminant_multipoles.py \
  --sys-type delta_function \
  --sys-ell-contam 2 6 20 60 \
  --sys-amp 0.05 \
  --verbose
```

**Outputs:**
- `cache/contamination_additive_deltafn_ellcontam6_amp0.0100.npz`
- `cache/contamination_additive_deltafn_ellcontam6_amp0.0100.yaml`
- `cache/rerun_contamination_additive_deltafn_ellcontam6_amp0.0100.sh`

OR for multiple ells:
- `cache/contamination_additive_deltafn_ellcontam2_6_20_60_amp0.0500.npz`

### Cache Naming

Multipole files are clearly labeled by systematic type and parameters:

```
✓ Power-law:        contamination_additive_powerlaw_alpha{alpha}_amp{amp}.npz
✓ Delta-function:   contamination_additive_deltafn_ellcontam{ell}_amp{amp}.npz
```

This makes it easy to identify and batch process variants.

### Step 3: Apply Binning & Compare (Optional)

Use top-level `apply_binning.py` to convert multipoles → P(k,μ) with various binning strategies:

```bash
cd ..
source ~/.desi_bashrc

# Binning strategy 1
"$PYBIN" apply_binning.py \
  --multipole-file sys_tests/cache/clean_baseline_z0.1_0.4.npz \
  --n-clean-bins 12 \
  --mu-binning-strategy nonuniform \
  --output-dir sys_tests/binned_results/

# Binning strategy 2 (same multipoles, different binning)
"$PYBIN" apply_binning.py \
  --multipole-file sys_tests/cache/clean_baseline_z0.1_0.4.npz \
  --n-clean-bins 25 \
  --mu-binning-strategy nonuniform \
  --output-dir sys_tests/binned_results/

# Same for contaminated
"$PYBIN" apply_binning.py \
  --multipole-file sys_tests/cache/contamination_additive_powerlaw_alpha-2.0_amp0.0100.npz \
  --n-clean-bins 12 \
  --mu-binning-strategy nonuniform \
  --output-dir sys_tests/binned_results/
```

## Reproducibility

Each script generates three artifacts per run:

1. **NPZ file** — Data (multipoles, k-bins, metadata)
2. **YAML file** — Full configuration, parameters, and CLI args
3. **Rerun script** (`.sh`) — Exact bash command to reproduce results

### Example: Re-run a contamination computation

```bash
source ~/.desi_bashrc
bash cache/rerun_contamination_additive_powerlaw_alpha-2.0_amp0.0100.sh
```

## CLI Arguments

### `compute_clean_baseline.py`

```
--config CONFIG           Path to YAML config file (optional)
--zmin ZMIN              Redshift lower (overrides config)
--zmax ZMAX              Redshift upper (overrides config)
--nmock NMOCK            Number of mocks (overrides config)
--nmesh NMESH            Mesh resolution (overrides config)
--target-nbar NBAR       Target number density (overrides config, e.g. 3e-4)
--force                  Recompute even if cached
--verbose                Verbose output
```

### `compute_contaminant_multipoles.py`

```
--config CONFIG                       Path to YAML config file
--sys-type {power_law,delta_function} Systematic type (overrides config)
--sys-amp SYS_AMP                     Amplitude (overrides config)
--sys-alpha SYS_ALPHA                 Power-law slope (for power_law only)
--sys-ell-contam ELL [ELL ...]        Multipole spike(s) (for delta_function only)
--sys-ell-min SYS_ELL_MIN             Min ell (for power_law only)
--sys-ell-max SYS_ELL_MAX             Max ell (for power_law only)
--nmock NMOCK                         Number of mocks
--zmin ZMIN                           Redshift lower (overrides config)
--zmax ZMAX                           Redshift upper (overrides config)
--target-nbar NBAR                    Target number density (overrides config, e.g. 3e-4)
--clean-baseline-file FILE            Path to clean baseline NPZ
--output-dir DIR                      Output directory
--run-mode {all,remaining,INDICES}    Mock selection: 'all' (default), 'remaining', or comma-sep indices
--force                               Recompute even if cached
--verbose                             Verbose output
```

## Per-Realization Caching & Incremental Runs

Both scripts support **per-realization caching** — individual mock multipoles are saved separately and later aggregated. This enables incremental mock runs.

### Cache Directory Structure

```
cache/
├── config_z0.1_0.4_nbarcatmesh512/
│   ├── clean_baseline/
│   │   ├── multipoles_mock_0.npz          # Individual mock results
│   │   ├── multipoles_mock_1.npz
│   │   ├── multipoles_mock_2.npz
│   │   ├── multipoles_agg.npz             # Aggregated all mocks
│   │   ├── config.yaml                    # Metadata
│   │   └── rerun.sh                       # Reproduction script
│   │
│   └── contamination_powerlaw_alpha-2.0_amp0.0100/
│       ├── multipoles_mock_0.npz
│       ├── multipoles_mock_1.npz
│       ├── multipoles_mock_2.npz
│       ├── multipoles_agg.npz
│       ├── config.yaml
│       └── rerun.sh
│
└── config_z0.1_0.4_nbarcat_mesh256/
    └── ...
```

### Mock Selection: `--run-mode`

Control which mock realizations to compute:

```bash
# Compute all mocks (default)
"$PYBIN" compute_clean_baseline.py --nmock 10 --run-mode all --verbose

# Compute only MISSING mocks (skip already cached)
# If mocks 0-4 exist and you want nmock=10, computes only 5-9:
"$PYBIN" compute_clean_baseline.py --nmock 10 --run-mode remaining --verbose

# Compute specific mock indices (comma or space-separated)
"$PYBIN" compute_clean_baseline.py --run-mode 0,2,5 --verbose
"$PYBIN" compute_clean_baseline.py --run-mode "1 3 7" --verbose
```

### Example: Incremental Mock Runs

Scenario: Start with 5 mocks, then add 5 more without recomputing the first batch.

```bash
# Step 1: Initial run with 5 mocks
"$PYBIN" compute_clean_baseline.py --nmock 5 --verbose
# Output: cache/config_*/clean_baseline/{multipoles_mock_0.npz ... multipoles_mock_4.npz, multipoles_agg.npz}

# Step 2: Later, extend to 10 mocks, skipping the first 5
"$PYBIN" compute_clean_baseline.py --nmock 10 --run-mode remaining --verbose
# Output: cache/config_*/clean_baseline/{multipoles_mock_5.npz ... multipoles_mock_9.npz, multipoles_agg.npz updated}

# Step 3: Force recomputation of specific subset
"$PYBIN" compute_clean_baseline.py --run-mode "0,3,7" --force --verbose
# Output: cache/config_*/clean_baseline/{multipoles_mock_0.npz, multipoles_mock_3.npz, multipoles_mock_7.npz recomputed}
```

## Logging

Both scripts use DESI logging via `pypower.setup_logging()` for consistent log formatting:

```
INFO: Configuration: z=[0.1, 0.4], nmesh=512, nmock=10
INFO: Config label: z0.1_0.4_nbarcatmesh512
INFO: Cache directory: /path/to/cache/config_z0.1_0.4_nbarcatmesh512/clean_baseline
INFO: Will compute 10 mocks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
INFO: [0] Computing mock 0/9
INFO: [0] Saved: /path/to/cache/.../multipoles_mock_0.npz
...
INFO: Aggregated 10 mocks -> /path/to/cache/.../multipoles_agg.npz
INFO: Saved metadata: /path/to/cache/.../config.yaml
```

Use `--verbose` for DEBUG-level output:

```bash
"$PYBIN" compute_clean_baseline.py --verbose
```

## Key Features

✓ **Clean baseline computed once** — Reused by all contamination variants  
✓ **Two systematic types** — Power-law (smooth) and delta-function (spikes)  
✓ **Clear cache naming** — Filenames encode systematic type and parameters  
✓ **YAML config + CLI** — Config files as defaults, CLI args override specifics  
✓ **Z-range flexibility** — Cache labels include redshift bin  
✓ **Per-realization caching** — Individual mock results saved separately  
✓ **Incremental mock runs** — Compute only missing mocks with `--run-mode remaining`  
✓ **DESI logging integration** — Consistent log formatting with `setup_logging()`  
✓ **Full traceability** — YAML + rerun scripts capture every parameter  

## Extending: Custom Configs

To create a custom config for a different redshift or mesh:

```bash
# Copy and edit default
cp config_clean_baseline.yaml config_clean_baseline_custom.yaml
# Edit config_clean_baseline_custom.yaml (e.g., change zmin, zmax, nmesh)

# Use it
"$PYBIN" compute_clean_baseline.py --config config_clean_baseline_custom.yaml --nmock 20
```

## Troubleshooting

### Clean baseline not found

```bash
"$PYBIN" compute_contaminant_multipoles.py --sys-amp 0.01
# Error: Could not find clean baseline for z=[0.1, 0.4]
```

**Solution:** Run `compute_clean_baseline.py` first with matching z-range.

### File already exists

```bash
"$PYBIN" compute_contaminant_multipoles.py --sys-amp 0.01
# Error: Output exists: ...
```

**Solution:** Use `--force` to recompute:
```bash
"$PYBIN" compute_contaminant_multipoles.py --sys-amp 0.01 --force
```

### PyYAML not available

```bash
"$PYBIN" compute_clean_baseline.py --config config.yaml
# Error: PyYAML not available
```

**Solution:** PyYAML should be available in the DESI environment. If not, scripts fall back to defaults.

## Integration with SLURM

To run on cluster:

```bash
cat > sys_tests/run_contaminant.sbatch << 'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=sys_contam
#SBATCH --time=00:30:00
#SBATCH --mem=16G

cd sys_tests
source ~/.desi_bashrc
"$PYBIN" compute_contaminant_multipoles.py "$@"
EOF

chmod +x sys_tests/run_contaminant.sbatch

# Submit
sbatch sys_tests/run_contaminant.sbatch --sys-type power_law --sys-amp 0.01 --nmock 10 --verbose
# OR
sbatch sys_tests/run_contaminant.sbatch --sys-type delta_function --sys-ell-contam 6 --sys-amp 0.01 --verbose
```

---

**Last Updated:** 2026-07-07  
**Status:** ✓ YAML config support + power-law & delta-function systematicsrelace
