# Analysis Scripts

This directory contains scripts for analyzing and visualizing experimental results.

## Scripts

### base_result_table_creation.py

Generates LaTeX tables comparing different chunking strategies across models and datasets.

**Usage on cluster:**
```bash
cd /scratch3/wan458/chunking-reproduce
python analysis/base_result_table_creation.py
```

**Output:**
- `results_table_regular_encoder.tex` - LaTeX table for RegularEncoder results

**Requirements:**
- LaTeX packages: `booktabs`, `multirow`

**Configuration:**
- Edit the `main()` function to customize:
  - `BASE_PATH`: Path to evaluation results
  - `DATASETS`: List of datasets to include
  - `MODELS`: List of embedding models
  - `CHUNKERS`: List of chunking strategies
  - `ENCODER`: Encoder type (RegularEncoder/LateEncoder)
