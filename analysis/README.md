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
- `CHUNKERS`: List of chunking strategies (e.g., `FixedSizeChunker-256`, `LumberChunker-GPT`, `LumberChunker-Gemini`)
- `ENCODER`: Encoder type (RegularEncoder/LateEncoder)

### research_question_plots.py

Creates one figure per research question (RQ1–RQ4), summarising evaluation results under `src/chunked_output/`.

**Usage:**
```bash
python analysis/research_question_plots.py \
  --base_path src/chunked_output \
  --output_dir analysis/figures
```

Figures are saved as PNG files. The script automatically skips plots when required runs are missing. Install `matplotlib` if it is not already available in your environment.
