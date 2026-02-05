# NAVI

This repository contains the implementation of NAVI; Entropy-aware Alignment with Header-Value Induction.

## Quick Setup

### Environment Setup

```bash
# Create conda environment from specification
conda env create -f environment.yml

# Activate environment
conda activate navi

# Download spaCy model (required for preprocessing)
python -m spacy download en_core_web_sm
```

### Verify Installation

```bash
# Test basic imports
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test project imports
python -c "from model.navi import NaviForMaskedLM; from dataset.dataset import NaviDataset; print('✓ Project imports successful')"
```

## Data Preparation

### Download Datasets
The datasets used for training are publicly available at Web Data Commons (`https://webdatacommons.org/structureddata/schemaorgtables/2023/`).
We constructed our pretraining data from the Top-100 subsets of the Product and Movie domains.

### Directory Structure

Raw data should be placed under `data/raw/...`. The preprocessing pipeline will create all derived artifacts under `data/flattened/...` and `data/cleaned/...`:

```text
data/
├── raw/
│   ├── Movie_top100/              # Raw Movie tables
│   │   ├── Movie_*.jsonl
│   │   └── ...
│   └── Product_top100/            # Raw Product tables
│       ├── Product_*.jsonl
│       └── ...
├── flattened/
│   ├── Movie_top100/              # Flattened Movie tables
│   └── Product_top100/            # Flattened Product tables
└── cleaned/
    ├── Movie_top100/              # Cleaned per-table Movie data
    ├── Product_top100/            # Cleaned per-table Product data
    ├── Movie/
    │   ├── train/                 # 80% split per cleaned Movie table
    │   └── validation/            # 10% split per cleaned Movie table
    ├── Product/
    │   ├── train/                 # 80% split per cleaned Product table
    │   └── validation/            # 10% split per cleaned Product table
    └── test/
        ├── WDC_movie_for_mp.jsonl
        ├── WDC_movie_for_cls.jsonl
        ├── WDC_product_for_mp.jsonl
        └── WDC_product_for_cls.jsonl
```

### Preprocessing

The unified preprocessing pipeline in `dataset/preprocess.py` follows a 5-step workflow:

#### Step 1: Stratified Resize of Product Dataset
- Counts total rows across all raw Product tables under `data/raw/Product_top100`.
- Computes a global sampling ratio so the total Product rows approximately match the Movie total (default target: **480,817**).
- Rewrites each raw Product table in-place with a random, deterministic subset of rows.

#### Step 2: Flatten Both Datasets
- Reads raw Movie/Product tables from `data/raw/...`.
- Flattens nested JSON structures into flat key–value maps.
- Writes flattened tables to:
  - `data/flattened/Movie_top100`
  - `data/flattened/Product_top100`

#### Step 3: Clean Flattened Datasets
- Applies language filtering (keeps primarily English tables).
- Ensures BERT tokenizability, truncates overly long fields, and subsamples indexed fields.
- Writes cleaned per-table data to:
  - `data/cleaned/Movie_top100`
  - `data/cleaned/Product_top100`

#### Step 4: Create Train / Validation / Test Splits
- For each cleaned table (Movie and Product):
  - Shuffles rows with a fixed random seed.
  - Splits into **80% train**, **10% validation**, **10% test**.
- Writes per-table train/validation splits to:
  - `data/cleaned/Movie/train`, `data/cleaned/Movie/validation`
  - `data/cleaned/Product/train`, `data/cleaned/Product/validation`
- Aggregates the remaining 10% test rows in memory for heldout file creation.

#### Step 5: Create Heldout Datasets (MP / CLS)
- From the aggregated Movie test rows:
  - `WDC_movie_for_mp.jsonl`: all Movie test rows (for masked prediction).
  - `WDC_movie_for_cls.jsonl`: Movie test rows with a unified non-empty `genres` field.
- From the aggregated Product test rows:
  - `WDC_product_for_mp.jsonl`: all Product test rows (for masked prediction).
  - `WDC_product_for_cls.jsonl`: Product test rows with a unified non-empty `category` field.
- All four files are written to `data/cleaned/test/`.

```bash
# Run with default settings
python dataset/preprocess.py

# Run with custom settings
python dataset/preprocess.py \
    --raw_movie_dir data/raw/Movie_top100 \
    --raw_product_dir data/raw/Product_top100 \
    --flattened_movie_dir data/flattened/Movie_top100 \
    --flattened_product_dir data/flattened/Product_top100 \
    --cleaned_movie_dir data/cleaned/Movie_top100 \
    --cleaned_product_dir data/cleaned/Product_top100 \
    --target_product_rows 480817 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --seed 42
```


## Usage

### Training Models

```bash
# Train Navi models (Default setting, hyperparam variants, ablations)
bash pretrain_navi.sh

# Train baseline models (BERT, TAPAS, HAETAE)
bash pretrain_baselines.sh
```

### Experiments

#### Masked Prediction
```bash
python experiments/masked_prediction.py --model baselines --domain Movie
python experiments/masked_prediction.py --model baselines --domain Product
```

#### Row Classification
```bash
python experiments/row_classification.py --model baselines --domain Movie
python experiments/row_classification.py --model baselines --domain Product
```

#### Header Clustering

*Prerequisites*: Canonical sets are prepared in artifacts/lexvar

```bash
# Run complete pipeline
./experiments/run_header_clustering.sh

# Run individual steps
python experiments/domain_consistency_get_header_embeddings.py \
    --data_dir data \
    --artifacts_dir artifacts/lexvar \
    --domains Movie_top100_cleaned Product_top100_cleaned \
    --models bert tapas haetae navi

python experiments/header_clustering.py \
    --artifacts_dir artifacts/lexvar \
    --domains Quarter_Movie_top100_cleaned Quarter_Product_top100_cleaned \
    --models bert tapas haetae navi
```

#### Robustness Analysis

```bash
# Run default mode
python experiments/robustness_exp.py

# Run on specific domains
python experiments/robustness_exp.py --domains cleaned/Movie
python experiments/robustness_exp.py --domains cleaned/Product

# Run specific models only
python experiments/robustness_exp.py --models bert tapas haetae navi
python experiments/robustness_exp.py --models woSSI woMSM woESA
```

Results are saved to `experiments/robustness_results/`:

#### Segment Visualization
```bash
# Run complete visualization pipeline
./experiments/run_segment_visualization.sh

# Run individual steps
python experiments/visualization_get_segment_embeddings.py \
    --model_path ./models/navi_movie/epoch_2 \
    --output_path ./artifacts/segment_visualization/segments \
    --n_tables 6 \
    --rows_per_table 169 \
    --random_state 42

python experiments/visualization_plot_segments.py \
    --input ./artifacts/segment_visualization/segments/segments.json \
    --outdir ./artifacts/segment_visualization/plots \
    --model-name "Navi" \
    --umap-n-neighbors 30 \
    --umap-min-dist 0.05 \
    --tsne-perplexity 30 \
    --seed 42 \
    --preprocessing l2_normalize
```

#### Output Locations

- **Masked Prediction**: Console output with accuracy scores
- **Row Classification**: Console output with F1 scores
- **Header Clustering**: `artifacts/lexvar/` directory
- **Robustness Analysis**: `artifacts/structvar/` directory
- **Segment Visualization**: `artifacts/segment_visualization/` directory