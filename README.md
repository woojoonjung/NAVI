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
The datasets used for training are publicly available at Web Data Commons (https://webdatacommons.org/structureddata/schemaorgtables/2023/).
We constructed our pretraining data from the Top-100 subsets of the Product and Movie domains.

**Required Directory Structure**
```
data/
├── Movie_top100/ # Raw Movie dataset files
│ ├── Movie_.jsonl # Individual table files
│ └── ...
├── Product_top100/ # Raw Product dataset files
│ ├── Product_.jsonl # Individual table files
│ └── ...
└── (output files will be created here)
```

### Preprocessing

Our unified preprocessing pipeline follows a 5-step workflow to prepare the data for training and evaluation:

#### Step 1: Resize Product Dataset

#### Step 2: Flatten Both Datasets

#### Step 3: Create Heldout Datasets
- *Output Files*:
  - `data/WDC_movie_for_mp.jsonl` - Movie heldout for masked prediction
  - `data/WDC_movie_for_cls.jsonl` - Movie heldout for rest of the experiments (unified `genres` field)
  - `data/WDC_product_for_mp.jsonl` - Product heldout for masked prediction  
  - `data/WDC_product_for_cls.jsonl` - Product heldout for rest of the experiments (unified `category` field)

#### Step 4: Remove Heldout Rows from Training Data

```bash
# Run with default settings
python dataset/preprocess.py

# Run with custom settings
python dataset/preprocess.py \
    --movie_dir data/Movie_top100 \
    --product_dir data/Product_top100 \
    --product_max_rows 5000 \
    --heldout_start 450 \
    --heldout_end 459
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

#### Row Clustering
```bash
python experiments/row_clustering.py --model baselines --domain Movie
python experiments/row_clustering.py --model baselines --domain Product
```

#### Domain Consistency Experiment

*Prerequisites*: Canonical sets are prepared in artifacts/lexvar

```bash
# Run complete pipeline
./experiments/run_domain_consistency_exp.sh

# Run individual steps
python experiments/domain_consistency_get_header_embeddings.py \
    --data_dir data \
    --artifacts_dir artifacts/lexvar \
    --domains Movie_top100_cleaned Product_top100_cleaned \
    --models bert tapas haetae navi

python experiments/domain_consistency_exp.py \
    --artifacts_dir artifacts/lexvar \
    --domains Quarter_Movie_top100_cleaned Quarter_Product_top100_cleaned \
    --models bert tapas haetae navi
```

#### Structural Consistency Experiment
```bash
python experiments/structural_consistency_exp.py \
    --data_dir data \
    --artifacts_dir artifacts/structvar \
    --domains Movie_top100_cleaned Product_top100_cleaned \
    --models bert tapas haetae navi \
    --n_samples 100 \
    --n_permutations 5
```

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
- **Row Clustering**: Console output with clustering metrics
- **Domain Consistency**: `artifacts/lexvar/` directory
- **Structural Consistency**: `artifacts/structvar/` directory
- **Segment Visualization**: `artifacts/segment_visualization/` directory
