# Hinglishly Tanya

This project trains BERT and Seq2Seq models on a Hinglish dataset.

## Setup

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data

The dataset is located in the `data/` directory:

- `train.tsv`
- `validation.tsv`
- `test.tsv`

## Training

Open `project.ipynb` in VS Code or Jupyter Lab to run the training pipeline.

The notebook covers:

1. **Error Detection**: Using `bert-base-multilingual-cased` to classify errors (e.g., Grammar, Spelling, Slang).
2. **Grammar Correction**: Using `google/mt5-small` to correct Hinglish sentences.

## UI Application

To run the Grammarly-like UI:

```bash
streamlit run app.py
```

## Models

The trained models will be saved in:

- `bert-domain-classifier/`
- `mt5-hinglish-correction/`
# hinglishly-repo
