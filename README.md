# GPT from Scratch

This project implements and compares three different language models: RNN, LSTM, and Transformer, trained on the Penn Treebank (PTB) dataset. It also includes fine-tuning experiments with GPT-2 on the PubMedQA dataset.

## Project Structure

```
.
├── data/               # Dataset directory
├── results/            # Experimental results
│   ├── A/             # Part A results (model training)
│   ├── B/             # Part B results (model comparison)
│   └── C/             # Part C results (fine-tuning)
├── plots/             # Generated plots
│   ├── A/             # Training loss curves
│   ├── B/             # Perplexity plots
│   └── C/             # Fine-tuning plots
├── src/               # Source code
│   └── plot_results.py # Plotting utilities
├── report.md          # Technical report (Chinese)
├── report_en.md       # Technical report (English)
└── requirements.txt   # Project dependencies
```

## Team Members

- **Ruize He**: Core implementation of RNN, LSTM, and Transformer models
- **Hengfei Zhao**: Technical documentation and visualization
- **Wenxi Wu**: Data analysis and presentation

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate plots from results:
```bash
python src/plot_results.py
```

## Results

The project compares three different model architectures:
- RNN (built from scratch)
- LSTM (PyTorch-based)
- Transformer

Key findings:
- Transformer shows the best performance with validation perplexity of 346.83
- LSTM achieves intermediate results with perplexity of 390.02
- RNN demonstrates baseline performance with perplexity of 411.18

For detailed analysis and results, please refer to the technical reports in [Chinese](report.md) or [English](report_en.md). 