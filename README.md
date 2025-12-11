# Drug Composition to Medical Indication Prediction

## NLP Project: Predicting Drug Indications from Composition Using BioBERT

This project implements a multi-label text classification system to predict medical indications (diseases/conditions) from drug composition and mechanism descriptions using state-of-the-art NLP techniques.

---

## 📚 Project Overview

**Goal:** Predict what medical conditions a drug can treat based on its chemical composition and mechanism of action.

**Approach:**
1. **Baseline Models**: TF-IDF + Logistic Regression, SentenceTransformers + Classifier
2. **Advanced Model**: Fine-tuned BioBERT (biomedical BERT)
3. **Task Type**: Multi-label text classification

**Dataset:** MID (Medicine Information Database) - 147,831+ drug entries with 509 medical condition labels

---

## 📂 Repository Structure

```
CS6120-NLP-Final-Project/
├── README.md                          # This file
├── extract_MID2_data.py               # Data extraction script for MID dataset
├── final_masked_data.csv              # Processed MID dataset 
├── MID.xlsx                           # Raw MID dataset (Excel format)
│
├── 1_data_preprocessing.ipynb         # Part 1: Data loading, cleaning, multi-label encoding
├── 2_baseline_models.ipynb            # Part 2: TF-IDF and SentenceTransformers baselines
├── 3_biobert_model.ipynb              # Part 3: BioBERT fine-tuning (Google Colab)
│
└── data/                              # Generated files (created during execution)
    ├── X_train_text.npy
    ├── X_test_text.npy
    ├── y_train.npy
    ├── y_test.npy
    ├── mlb.pkl                        # MultiLabelBinarizer
    └── ...
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
# Jupyter Notebook or JupyterLab
# GPU recommended (but not required) for BioBERT training
```

### Installation

```bash
# Clone or download this repository
cd CS6120-NLP-Final-Project

# Install dependencies (run in notebook or terminal)
pip install pandas numpy matplotlib seaborn
pip install transformers datasets torch accelerate
pip install sentence-transformers scikit-learn
pip install wordcloud plotly tqdm
```

### Running the Project

**Step 1: Data Preprocessing** (Required)
```bash
jupyter notebook 1_data_preprocessing.ipynb
```
- Loads MID dataset (final_masked_data.csv)
- Explores data distribution (147,831 drugs, 509 labels)
- Creates multi-label encodings using MultiLabelBinarizer
- Splits data: 80% train, 20% test (then 80/20 train/val split)
- Saves preprocessed data to `data/` directory
- **Runtime:** ~10-15 minutes

**Step 2: Baseline Models** (Recommended)
```bash
jupyter notebook 2_baseline_models.ipynb
```
- **Model 1**: TF-IDF (5000 features) + Logistic Regression
  - Macro F1: ~0.43, Micro F1: ~0.95
- **Model 2**: SentenceTransformers (all-MiniLM-L6-v2) + Logistic Regression
  - Macro F1: ~0.31, Micro F1: ~0.89
- Comprehensive evaluation and comparison
- **Runtime:** ~20-30 minutes

**Step 3: BioBERT Fine-tuning** (Main Model) - **Google Colab Required**
```bash
jupyter notebook 3_biobert_model.ipynb
```
- Fine-tune BioBERT (dmis-lab/biobert-v1.1) on drug indication data
- Multi-label classification with 509 labels
- Training: 3 epochs, batch size 8, learning rate 2e-5
- Evaluate on test set and compare with baselines
- **Runtime:** ~2-4 hours (GPU) or ~8-12 hours (CPU)
- **Note:** Designed for Google Colab with GPU support

---

## 📊 Expected Results

| Model | F1 (Macro) | F1 (Micro) | Hamming Loss | Exact Match |
|-------|------------|------------|--------------|-------------|
| TF-IDF + LR | ~0.43 | ~0.95 | ~0.0003 | ~0.89 |
| SentenceEmb + LR | ~0.31 | ~0.89 | ~0.0006 | ~0.77 |
| **BioBERT** | **TBD** | **TBD** | **TBD** | **TBD** |

*Note: BioBERT results depend on training (Google Colab with GPU recommended)*

---

## 🔬 NLP Concepts Covered

### Classical NLP (Notebook 2)
- ✅ **Tokenization & Text Preprocessing**
- ✅ **TF-IDF Vectorization**: Bag-of-words with importance weighting
- ✅ **Feature Engineering**: Unigrams and bigrams, min/max document frequency
- ✅ **Sparse Representations**: 5000 features from vocabulary
- ✅ **Multi-label Classification**: OneVsRestClassifier with Logistic Regression

### Modern NLP (Notebook 2 & 3)
- ✅ **Sentence Embeddings**: Pre-trained SentenceTransformers (all-MiniLM-L6-v2)
- ✅ **Dense Representations**: 384-dimensional embeddings
- ✅ **Transfer Learning**: Using pre-trained BioBERT
- ✅ **Transformer Architecture**: BERT-based attention mechanisms
- ✅ **Fine-tuning Strategies**: Domain adaptation for biomedical text
- ✅ **Multi-label Classification**: 509 medical conditions per drug

### Evaluation Metrics
- ✅ **Macro/Micro F1 Scores**: Precision, recall, F1-score
- ✅ **Hamming Loss**: Label-wise accuracy
- ✅ **Exact Match Ratio**: Full prediction accuracy

---

## 📈 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Medicine Data                         │
│  (Drug descriptions, indications, mechanisms)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Notebook 1: Data Preprocessing                    │
│  • Load MID dataset (final_masked_data.csv)                  │
│  • Parse medical conditions from labels                     │
│  • Create multi-label encodings                             │
│  • Train/Val/Test split                                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────────┐  ┌──────────────────────┐
│   Notebook 2:        │  │   Notebook 3:        │
│   Baseline Models    │  │   BioBERT Model      │
│                      │  │                      │
│  • TF-IDF + LR       │  │  • Load BioBERT      │
│  • SentenceEmb + LR  │  │  • Fine-tune         │
│  • Evaluation        │  │  • Evaluate          │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └────────┬────────────────┘
                    ▼
        ┌──────────────────────┐
        │  Final Comparison    │
        │  & Results Analysis  │
        └──────────────────────┘
```

---

## 💡 Key Features

1. **Multi-Label Classification**: Each drug can treat multiple conditions (509 labels)
2. **Biomedical NLP**: Uses domain-specific pre-trained models (BioBERT)
3. **Sentence Embeddings**: Dense representations using SentenceTransformers
4. **Comprehensive Evaluation**: Multiple metrics (F1, precision, recall, Hamming loss)
5. **Baseline Comparison**: Shows improvement from classical to modern NLP

---

## 🎯 Project Goals Achieved

- [x] Preprocess medical text data
- [x] Extract structured labels from free text
- [x] Implement classical NLP baseline (TF-IDF)
- [x] Implement embedding-based baseline (SentenceTransformers)
- [x] Fine-tune transformer model (BioBERT)
- [x] Multi-label classification with 509 classes
- [x] Comprehensive evaluation and comparison
- [x] Well-documented Jupyter notebooks

---

## 📝 Data Description

### Dataset Statistics
- **Total drug entries**: 147,831
- **Number of unique labels**: 509 medical conditions
- **Average labels per drug**: 1.57
- **Label range**: 1-9 conditions per drug
- **Data split**: 80% train, 20% test (with 20% of train as validation)

### Input Features (X)
Combination of:
- **Drug Composition (CONTAINS)**: Chemical composition and active ingredients
- **Mechanism of Action (HOW_WORKS)**: How the drug works biologically
- **Format**: "Drug Composition: [composition] Mechanism of Action: [mechanism]"

### Output Labels (y)
- **Medical Conditions**: Multi-hot encoded binary vector (509 dimensions)
- **Top conditions**: 
  - Bacterial infections (20,439 samples)
  - Pain relief (13,110 samples)
  - Gastroesophageal reflux disease (10,297 samples)
  - Type 2 diabetes mellitus (8,966 samples)
  - Hypertension (7,587 samples)
- **Format**: Binary matrix (samples × 509 conditions)

---

## 🧹 Data Preprocessing Pipeline

We applied a structured preprocessing pipeline (available in `extract_MID2_data.py`) to prepare the data:

1. **Text Cleaning & Normalization**
   - Preserved newlines in `USES` column to distinguish multiple conditions
   - Normalized whitespace and removed carriage returns in feature columns
   - Removed unrelated metadata and empty values

2. **Label Extraction**
   - Extracted medical conditions by splitting `USES` on newlines
   - Removed numbering (e.g., "1.") and "Treatment of" prefixes
   - Filtered out short labels (length < 3)

3. **Data Filtering & Deduplication**
   - Removed duplicates based on Drug Name and Composition
   - Filtered out conditions appearing fewer than 5 times (min_freq=5)
   - Excluded samples with no valid labels

4. **Input Feature Construction**
   - Merged relevant fields into a unified text description:
     `Drug Composition: [CONTAINS] Mechanism of Action: [HOW_WORKS]`
   - This text became the input for all models

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory (OOM) during BioBERT training**
- Use Google Colab with GPU (Tesla T4 or better)
- Reduce batch size in training arguments (e.g., from 8 to 4)
- Reduce max_length in tokenizer (e.g., from 512 to 256)

**2. Missing data files**
- Ensure you run Notebook 1 first to generate data files
- Check that `data/` directory exists with .npy and .pkl files

**3. GPU not detected**
- For BioBERT training, use Google Colab with GPU runtime
- Runtime → Change runtime type → GPU

**4. Module import errors**
- Install all required packages: `pip install transformers datasets torch accelerate sentence-transformers scikit-learn`

---

## 📚 Additional Resources

### Papers
- [BioBERT: Pre-trained Biomedical Language Representation Model](https://arxiv.org/abs/1901.08746)
- [Attention Is All You Need (Transformers)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

### Datasets
- [MID (Medicine Information Database)](https://data.mendeley.com/datasets/2vk5khfn6v/3/)

### Models
- [BioBERT on HuggingFace](https://huggingface.co/dmis-lab/biobert-v1.1)
- [SentenceTransformers](https://www.sbert.net/)

---

## 👥 Authors

- **Course**: CS6120 Natural Language Processing
- **Project Type**: Drug Composition to Medical Indication Prediction
- **Date**: November 2025

---

## 📄 License

This project is for educational purposes as part of an NLP course.

---

## 🙏 Acknowledgments

- MID (Medicine Information Database) for comprehensive drug data
- HuggingFace for transformer models and libraries
- dmis-lab for BioBERT pre-trained model
- SentenceTransformers for embedding models
- Google Colab for providing free GPU resources

---

**Happy Coding! 🚀**
