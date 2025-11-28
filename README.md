# Drug Composition to Medical Indication Prediction

## NLP Project: Predicting Drug Indications from Composition Using BioBERT

This project implements a multi-label text classification system to predict medical indications (diseases/conditions) from drug composition and mechanism descriptions using state-of-the-art NLP techniques.

---

## 📚 Project Overview

**Goal:** Predict what medical conditions a drug can treat based on its chemical composition, mechanism of action, and pharmacodynamics.

**Approach:**
1. **Baseline Models**: TF-IDF + Logistic Regression, SentenceTransformers
2. **Advanced Model**: Fine-tuned BioBERT (biomedical BERT)
3. **Task Type**: Multi-label text classification

**Dataset:** DrugBank Simplified (17,000+ drugs with detailed medical descriptions)

---

## 📂 Repository Structure

```
CS6120-NLP-Final-Project/
├── README.md                          # This file
├── NLP_project_analysis.md            # Detailed project analysis and recommendations
├── drugbank_simplified.csv            # Main dataset
├── Medicine_Details.csv               # Alternative dataset
├── MID.csv                            # Large supplementary dataset
│
├── 1_data_preprocessing.ipynb         # Part 1: Data loading, cleaning, NER extraction
├── 2_baseline_models.ipynb            # Part 2: TF-IDF and SentenceTransformers baselines
├── 3_biobert_model.ipynb              # Part 3: BioBERT fine-tuning
│
└── outputs/                           # Generated files (created during execution)
    ├── X_train.npy, X_val.npy, X_test.npy
    ├── y_train.npy, y_val.npy, y_test.npy
    ├── mlb.pkl
    ├── tfidf_vectorizer.pkl
    ├── biobert_finetuned/
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
pip install transformers datasets torch
pip install sentence-transformers scikit-learn
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
pip install wordcloud plotly tqdm
```

### Running the Project

**Step 1: Data Preprocessing** (Required)
```bash
jupyter notebook 1_data_preprocessing.ipynb
```
- Loads and cleans DrugBank dataset
- Extracts medical conditions using NER
- Creates multi-label encodings
- Saves preprocessed data
- **Runtime:** ~20-30 minutes

**Step 2: Baseline Models** (Recommended)
```bash
jupyter notebook 2_baseline_models.ipynb
```
- TF-IDF + Logistic Regression
- SentenceTransformers + Classifier
- Evaluation and comparison
- **Runtime:** ~10-15 minutes

**Step 3: BioBERT Fine-tuning** (Main Model)
```bash
jupyter notebook 3_biobert_model.ipynb
```
- Fine-tune BioBERT on drug indication data
- Evaluate on test set
- Compare with baselines
- **Runtime:** ~2-4 hours (GPU) or ~8-12 hours (CPU)

---

## 📊 Expected Results

| Model | F1 (Macro) | F1 (Micro) | Hamming Loss |
|-------|------------|------------|--------------|
| TF-IDF + LR | ~0.45-0.55 | ~0.60-0.70 | ~0.05-0.10 |
| SentenceEmb + LR | ~0.50-0.60 | ~0.65-0.75 | ~0.04-0.08 |
| **BioBERT** | **~0.65-0.75** | **~0.75-0.85** | **~0.02-0.05** |

*Note: Actual results may vary based on data preprocessing choices and hyperparameters*

---

## 🔬 NLP Concepts Covered

### Classical NLP (Notebook 2)
- ✅ **Tokenization & Text Preprocessing**
- ✅ **TF-IDF Vectorization**: Bag-of-words with importance weighting
- ✅ **Feature Engineering**: N-grams, min/max document frequency
- ✅ **Sparse vs Dense Representations**

### Modern NLP (Notebook 3)
- ✅ **Transfer Learning**: Using pre-trained BioBERT
- ✅ **Transformer Architecture**: Attention mechanisms
- ✅ **Contextualized Embeddings**: Word meaning depends on context
- ✅ **Fine-tuning Strategies**: Domain adaptation for biomedical text
- ✅ **Multi-label Classification**: Predicting multiple labels per sample

### Medical NLP (Notebook 1)
- ✅ **Named Entity Recognition (NER)**: Extracting disease entities
- ✅ **Medical Text Processing**: Handling scientific terminology
- ✅ **Domain-Specific Models**: BioBERT, ScispaCy

---

## 📈 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw DrugBank Data                         │
│  (Drug descriptions, indications, mechanisms)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Notebook 1: Data Preprocessing                    │
│  • Clean text (remove references, HTML)                     │
│  • Extract medical conditions using NER                     │
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

1. **Multi-Label Classification**: Each drug can treat multiple conditions
2. **Biomedical NLP**: Uses domain-specific pre-trained models (BioBERT)
3. **Medical NER**: Automatically extracts disease entities from text
4. **Comprehensive Evaluation**: Multiple metrics (F1, precision, recall, Hamming loss)
5. **Baseline Comparison**: Shows improvement from classical to modern NLP

---

## 🎯 Project Goals Achieved

- [x] Preprocess medical text data
- [x] Extract structured labels from free text
- [x] Implement classical NLP baseline (TF-IDF)
- [x] Implement embedding-based baseline (SentenceTransformers)
- [x] Fine-tune transformer model (BioBERT)
- [x] Multi-label classification with 200+ classes
- [x] Comprehensive evaluation and comparison
- [x] Well-documented Jupyter notebooks

---

## 📝 Data Description

### Input Features (X)
Combination of:
- **Description**: Chemical composition, molecular structure
- **Mechanism of Action**: How the drug works biologically
- **Pharmacodynamics**: Drug effects on the body

### Output Labels (y)
- **Medical Conditions**: Multi-hot encoded vector
- **Examples**: cancer, hypertension, diabetes, infection, pain, etc.
- **Number of Labels**: ~200-300 (frequent conditions)
- **Format**: Binary matrix (samples × conditions)

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory (OOM) during BioBERT training**
```python
# In notebook 3, reduce batch size:
per_device_train_batch_size=4  # instead of 8
```

**2. NER extraction too slow**
```python
# In notebook 1, use smaller sample:
df_sample = df.head(5000)  # Test on smaller subset first
```

**3. GPU not detected**
```python
# Install PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Missing libraries**
```bash
# Run all install commands in notebooks sequentially
```

---

## 📚 Additional Resources

### Papers
- [BioBERT: Pre-trained Biomedical Language Representation Model](https://arxiv.org/abs/1901.08746)
- [Attention Is All You Need (Transformers)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

### Datasets
- [DrugBank Database](https://go.drugbank.com/)
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)

### Models
- [BioBERT on HuggingFace](https://huggingface.co/dmis-lab/biobert-v1.1)
- [SentenceTransformers](https://www.sbert.net/)
- [ScispaCy](https://allenai.github.io/scispacy/)

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

- DrugBank for providing comprehensive drug data
- HuggingFace for transformer models and libraries
- AllenAI for ScispaCy medical NLP tools
- dmis-lab for BioBERT pre-trained model

---

## 📧 Contact

For questions or issues, please refer to the `NLP_project_analysis.md` file for detailed explanations or create an issue in the repository.

---

**Happy Coding! 🚀**
