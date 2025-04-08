# BirdCLEF 2025 - Audio Classification Challenge

This repository contains our team's solution for the [Kaggle BirdCLEF 2025](https://www.kaggle.com/competitions/birdclef-2025/data) competition. We aim to classify bird species from audio recordings using deep learning techniques such as spectrogram-based inputs and EfficientNet backbones.

---

## 🧠 Overview

Our pipeline involves the following components:

- **Preprocessing**: Audio-to-mel spectrogram transformation
- **Training**: EfficientNet-based PyTorch models using techniques like mixup and spec augmentations
- **Evaluation**: ROC AUC scoring on a stratified 5-fold validation setup
- **Modularity**: Code is organized into reusable components for preprocessing, training, validation, and inference

---

## 🗂 Repository Structure

```text
.
├── data/                            # BirdCLEF dataset (download from Kaggle)
├── args.py                          # Configuration
├── model.py                         # Model definition
├── utils.py                         # Preprocessing, dataset, audio, optimizer utilities
├── train.py                         # Training logic with k-fold cross-validation
├── valid.py                         # Evaluation / inference code
├── main.py                          # Central script to run pre / train / valid modes
├── pyproject.toml                   # Project metadata and dependencies
└── README.md                        # You're reading it
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation) (or pip, if preferred)

### Install with Poetry

```bash
poetry install
poetry shell
```

If you're using `pip` directly instead of Poetry:

```bash
pip install -e .
```

---

## 🚀 Usage

Each operation mode is controlled via `main.py`.

### 📥 Preprocessing

```bash
python main.py --mode pre
```

Generates pre-computed mel spectrograms saved as a `.npy` file.

### 🎯 Training

```bash
python main.py --mode train
```

Optional flags:

```bash
--epochs 20 --batch_size 64 --model_name efficientnet_b1 --preprocessing on_the_fly
```

### 📊 Validation

```bash
python main.py --mode valid --checkpoint ./model_fold0.pth
```

### 🔁 Train & Validate (Combo)

```bash
python main.py --mode train_valid --checkpoint ./model_fold0.pth
```

---

## 👥 Team Contributions

- **Jirsak Buranathawornsom**  
  📧 [wan.jirasak@gmail.com](mailto:wan.jirasak@gmail.com)  

- **Pasit Tiwawongrut**  
  📧 [maxkung55@gmail.com](mailto:maxkung55@gmail.com)  

- **Todsavad Tangtortan**  
  📧 [todsavadt@gmail.comm](mailto:todsavadt@gmail.comm)  

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/competitions/birdclef-2025/data) for organizing the BirdCLEF 2025 challenge.
- The Timm library for providing pre-trained models.
