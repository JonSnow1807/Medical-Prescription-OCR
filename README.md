# Medical Prescription OCR 🏥

A transformer-based Optical Character Recognition (OCR) system for handwritten medical prescriptions built on NAVER Clova’s **Donut** architecture, extended with zero-shot document classification.

<div align="center">

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/chinmays18/medical-prescription-ocr)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/chinmays18/medical-prescription-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Features](#-features)
3. [Performance](#-performance)
4. [Quick Start](#-quick-start)
5. [Model Usage](#-model-usage)
6. [Dataset](#-dataset)
7. [Training](#-training)
8. [Tech Stack](#️-tech-stack)
9. [Project Structure](#-project-structure)
10. [Contributing](#-contributing)
11. [License](#-license)
12. [Acknowledgments](#-acknowledgments)

---

## 🚀 Overview

**Medical Prescription OCR** (formerly **RxReader**) converts doctors’ handwritten prescriptions into structured, machine-readable text with high accuracy.

### Key Capabilities
- **Accurate OCR** – Transcribes drug names, dosages, frequencies and instructions  
- **Structured Output** – Returns clean JSON with parsed prescription elements  
- **Zero-shot Classification** – Detects prescription documents vs. other medical forms  
- **Robust Performance** – Handles diverse handwriting styles and image qualities  

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Pre-trained Model** | Ready to use on [HF Model Hub](https://huggingface.co/chinmays18/medical-prescription-ocr) |
| 📊 **Comprehensive Dataset** | 1 000 synthetic, fully-annotated images on [HF Datasets](https://huggingface.co/datasets/chinmays18/medical-prescription-dataset) |
| 🖥️ **User-Friendly Interface** | Gradio web app for drag-and-drop testing |
| 🔄 **Gradual Augmentation** | Novel curriculum for robust learning |
| 📈 **Production Ready** | Download script and deployment guide included |

---

## 📊 Performance

| Metric | Score | Notes |
|--------|-------|-------|
| **Character-level accuracy** | **71 %** | Individual character recognition |
| **Word-level accuracy** | **84 %** | Complete word recognition |
| **Processing speed** | **≈ 2 s/img** | CPU – Apple M1 |

*Benchmarked on 100 held-out prescriptions with varied handwriting.*

---

## ⚡ Quick Start

### Prerequisites
* Python ≥ 3.8  
* ~2 GB free disk space for model files  
* (Optional) CUDA GPU for faster inference  

### Installation

```bash
# 1 – Clone the repo
git clone https://github.com/JonSnow1807/medical-prescription-ocr.git
cd medical-prescription-ocr

# 2 – Install dependencies
pip install -r requirements.txt

# 3 – Download the pre-trained model (~800 MB)
python model_download.py

# 4 – Launch the Gradio app
python app.py
