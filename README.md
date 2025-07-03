# Handwritten Medical Prescription OCR

**Handwritten Medical Prescription OCR** (internal codename **RxReader**) is a transformer‑based Optical Character Recognition (OCR) system that turns doctors’ handwritten prescriptions into clean, structured text. It is built with **PyTorch Lightning** and **NAVER Clova Donut** and includes a full training pipeline, gradual augmentations, and detailed evaluation metrics.

---

## 🚀 Overview

| | |
|---|---|
| **Goal** | Accurately transcribe drug name, dosage, frequency, and instructions from prescription images and return JSON. |
| **Model** | VisionEncoderDecoderModel (Donut) fine‑tuned on a curated prescription dataset. |
| **Workflow** | 1. **Data** in train/val/test folders with JSON annotations  2. **Augmentations**: basic early → advanced later  3. **Training** automated by PyTorch Lightning  4. **Evaluation**: character‑ and word‑level accuracy. |

---

## 🛠️ Tech Stack
* **Python 3.8+**  
* **PyTorch & PyTorch Lightning** – core DL & training loop  
* **Hugging Face Transformers** – Donut model & processor  
* **Albumentations** – fast, rich image augmentation  
* **SentencePiece** – sub‑word tokenizer for Donut  
* **TQDM, NumPy, Pillow** – utilities & image I/O

---

## 🔧 Quick Start
```bash
# 1  Clone
 git clone https://github.com/JonSnow1807/Medical-Prescription-OCR.git
 cd Medical-Prescription-OCR

# 2  Install deps (virtualenv recommended)
 pip install torch torchvision torchaudio pytorch-lightning              transformers sentencepiece Pillow albumentations[imgaug]

# 3  Dataset layout
 data/
   └── train/
       ├── images/
       └── annotations/
   └── val/
   └── test/
 train.txt  # list of training image filenames
 val.txt
 test.txt

# 4  Run the notebook step‑by‑step
 jupyter notebook ocr.ipynb
```

> **Note**  Set `data_root` inside the notebook to your dataset path.

---

## 📌 Code Structure
1. **Hyper‑parameters** (learning rate, epochs, batch_size…) live at the top of `ocr.ipynb`.  
2. **Data Module** (`PrescriptionDataModule`) – handles splits & augmentations (gradual strategy).  
3. **Model Module** (`PrescriptionOCRModule`) – wraps Donut, enables gradient checkpointing, logs losses.  
4. **Callbacks** – `ModelCheckpoint`, `EarlyStopping`, and a custom `GradualAugmentationCallback`.  
5. **Evaluation** – accuracy metrics + sample visualisations.

---

## 🏆 Baseline Results
| Metric | Value* |
|---|---|
| Character‑level Accuracy | ~71 % |
| Word‑level Accuracy | ~84 % |

*Numbers reported on an internal set of 200 test prescriptions; expect variation by dataset.

---

## 🤖 Why These Libraries?
* **PyTorch Lightning** – clean training loops, fewer bugs.  
* **Donut** – state‑of‑the‑art for document OCR without heavy detectors.  
* **Albumentations** – high‑quality augmentations with speed.  
* **SentencePiece** – efficient tokenisation for mixed handwriting/typed text.

---

## 🎯 Planned Improvements
* Self‑supervised pre‑training on unlabeled prescription scans.
* Context‑aware spell‑correction using FDA drug dictionary.
* Lightweight model distillation for mobile inference.
* Larger, more diverse dataset (different scripts & languages).

---

## 🙋‍♂️ Author
Chinmay Shrivastava – MS CSE, self‑taught ML engineer passionate about AI.

> *Feel free to create an issue or submit a PR if you’d like to contribute!*
