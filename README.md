# AccuScan - OCR

AccuScan is a powerful **Optical Character Recognition (OCR)** system built using **PyTorch Lightning** and the **Donut-based VisionEncoderDecoderModel**. This project is designed to efficiently extract text from images using **state-of-the-art deep learning techniques**. It features a structured training pipeline, optimized augmentations, and advanced evaluation metrics to ensure high accuracy.

## 🚀 Overview
**Objective:** Train an OCR system to convert images of text into structured transcriptions.  
**Model:** Based on **NAVER Clova's Donut architecture** for OCR tasks.  
**Workflow:**  
- **Data Handling:** Organized into train, validation, and test sets with JSON annotations.  
- **Augmentations:** Gradual augmentation strategy with basic transforms in early epochs and advanced augmentations later.  
- **Training:** PyTorch Lightning automates training, logging, checkpointing, and early stopping.  
- **Evaluation:** Computes **character-level and word-level accuracy** to assess performance.

## 🛠️ Technologies and Libraries Used
- **Python 3.7+**
- **PyTorch & PyTorch Lightning** - Core deep learning framework & training management
- **Hugging Face Transformers** - Provides Donut-based VisionEncoderDecoderModel
- **Albumentations** - Image augmentation library
- **TQDM** - For progress tracking
- **Numpy & Pillow** - Numeric computations & image processing
- **SentencePiece** - Tokenizer for the Donut model

## 🔧 How to Run
1. **Clone this repository:**
   ```bash
   git clone https://github.com/shashank11yadav/AccuScan.git
   cd AccuScan
   ```
2. **Install dependencies (Recommended: use a virtual environment):**
   ```bash
   pip install torch torchvision torchaudio transformers pytorch-lightning sentencepiece Pillow albumentations[imgaug]
   ```
3. **Prepare the dataset:**
   - Store images in `train/images`, `val/images`, `test/images`.
   - Store corresponding JSON annotations in `train/annotations`, `val/annotations`, `test/annotations`.
   - Ensure `train.txt`, `val.txt`, and `test.txt` contain image filenames.
   - Update dataset paths if needed:
     ```python
     data_root = "/your/data/path"
     ```
4. **Run the training script:**
   - Open and execute `ocr.ipynb` step by step.
5. **Monitor Training:**
   - PyTorch Lightning logs training progress.
   - Best model checkpoint is saved in `donut_trained_model/`.
6. **Evaluate Model:**
   - The script runs inference on the test set.
   - Accuracy metrics and sample predictions are displayed.

## 📌 Detailed Code Breakdown
### **1. Paths and Hyperparameters**
- Dataset paths are structured for training, validation, and testing.
- Example hyperparameters:
  ```python
  learning_rate = 3e-5
  weight_decay = 0.001
  max_epochs = 30
  warmup_ratio = 0.05
  batch_size = 2
  num_workers = 4
  max_length = 512
  precision = 16
  ```
- Modify these based on your dataset and hardware.

### **2. Data Handling & Augmentation**
- Uses **Albumentations** for transformations.
- **Gradual Augmentation Strategy:**
  - **Early Epochs:** Basic augmentations.
  - **Later Epochs:** Advanced augmentations.

### **3. PyTorch Lightning Modules**
#### **AccuScanDataModule** (for dataset management)
- Handles training, validation, and testing datasets.
- Uses appropriate augmentations for different phases.

#### **AccuScanModelModule** (for model training)
- Wraps the **VisionEncoderDecoderModel** (Donut architecture).
- Implements gradient checkpointing for memory optimization.
- Uses special tokenizer tokens (`<s_ocr>` and `</s>`).
- Logs training & validation loss dynamically.

### **4. Training & Saving the Model**
- Loads `DonutProcessor` and model from Hugging Face.
- Configures optimizer and learning rate scheduler.
- **Callbacks Used:**
  - **ModelCheckpoint** (Saves the best model based on `val_loss`).
  - **EarlyStopping** (Stops training if no improvement in `val_loss`).
  - **GradualAugmentationCallback** (Switches augmentations after a set epoch).
- **Run Training:**
  ```python
  trainer.fit(model_module, datamodule=dm)
  ```

### **5. Model Evaluation**
- Load best checkpoint:
  ```python
  model_module = DonutModelModule.load_from_checkpoint(best_model_path, processor=processor)
  ```
- Compute **character-level and word-level accuracy**.
- Compare model predictions with ground truth.

## 🏆 Results
- **Character-Level Accuracy:** ~71.43%
- **Word-Level Accuracy:** ~84.24%
- Results depend on dataset quality and tuning. Further fine-tuning may improve performance.

## 📈 Sample Predictions
| **Reference** | **Prediction** |
|--------------|--------------|
| "Since 1958, 13 Labour life Peers and Peeresses have been created..." | "Since 1958, 13 Labour life Peers and Peeresses have been created..." |

## 🤖 Why These Libraries?
- **PyTorch** - Industry-standard deep learning framework.
- **PyTorch Lightning** - Simplifies training, logging, checkpointing.
- **Hugging Face Transformers** - Provides state-of-the-art OCR models.
- **Albumentations** - Efficient, high-quality image augmentations.
- **TQDM** - Progress bars for monitoring training and inference.

## 🎯 Future Improvements
- Implement **self-supervised pretraining** for OCR tasks.
- Introduce **self-correction mechanisms** for misread characters.
- Enhance **image preprocessing** techniques for noisy backgrounds.
- Expand **dataset diversity** to improve generalization.
