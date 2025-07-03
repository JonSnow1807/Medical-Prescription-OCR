from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load model and processor
processor = DonutProcessor.from_pretrained("chinmays18/medical-prescription-ocr")
model = VisionEncoderDecoderModel.from_pretrained("chinmays18/medical-prescription-ocr")

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Process an image
image = Image.open("prescription.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate text
task_prompt = "<s_ocr>"
decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=512,
    num_beams=1,
    early_stopping=True
)

# Decode output
prescription_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(prescription_text)

Advanced Usage with Classification
python# The app.py includes zero-shot classification to verify if an image is a prescription
# See app.py for complete implementation including confidence scoring
📚 Dataset
Our model was trained on a carefully curated dataset of synthetic medical prescriptions:

Total Samples: 1,000 high-quality images
Training Set: 800 samples
Validation Set: 100 samples
Test Set: 100 samples
Format: PNG images with JSON annotations
Annotations: Structured text including doctor info, patient details, medications, and dosages

Access the full dataset: chinmays18/medical-prescription-dataset
Dataset Structure
prescription_XXXXX.png  → Image file
prescription_XXXXX.json → Annotation with ground truth text
🛠️ Training
The model training pipeline is fully documented in OCR_training.ipynb.
Training Highlights

Base Model: NAVER Clova Donut (Document Understanding Transformer)
Training Strategy: Gradual augmentation (basic → advanced)
Framework: PyTorch Lightning for clean, reproducible training
Optimization: AdamW with linear warmup
Hardware: Trained on NVIDIA GPU with mixed precision

Key Innovations

Gradual Augmentation: Starts with light augmentations, progressively introduces harder ones
Smart Callbacks: Early stopping, model checkpointing, and custom augmentation scheduling
Efficient Training: Gradient checkpointing and mixed precision for memory efficiency

🔧 Tech Stack
ComponentTechnologyPurposeCore FrameworkPyTorch 2.0+Deep learning foundationTrainingPyTorch LightningClean training loops, loggingModel ArchitectureDonut (NAVER)State-of-the-art document OCRData AugmentationAlbumentationsFast, flexible augmentationsTokenizationSentencePieceSubword tokenizationClassificationBART (Facebook)Zero-shot classificationInterfaceGradioWeb applicationModel HostingHugging Face HubModel distribution
📁 Project Structure
medical-prescription-ocr/
├── app.py                  # Gradio web application
├── model_download.py       # Model download utility
├── OCR_training.ipynb      # Complete training pipeline
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── README.md              # This file
└── model/                 # Downloaded model files (after setup)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
🤝 Contributing
Contributions are welcome! Here's how you can help:

🐛 Report Bugs: Open an issue with details
💡 Suggest Features: Share your ideas in discussions
🔧 Submit PRs: Fork, create a feature branch, and submit a pull request

Development Setup
bash# Clone your fork
git clone https://github.com/YOUR_USERNAME/medical-prescription-ocr.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install jupyter black pytest
⚠️ Important Notes

Research Use Only: This model is NOT validated for clinical use
Synthetic Data: Trained on synthetic prescriptions, not real patient data
No Medical Advice: Should not be used for actual medical prescription processing
Privacy: Never upload real patient prescriptions to the demo

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

NAVER Clova AI - For the amazing Donut architecture
Hugging Face - For model and dataset hosting
Facebook Research - For BART zero-shot classifier
IAM Handwriting Database - Inspiration for dataset structure

👤 Author
Chinmay Shrivastava

MS Computer Science & Engineering
AI/ML Engineer passionate about healthcare applications
GitHub: @chinmays18
Hugging Face: @chinmays18
