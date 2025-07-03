import os
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor, pipeline
import gradio as gr

# Dynamically determine the current directory and construct the relative path to the model folder
current_dir = os.path.dirname(os.path.abspath(__file__))
donut_model_path = os.path.join(current_dir, "model")

try:
    # Load the processor and model from the trained model directory
    processor = DonutProcessor.from_pretrained(donut_model_path)
    donut_model = VisionEncoderDecoderModel.from_pretrained(donut_model_path)
except Exception as e:
    print(f"Error loading Donut model: {e}")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
donut_model.to(device)
donut_model.eval()

try:
    # Initialize zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device == "cuda" else -1)
except Exception as e:
    print(f"Error initializing Zero-Shot Classifier: {e}")
    exit(1)

candidate_labels = ["medical prescription", "not medical prescription"]

def extract_text_from_image(image):
    """
    Extracts text from an image using the Donut OCR model.
    This function relies entirely on the DonutProcessor to handle preprocessing.
    """
    image = image.convert("RGB")  # Ensure RGB, as trained
    encoding = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = donut_model.generate(
            encoding.pixel_values, 
            max_length=512, 
            num_beams=1,
            early_stopping=True,
            decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<s_ocr>")
        )
    generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

medical_keywords = [
    "prescribed", "take", "mg", "ml", "capsules", "dosage",
    "dr.", "doctor", "patient", "medications", "apply", "signature",
    "clinic", "pharmacy", "rx", "dose", "medicine", "drug"
]

def classify_prescription_zero_shot(text):
    """
    Classifies the extracted text using Zero-Shot Classification and heuristic keywords.
    """
    if not text:
        return "No text found", 0.0

    result = classifier(text, candidate_labels)
    predicted_label = result["labels"][0]
    confidence = result["scores"][0]

    # Heuristic check
    text_lower = text.lower()
    has_medical_keywords = any(keyword in text_lower for keyword in medical_keywords)

    # Adjust prediction based on heuristics
    if predicted_label == "not medical prescription" and has_medical_keywords:
        predicted_label = "medical prescription"
        confidence = max(confidence, 0.75)
    elif predicted_label == "medical prescription" and not has_medical_keywords:
        predicted_label = "not medical prescription"
        confidence = max(confidence, 0.75)

    return predicted_label, confidence

def classify_image_gradio(image):
    """
    Processes the uploaded image, extracts text, and classifies it.
    """
    extracted_text = extract_text_from_image(image)
    predicted_label, confidence = classify_prescription_zero_shot(extracted_text)
    return extracted_text, predicted_label, round(confidence, 3)

demo = gr.Interface(
    fn=classify_image_gradio,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Extracted Text"),
        gr.Textbox(label="Predicted Label"),
        gr.Number(label="Confidence Score")
    ],
    title="Medical Prescription Classification",
    description="Upload an image containing handwritten text. The system will extract the text and determine whether it's a medical prescription."
    # No examples included.
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

