import os
from pdf2image import convert_from_path
import pytesseract
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

# === CONFIGURATION ===
PDF_PATH = "data/ehhn104.pdf"  # PDF file path
RAW_OUTPUT_DIR = "raw_text"  # Directory to store raw extracted text
REFINED_OUTPUT_DIR = "refined_text"  # Directory to store refined text
LANG = "hin"  # Hindi OCR language

# Create output directories if they don't exist
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(REFINED_OUTPUT_DIR, exist_ok=True)

# Initialize Mistral API
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# === STEP 1: Convert PDF to Images ===
print("Converting PDF to Images...")
images = convert_from_path(PDF_PATH)

# === STEP 2: Extract and Refine Text Page by Page ===
print("Extracting and Refining Text Using OCR...")

custom_config = r'--oem 3 --psm 6 -l hin'

for i, img in enumerate(images):
    print(f"Processing Page {i + 1}/{len(images)}...")

    # Extract text from the current page
    extracted_text = pytesseract.image_to_string(img, config=custom_config)
    
    # Save raw extracted text
    raw_filename = os.path.join(RAW_OUTPUT_DIR, f"page_{i + 1}.txt")
    with open(raw_filename, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    # Refine text using Mistral AI to retain only the main story and title
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": f"Extract only the story text, removing any unnecessary elements like page numbers, footnotes, or other unrelated content: {extracted_text}"}
        ]
    )
    refined_text = chat_response.choices[0].message.content

    # Save the refined text for this page
    refined_filename = os.path.join(REFINED_OUTPUT_DIR, f"page_{i + 1}.txt")
    with open(refined_filename, "w", encoding="utf-8") as f:
        f.write(refined_text)

    print(f"Page {i + 1} saved: Raw -> {raw_filename}, Refined -> {refined_filename}")

print("All pages processed! Extracted text saved in 'raw_text' folder and refined text saved in 'refined_text' folder.")
