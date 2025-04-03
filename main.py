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
    
    # Refine text using Mistral AI to retain only the main story and title, removing decorative text and unwanted characters
    prompt = ("Extract only the main story and title from this text, removing any decorative text, captions, page numbers, "
              "special characters, and formatting errors while keeping the correct sentence flow. "
              "For example, if the given text is:\n"
              "इन घंटियों को बनाना ही सबसे ज़्यादा मुश्किल था। केशव जानता था कि एक\n"
              "दिन तो ऐसा ज़रूर आएगा जब वह बहुत बारीक जालियाँ, महीन-नफ़ीस\n"
              "बेल-बूटे, कमल के फूल, लहराते हुए साँप और इठलाकर चलते हुए\n"
              "घोड़े-ये सब पत्थर पर उकेर पाएगा... ठीक उसी\n"
              "तरह जैसे उसके पिता बनाते हैं।\n"
              "कछ १//९७९७-.. कई साल पहले, जब वह पैदा भी नहीं\n"
              "9, औ ८७) आ ' हुआ था, उसके माता-पिता गुजरात से आगरा\n"
              "/ 8/ ० “8 आकर बस गए थे। बादशाह\n"
              "१५०७ के ५ 67 अकबर उस वक्‍त आगरे का\n"
              "रे आज हाँ; 2727 किला बनवा रहे थे और केशव\n"
              "<्क्यट्यद्रस्यट) (75५ ५ ९२० ८ के पिता को यहीं\n"
              "जज 23 /  ठस 4 / 27 “30 ८-2 |» केशव का जन्म भी\n"
              "के ५9 08/£/ 5 | मल 0 डे हा\n"
              "का ०४5, डक के, कि “का उ है मं न\n"
              "का हु 7२७७॥॥ 2024-25\n"
              "Then the cleaned-up version should be:\n"
              "इन घंटियों को बनाना ही सबसे ज़्यादा मुश्किल था। केशव जानता था कि एक\n"
              "दिन तो ऐसा ज़रूर आएगा जब वह बहुत बारीक जालियाँ, महीन-नफ़ीस\n"
              "बेल-बूटे, कमल के फूल, लहराते हुए साँप और इठलाकर चलते हुए\n"
              "घोड़े-ये सब पत्थर पर उकेर पाएगा... ठीक उसी\n"
              "तरह जैसे उसके पिता बनाते हैं।\n"
              "कई साल पहले, जब वह पैदा भी नहीं\n"
              "हुआ था, उसके माता-पिता गुजरात से आगरा\n"
              "आकर बस गए थे। बादशाह\n"
              "अकबर उस वक्‍त आगरे का\n"
              "किला बनवा रहे थे और केशव\n"
              "के पिता को यहीं\n"
              "केशव का जन्म भी\n"
              "Now process the following text accordingly: " + extracted_text)
    
    chat_response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    refined_text = chat_response.choices[0].message.content

    # Save the refined text for this page
    refined_filename = os.path.join(REFINED_OUTPUT_DIR, f"page_{i + 1}.txt")
    with open(refined_filename, "w", encoding="utf-8") as f:
        f.write(refined_text)

    print(f"Page {i + 1} saved: Raw -> {raw_filename}, Refined -> {refined_filename}")

print("All pages processed! Extracted text saved in 'raw_text' folder and refined text saved in 'refined_text' folder.")
