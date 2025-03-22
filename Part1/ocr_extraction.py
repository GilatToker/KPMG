import os
import logging
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import time

POLLING_INTERVAL = 1
MAX_ATTEMPTS = 10

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load API credentials from .env
load_dotenv()

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

# Ensure API keys are loaded correctly
if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or not AZURE_DOCUMENT_INTELLIGENCE_KEY:
    raise ValueError("API Key or Endpoint not found. Check your .env file!")

# Initialize the Azure Document Intelligence client
client = DocumentIntelligenceClient(AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                                    AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY))

def extract_text_from_pdf(file_path):
    """
        Extracts text from a given PDF or image using Azure Document Intelligence.
        :param file_path: Path to the PDF or image file.
        :return: Extracted text or error message.
        """
    try:
        logging.info(f"Processing file: {file_path}")
        with open(file_path, "rb") as file:
            poller = client.begin_analyze_document("prebuilt-layout", file)

        attempts = 0
        while not poller.done():
            if attempts >= MAX_ATTEMPTS:
                logging.error("OCR processing timeout.")
                return "ERROR: OCR processing timeout.", []
            time.sleep(POLLING_INTERVAL)
            attempts += 1

        result = poller.result()
        extracted_text = []
        word_confidences = []

        for page in result.pages:
            page_text = []
            for line in page.lines:
                page_text.append(line.content)
            extracted_text.append("\n".join(page_text))

            for word in page.words:
                word_confidences.append({"text": word.content, "confidence": word.confidence})

        return "\n".join(extracted_text), word_confidences

    except Exception as e:
        logging.error(f"Error during OCR processing: {str(e)}")
        return f"ERROR: {str(e)}", []


if __name__ == "__main__":
    # Example file (Replace with your actual file path)
    file_path = "phase1_data/283_ex1.pdf"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        print("Extracting text... Please wait.")
        extracted_text, word_confidences = extract_text_from_pdf(file_path)

        print("\n--- Extracted Text ---\n")
        print(extracted_text)

        print("\n--- Word Confidences ---\n")
        for wc in word_confidences:
            print(f"Word: '{wc['text']}', Confidence: {wc['confidence']:.2f}")