from openai import AzureOpenAI
import config
import json
import logging
import re
from ocr_extraction import extract_text_from_pdf  # Import OCR function

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    api_version=config.AZURE_OPENAI_API_VERSION,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT
)

# JSON template (Hebrew) defining fields to extract from the National Insurance form.
json_template_he = {
    "שם משפחה": "",
    "שם פרטי": "",
    "מספר זהות": "",
    "מין": "",
    "תאריך לידה": {"יום": "", "חודש": "", "שנה": ""},
    "כתובת": {
        "רחוב": "", "מספר בית": "", "כניסה": "",
        "דירה": "", "ישוב": "", "מיקוד": "", "תא דואר": ""
    },
    "טלפון קווי": "",
    "טלפון נייד": "",
    "סוג העבודה": "",
    "תאריך הפגיעה": {"יום": "", "חודש": "", "שנה": ""},
    "שעת הפגיעה": "",
    "מקום התאונה": "",
    "כתובת מקום התאונה": "",
    "תיאור התאונה": "",
    "האיבר שנפגע": "",
    "חתימה": "",
    "תאריך מילוי הטופס": {"יום": "", "חודש": "", "שנה": ""},
    "תאריך קבלת הטופס בקופה": {"יום": "", "חודש": "", "שנה": ""},
    "למילוי ע\"י המוסד הרפואי": {
        "חבר בקופת חולים": "", "מהות התאונה": "", "אבחנות רפואיות": ""
    }
}
# Mapping dictionary to translate Hebrew field names to English for UI presentation.
field_translation_map = {
    "שם משפחה": "lastName",
    "שם פרטי": "firstName",
    "מספר זהות": "idNumber",
    "מין": "gender",
    "תאריך לידה": "dateOfBirth",
    "יום": "day",
    "חודש": "month",
    "שנה": "year",
    "כתובת": "address",
    "רחוב": "street",
    "מספר בית": "houseNumber",
    "כניסה": "entrance",
    "דירה": "apartment",
    "ישוב": "city",
    "מיקוד": "postalCode",
    "תא דואר": "poBox",
    "טלפון קווי": "landlinePhone",
    "טלפון נייד": "mobilePhone",
    "סוג העבודה": "jobType",
    "תאריך הפגיעה": "dateOfInjury",
    "שעת הפגיעה": "timeOfInjury",
    "מקום התאונה": "accidentLocation",
    "כתובת מקום התאונה": "accidentAddress",
    "תיאור התאונה": "accidentDescription",
    "האיבר שנפגע": "injuredBodyPart",
    "חתימה": "signature",
    "תאריך מילוי הטופס": "formFillingDate",
    "תאריך קבלת הטופס בקופה": "formReceiptDateAtClinic",
    "למילוי ע\"י המוסד הרפואי": "medicalInstitutionFields",
    "חבר בקופת חולים": "healthFundMember",
    "מהות התאונה": "natureOfAccident",
    "אבחנות רפואיות": "medicalDiagnoses"
}
def clean_text(text: str) -> str:
    """
    Cleans up extracted text by removing extra spaces, newlines, and special characters.
    """
    text = text.replace("\n", " ").strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

def get_low_confidence_words_from_json(json_data: dict, word_confidences: list, threshold: float = 0.75):
    """
      Identifies OCR-extracted words within the JSON data having confidence below a threshold.
      Useful for validating extraction accuracy.

      Args:
          json_data (dict): Extracted data JSON.
          word_confidences (list): List of OCR words with confidence scores.
          threshold (float): Confidence threshold for identifying problematic words.

      Returns:
          list: Words from JSON with confidence below threshold.
      """
    def extract_values(obj):
        if isinstance(obj, dict):
            return [v for val in obj.values() for v in extract_values(val)]
        elif isinstance(obj, list):
            return [v for item in obj for v in extract_values(item)]
        elif isinstance(obj, str):
            return [obj]
        return []

    json_values = extract_values(json_data)
    all_words = set()
    for val in json_values:
        all_words.update(re.findall(r"\w+", val))

    low_conf_words = [
        {"text": wc["text"], "confidence": wc["confidence"]}
        for wc in word_confidences
        if wc["text"] in all_words and wc["confidence"] < threshold
    ]

    return low_conf_words

def clean_gpt_response(json_output: str) -> str:
    """
    Remove triple-backticks (```...```) from the GPT response so it can be parsed as valid JSON.
    """
    json_output = json_output.strip()
    json_output = re.sub(r'^```[a-zA-Z]*\s*\n?', '', json_output)
    json_output = re.sub(r'\n?```$', '', json_output)
    return json_output.strip()
def generate_json_from_text(ocr_text: str, word_confidences) -> dict:
    """
      Uses Azure OpenAI GPT to extract structured data (JSON) from OCR text.
      Returns extracted data in Hebrew (for UI mapping).
      """

    prompt = f"""
    You are an expert in extracting structured data from OCR text.
    The following text was extracted from a National Insurance Institute form, possibly in Hebrew or English.

    Please extract the fields and format them into valid JSON:
    {json.dumps(json_template_he, indent=2, ensure_ascii=False)}

    If any field is missing, return an empty string.

    Here is the extracted text:
    {clean_text(ocr_text)}

    Respond ONLY with the JSON object (no explanations).
    """

    try:
        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a JSON extraction expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1000
        )

        # Extract the assistant's text (JSON) from the response
        json_output = response.choices[0].message.content
        json_output = clean_gpt_response(json_output)

        try:
            extracted_data = json.loads(json_output)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON, returning empty structure.")
            logging.error(f"GPT Response: {json_output}")
            extracted_data = json_template_he

        # # Create a confidence map for extracted fields
        # confidence_map = {word["text"]: word["confidence"] for word in word_confidences}

        for key in json_template_he:
            if key not in extracted_data:
                extracted_data[key] = json_template_he[key]

        return extracted_data

    except json.JSONDecodeError:
        logging.error("Failed to parse JSON, returning empty structure.")
        logging.error(f"GPT Response: {json_output}")
        return json_template_he

    except Exception as e:
        logging.error(f"Error calling Azure OpenAI: {str(e)}")
        return json_template_he


def translate_json_to_english(hebrew_json):
    """
        Translates Hebrew JSON structure to English using the predefined mapping.

        Args:
            hebrew_json (dict): Extracted data in Hebrew.

        Returns:
            dict: JSON data translated to English fields for UI display.
        """
    def translate(item):
        if isinstance(item, dict):
            return {field_translation_map.get(k, k): translate(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [translate(v) for v in item]
        else:
            return item

    return translate(hebrew_json)