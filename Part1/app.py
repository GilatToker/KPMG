import streamlit as st
import json
import os
import tempfile
import config

from ocr_extraction import extract_text_from_pdf
from parse_ocr_to_json import (
    generate_json_from_text,
    translate_json_to_english,
    get_low_confidence_words_from_json
)
def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded file using OCR extraction and Azure OpenAI for structured data extraction.

    Args:
        uploaded_file (UploadedFile): Streamlit UploadedFile object.

    Returns:
        tuple: Structured JSON in Hebrew, English, and list of low-confidence words.
    """
    # Save uploaded file temporarily for OCR processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    structured_json_he, structured_json_en, low_conf_words = None, None, []

    try:
        # Perform OCR extraction from PDF/image file
        extracted_text, word_confidences = extract_text_from_pdf(temp_file_path)
        if isinstance(extracted_text, str) and extracted_text.startswith("ERROR"):
            raise ValueError(extracted_text)

        # Extract structured data using Azure OpenAI
        structured_json_he = generate_json_from_text(extracted_text, word_confidences )
        structured_json_en = translate_json_to_english(structured_json_he)

        # Get low-confidence words
        low_conf_words = get_low_confidence_words_from_json(structured_json_he, word_confidences)

    except Exception as e:
        st.error(f"Extraction Error: {e}")
        structured_json_he, structured_json_en, low_conf_words = None, None, []
    finally:
        # Ensure temporary file is deleted after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return structured_json_he, structured_json_en, low_conf_words

# Highlight low-confidence words directly in JSON
def highlight_low_conf_words_in_json(json_obj, low_conf_words):
    """
       Highlights low-confidence OCR-extracted words directly in JSON data for easier validation.

       Args:
           json_obj (dict): Structured JSON data.
           low_conf_words (list): List of words with confidence scores below threshold.

       Returns:
           dict: JSON data with low-confidence words annotated.
       """
    low_word_map = {w['text']: w['confidence'] for w in low_conf_words}
    def recursive_mark(obj):
        if isinstance(obj, dict):
            return {k: recursive_mark(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_mark(v) for v in obj]
        elif isinstance(obj, str):
            words = obj.split()
            marked = [f"{w} (⚠️ {low_word_map[w]*100:.0f}%)" if w in low_word_map else w for w in words]
            return " ".join(marked)
        return obj

    return recursive_mark(json_obj)

def main():
    st.set_page_config(page_title="NI Form Extractor", layout="centered")
    st.title("National Insurance Form Extraction")

    # User Guide**
    with st.expander("ℹ️ How It Works - Click to Expand!"):
        st.markdown("""
            **Welcome!** Here's how to use this tool:  

            **Step 1:** Upload a PDF or Image containing a National Insurance form.  
            **Step 2:** The system will automatically extract the form details for you.  
            **Step 3:** You can switch between **Hebrew** and **English** views for convenience.
            - Some extracted fields may have **low confidence** - These fields will be marked with a **⚠️ symbol**. Please double-check them carefully!  
            """)

    # Initialize session state explicitly
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
        st.session_state.structured_json_he = None
        st.session_state.structured_json_en = None
        st.session_state.confidence_scores = []

    uploaded_file = st.file_uploader(
        "Upload a PDF or Image (JPG/JPEG/PNG)",
        type=["pdf", "jpg", "jpeg", "png"]
    )

    # Process file only if a new file is uploaded
    if uploaded_file:
        if uploaded_file.name != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file.name
            st.success(f"File uploaded: {uploaded_file.name}")

            # Run extraction pipeline
            structured_json_he, structured_json_en, low_conf_words = process_uploaded_file(uploaded_file)

            if structured_json_he is None:
                st.error("Processing failed.")
                return

            # Highlight low-confidence fields in JSON data
            st.session_state.structured_json_he = highlight_low_conf_words_in_json(structured_json_he, low_conf_words)
            st.session_state.structured_json_en = highlight_low_conf_words_in_json(structured_json_en, low_conf_words)
            st.session_state.low_conf_words = low_conf_words

            st.success("Extraction completed!")

    # Display extraction results with language selection
    if st.session_state.structured_json_he:
        view_language = st.radio("Select view language:", ["עברית", "English"], horizontal=True)
        if view_language == "עברית":
            st.subheader("תוצאה בעברית")
            st.json(st.session_state.structured_json_he)
        else:
            st.subheader("Result in English")
            st.json(st.session_state.structured_json_en)




if __name__ == "__main__":
    main()
