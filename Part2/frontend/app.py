import numpy as np
import streamlit as st
# Set up the Streamlit page configuration
st.set_page_config("HMO Chatbot")

from openai import AzureOpenAI
import os
import json
import glob
import re
from dotenv import load_dotenv
import logging
import config
import requests
from langdetect import detect

# Load environment variables (ensure you have a .env file with the required keys)
load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

EMBEDDING_MODEL = "text-embedding-ada-002"

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
# Multi-language support messages (English and Hebrew)
MESSAGES = {
    "greeting": {
        "en": "Hello, I am your assistant. To provide you with the best support, let's begin by collecting your personal details. How you are feeling today?",
        "he": "שלום, אני העוזר האישי שלך. כדי לספק לך את השירות הטוב ביותר, נתחיל באיסוף פרטים אישיים. לפני כן, מה שלומך היום?"
    },
    "ask_first_name": {
        "en": "Let's start by collecting your details. Please enter your first name.",
        "he": "בוא נתחיל באיסוף הפרטים שלך. אנא הזן/י את שמך הפרטי."
    },
    "ask_last_name": {
        "en": "Great! Please enter your last name.",
        "he": "מצוין! אנא הזן/י את שם המשפחה שלך."
    },
    "ask_id_number": {
        "en": "Please enter your ID number (9 digits).",
        "he": "אנא הזן/י את תעודת הזהות שלך (9 ספרות)."
    },
    "ask_gender": {
        "en": "Please enter your gender.",
        "he": "אנא הזן/י את המגדר שלך."
    },
    "ask_age": {
        "en": "Please enter your age.",
        "he": "אנא הזן/י את גילך."
    },
    "ask_hmo_name": {
        "en": "Please enter your health fund. Options: מכבי, מאוחדת, כללית.",
        "he": "אנא הזן/י את קופת החולים שלך. אפשרויות: מכבי, מאוחדת, כללית."
    },
    "ask_hmo_card_number": {
        "en": "Please enter your HMO card number (9 digits).",
        "he": "אנא הזן/י את מספר כרטיס הקופה (9 ספרות)."
    },
    "ask_insurance_tier": {
        "en": "Please enter your insurance tier. Options: זהב, כסף, ארד.",
        "he": "אנא הזן/י את רמת הביטוח שלך. אפשרויות: זהב, כסף, ארד."
    },
    "confirm": {
        "en": "Please confirm your details:",
        "he": "אנא אשר/י את הפרטים שלך:"
    },
    "confirm_input": {
        "en": "Please type 'confirm' to proceed or explain which correction you'd like to make.",
        "he": "אנא הקלד/י 'אשר' כדי להמשיך או תפרטי מה תרצי לשנות."
    },
    "qa_phase": {
        "en": "How can I assist you? Please let me know your question or the topic you need help with.",
        "he": "איך אני יכול לעזור לך? אנא ספר/י מה השאלה או הנושא שבו אתה זקוק לעזרה."
    }
}
def get_message(key):
    lang = st.session_state.get("user_info", {}).get("language", "en")
    return MESSAGES.get(key, {}).get(lang, MESSAGES.get(key, {}).get("en", ""))
def detect_language(text):
    try:
        detected_lang = detect(text)
        if detected_lang == "he":
            return "he"
        else:
            return "en"
    except Exception as e:
        return "en"

def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to get embedding for text. Error: {e}")
        return None


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def semantic_search_knowledge_base(query, kb, kb_embeddings, top_k=3):
    """
       Performs semantic search using embeddings to find top matching paragraphs
       from the knowledge base for the user's query .
       """
    query_embedding = get_embedding(query)
    scored = []
    for key, content in kb.items():
        emb = kb_embeddings.get(key)
        if emb is None:
            continue
        score = cosine_similarity(query_embedding, emb)
        scored.append((score, key, content))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_matches = scored[:top_k]

    # Combine matched paragraphs and metadata as context for the LLM
    combined_snippet = "\n\n---\n\n".join(
        [
            f"{match[2]['text']} (Source: {match[2]['metadata']['filename']}, Paragraph: {match[2]['metadata']['para_num']})"
            for match in top_matches]
    )

    logging.debug(f"Top matches: {[match[1] for match in top_matches]}")

    return combined_snippet, [match[1] for match in top_matches]


# ---------------------------
# Load Knowledge Base
# ---------------------------
@st.cache_data
def load_knowledge_base():
    """
    Loads and splits HTML knowledge base files into paragraphs, storing each with
    metadata (filename and paragraph number). Used for improved retrieval accuracy.
    """
    kb = {}
    for filepath in glob.glob("phase2_data/*.html"):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
            text = re.sub("<[^<]+?>", "", raw_text)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs):
                key = f"{os.path.basename(filepath)}_para_{idx}"
                kb[key] = {
                    "text": para,
                    "metadata": {
                        "filename": os.path.basename(filepath),
                        "para_num": idx
                    }
                }
    return kb



@st.cache_data
def precompute_embeddings(kb):
    """Pre-computes embeddings for knowledge base paragraphs for faster retrieval."""
    embeddings = {}
    for key, content in kb.items():
        emb = get_embedding(content["text"][:5000])
        embeddings[key] = emb
    return embeddings


def extract_field(field_name, user_input):
    st.info("Validating your input, please wait...")
    # בונים prompt מותאם לכל שדה
    system_prompt = f"You are an assistant that extracts a valid {field_name}. The expected format for {field_name} is: "
    if field_name == "ID number":
        system_prompt += (
            "a valid ID number is exactly 9 digits long. "
            "If the input contains multiple numbers, extract the one that is exactly 9 digits. "
            "If no valid 9-digit number is found, return 'Invalid'."
        )
    elif field_name == "age":
        system_prompt += (
            "a valid age is an integer between 0 and 120. "
            "If the input contains additional text, extract the integer value. "
            "If no valid age is found, return 'Invalid'."
        )
    elif field_name == "HMO name":
        system_prompt += (
            "a valid HMO name is one of the following: מכבי, מאוחדת, כללית. "
            "If the input is not one of these, return 'Invalid'."
        )
    elif field_name == "HMO card number":
        system_prompt += (
            "a valid HMO card number is exactly 9 digits long. "
            "If the input contains multiple numbers, extract the one that is exactly 9 digits. "
            "If no valid 9-digit number is found, return 'Invalid'."
        )
    elif field_name == "insurance membership tier":
        system_prompt += (
            "a valid insurance membership tier is one of the following: זהב, כסף, ארד. "
            "If the input is not one of these, return 'Invalid'."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract the valid {field_name} from the following input: '{user_input}'."}
    ]
    try:
        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=50
        )
        extracted = response.choices[0].message.content.strip()
        if extracted == "Invalid":
            return None
        return extracted
    except Exception as e:
        logging.error(f"Error extracting field {field_name}: {e}")
        return None
def analyze_confirmation(user_input):
    system_prompt = (
        "You are an assistant that analyzes a user's response regarding confirmation of their details. "
        "The user input can be in English or Hebrew. "
        "The available fields in the user data are: first_name, last_name, id_number, gender, age, hmo_name, hmo_card_number, insurance_tier. "
        "If the response indicates that the user wants to proceed, respond with the single word 'confirm'. "
        "If the response indicates that the user wants to make corrections, respond with a JSON object in the following format: "
        '{"action": "edit", "field": "<field_name>", "new_value": "<new_value>"} '
        "Make sure the JSON is valid. Only output either 'confirm' or the JSON object."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": 'תשנה את המגדר שלי לזכר'},
        {"role": "assistant", "content": '{"action": "edit", "field": "gender", "new_value": "זכר"}'},
        {"role": "user", "content": user_input}
    ]
    try:
        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=100
        )
        result = response.choices[0].message.content.strip().lower()
        if result.lower() == "confirm":
            return {"action": "confirm"}
        else:
            logging.debug(f"result: {result}")
            try:
                parsed = json.loads(result)
                if parsed.get("action") == "edit" and "field" in parsed and "new_value" in parsed:
                    return parsed
                else:
                    return None
            except Exception as e:
                logging.error(f"Error parsing JSON from analyze_confirmation: {e}")
                return None
    except Exception as e:
        logging.error(f"Error analyzing confirmation: {e}")
        return None

# ---------------------------
# Load KnowledgeBase
# ---------------------------
knowledge_base = load_knowledge_base()
kb_embeddings = precompute_embeddings(knowledge_base)
logging.debug(f"Number of files in knowledge_base: {len(knowledge_base)}")
logging.debug(f"Number of embeddings in kb_embeddings: {len(kb_embeddings)}")


#######################
# Streamlit + Chat Flow
#######################
st.title("HMO Chatbot - מידע רפואי לפי קופות החולים בישראל")

# Client-side state management with Streamlit session state
if "phase" not in st.session_state:
    st.session_state["phase"] = "greeting"
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {}
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

chat_placeholder = st.container()

def build_conversation_history():
    """
    Converts st.session_state["chat_history"] to the format:
    [
        {"user": "...", "bot": "..."},
        {"user": "...", "bot": "..."},
        ...
    ]
    This is what the backend /chat endpoint expects.
    """
    conversation = []
    user_text = None

    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            user_text = msg["content"]
        elif msg["role"] == "assistant":
            if user_text is not None:
                conversation.append({
                    "user": user_text,
                    "bot": msg["content"]
                })
                user_text = None
    return conversation

def add_message(role, content):
    """Adds a message to the Streamlit session chat history."""
    st.session_state["chat_history"].append({"role": role, "content": content})
    logging.info(f"{role.upper()}: {content}")

def render_chat():
    """Renders chat history visually in Streamlit."""
    chat_placeholder.empty()
    for msg in st.session_state["chat_history"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

# Initialize greeting if chat history is empty
if st.session_state["phase"] == "greeting" and not st.session_state["chat_history"]:
    add_message("assistant", get_message("greeting"))

# Function to process a new user message based on the current phase
def process_message(message):
    phase = st.session_state["phase"]

    # Phase: Greeting – detect language and move to first name request
    if phase == "greeting":
        detected_lang = detect_language(message)
        st.session_state["user_info"]["language"] = detected_lang
        add_message("assistant", f"Detected language: {detected_lang}")
        st.session_state["phase"] = "ask_first_name"
        add_message("assistant", get_message("ask_first_name"))

    # Phase: Ask First Name
    if phase == "ask_first_name":
        if message.strip() == "":
            add_message("assistant", "Please enter a valid first name.")
            return
        st.session_state["user_info"]["first_name"] = message.strip()
        st.session_state["phase"] = "ask_last_name"
        add_message("assistant", get_message("ask_last_name"))

    # Phase: Ask Last Name
    if phase == "ask_last_name":
        if message.strip() == "":
            add_message("assistant", "Please enter a valid last name.")
            return
        st.session_state["user_info"]["last_name"] = message.strip()
        st.session_state["phase"] = "ask_id_number"
        add_message("assistant", get_message("ask_id_number"))

    # Phase: Ask ID Number
    if phase == "ask_id_number":
        answer = message.strip()
        valid_id = answer if (len(answer) == 9 and answer.isdigit()) else None
        if valid_id is None:
            extracted = extract_field("ID number", answer)
            logging.debug(f"Extracted ID number: {extracted}")
            valid_id = extracted
        if valid_id:
            st.session_state["user_info"]["id_number"] = valid_id
            add_message("assistant", f"Extracted valid ID number: {valid_id}")
            st.session_state["phase"] = "ask_gender"
            add_message("assistant", get_message("ask_gender"))
        else:
            add_message("assistant", "ID number is invalid and could not be extracted. Please try again.")

    # Phase: Ask Gender
    if phase == "ask_gender":
        if message.strip() == "":
            add_message("assistant", "Please enter a valid gender.")
            return
        st.session_state["user_info"]["gender"] = message.strip()
        st.session_state["phase"] = "ask_age"
        add_message("assistant", get_message("ask_age"))

    # Phase: Ask Age
    if phase == "ask_age":
        answer = message.strip()
        valid_age = None
        try:
            age_val = int(answer)
            if 0 <= age_val <= 120:
                valid_age = age_val
        except ValueError:
            pass
        if valid_age is None:
            extracted = extract_field("age", answer)
            logging.debug(f"Extracted age: {extracted}")
            valid_age = extracted
        if valid_age is not None:
            st.session_state["user_info"]["age"] = valid_age
            add_message("assistant", f"Extracted valid age: {valid_age}")
            st.session_state["phase"] = "ask_hmo_name"
            add_message("assistant", get_message("ask_hmo_name"))
        else:
            add_message("assistant", "Age is invalid and could not be extracted. Please try again.")


    # Phase: Ask HMO Name
    if phase == "ask_hmo_name":
        answer = message.strip()
        valid_hmo = answer if answer in ["מכבי", "מאוחדת", "כללית"] else None
        if valid_hmo is None:
            extracted = extract_field("HMO name", answer)
            logging.debug(f"Extracted HMO name: {extracted}")
            valid_hmo = extracted
        if valid_hmo:
            st.session_state["user_info"]["hmo_name"] = valid_hmo
            add_message("assistant", f"Extracted valid health fund: {valid_hmo}")
            st.session_state["phase"] = "ask_hmo_card_number"
            add_message("assistant", get_message("ask_hmo_card_number"))
        else:
            add_message("assistant", "Health fund is invalid and could not be extracted. Please try again.")

    # Phase: Ask HMO Card Number
    if phase == "ask_hmo_card_number":
        answer = message.strip()
        valid_card = answer if (len(answer) == 9 and answer.isdigit()) else None
        if valid_card is None:
            extracted = extract_field("HMO card number", answer)
            logging.debug(f"Extracted HMO card number: {extracted}")
            valid_card = extracted
        if valid_card:
            st.session_state["user_info"]["hmo_card_number"] = valid_card
            add_message("assistant", f"Extracted valid HMO card number: {valid_card}")
            st.session_state["phase"] = "ask_insurance_tier"
            add_message("assistant", get_message("ask_insurance_tier"))
        else:
            add_message("assistant", "HMO card number is invalid and could not be extracted. Please try again.")

    # Phase: Ask Insurance Tier
    if phase == "ask_insurance_tier":
        answer = message.strip()
        valid_tier = answer if answer in ["זהב", "כסף", "ארד"] else None
        if valid_tier is None:
            extracted = extract_field("insurance membership tier", answer)
            logging.debug(f"Extracted insurance tier: {extracted}")
            valid_tier = extracted
        if valid_tier:
            st.session_state["user_info"]["insurance_tier"] = valid_tier
            add_message("assistant", f"Extracted valid insurance tier: {valid_tier}")
            st.session_state["phase"] = "confirm"
            add_message("assistant", get_message("confirm"))
            add_message("assistant", f"Your details: {st.session_state['user_info']}")
            add_message("assistant","Type 'confirm' to proceed to chat, or type 'edit' with the details of the field you want to update.")
        else:
            add_message("assistant", "Insurance tier is invalid and could not be extracted. Please try again.")

    # Phase: Confirm details
    if phase == "confirm":
        analysis = analyze_confirmation(message)
        if analysis is None:
            add_message("assistant", "Your response was unclear. " + get_message("confirm_input"))
            return
        if analysis.get("action") == "confirm":
            st.session_state["chat_history"] = []
            st.session_state["phase"] = "qa"
            add_message("assistant", get_message("qa_phase"))
        elif analysis.get("action") == "edit":
            field_to_edit = analysis.get("field")
            new_value = analysis.get("new_value")
            if field_to_edit and new_value:
                st.session_state["user_info"][field_to_edit] = new_value
                add_message("assistant", f"Updated {field_to_edit} to {new_value}.")
                add_message("assistant", f"Your updated details: {st.session_state['user_info']}")
                add_message("assistant", get_message("confirm_input"))
            else:
                add_message("assistant", "Could not parse the edit details. " + get_message("confirm_input"))
        else:
            add_message("assistant", "Your response was unclear. " + get_message("confirm_input"))


    # PHASE: Q&A
    if phase == "qa":
        question = message.strip()
        if not question:
            add_message("assistant", "Please enter a valid question.")
            return

        with st.spinner("Searching for an answer..."):
            snippet, source_doc = semantic_search_knowledge_base(question, knowledge_base, kb_embeddings, top_k=4)
            logging.debug(f"Snippet retrieved from file: {source_doc}")

            if not snippet:
                add_message("assistant", "Sorry, I couldn't find relevant information in the knowledge base.")
                return

            conversation_history_for_server = build_conversation_history()
            logging.debug(f"[FRONTEND] chat_history = {st.session_state['chat_history']}")
            logging.debug(f"[FRONTEND] conversation_history_for_server = {conversation_history_for_server}")

            payload = {
                "user_info": st.session_state["user_info"],
                "question": question,
                "language": st.session_state["user_info"].get("language", "en"),
                "context": snippet,
                "conversation_history": []
            }
            logging.debug("[FRONTEND] Payload being sent = " + json.dumps(payload, ensure_ascii=False, indent=2))

            try:
                resp = requests.post("http://localhost:8000/chat", json=payload)
                if resp.status_code != 200:
                    add_message("assistant", f"Error from server: {resp.text}")
                    return
                data = resp.json()
                answer_text = data.get("answer", "")
                if not answer_text:
                    add_message("assistant", "Server returned empty answer.")
                    return
                # Now we add both user & assistant messages
                add_message("user", question)
                add_message("assistant", answer_text)
            except Exception as e:
                logging.error(f"Error parsing JSON: {e}")
                add_message("assistant", f"Error parsing server response: {str(e)}")

        return

new_message = st.chat_input("Type your message here...")
if new_message:
    if st.session_state["phase"] == "qa":
        process_message(new_message)
    else:
        add_message("user", new_message)
        process_message(new_message)
render_chat()