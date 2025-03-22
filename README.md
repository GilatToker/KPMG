# KPMG AI Project

This repository contains **two main parts**:

1. **Part 1: OCR Form Extraction**  
   Extracts structured data from National Insurance forms (ביטוח לאומי) using Azure Document Intelligence and Azure OpenAI.

2. **Part 2: HMO Chatbot**  
   Answers medical questions related to Israeli health funds (מכבי, מאוחדת, כללית) based on user info, using FastAPI (for backend) and Streamlit (for frontend).

---

## Summary of Project Files and Their Roles

### **Part 1**

#### **`ocr_extraction.py`**
**Purpose**  
Responsible for performing OCR on PDF or image files using **Azure Document Intelligence**.  
Generates the extracted text and also retrieves word-level confidence scores, which can later be used for highlighting low-confidence regions.

**Logic**  
- Loads API credentials from the `.env` file.  
- Employs `DocumentIntelligenceClient` to analyze the document in a polling manner until OCR completes or times out.  
- Returns both the extracted text and a list of words with associated confidence values.

#### **`parse_ocr_to_json.py`**
**Purpose**  
Converts the raw OCR text into a structured JSON representation using **Azure OpenAI (GPT)**.  
Also includes functions for translating field names from Hebrew to English and detecting low-confidence words within the final JSON.

**Logic**  
- Maintains a Hebrew JSON template (`json_template_he`) defining the fields expected in the National Insurance (ביטוח לאומי) form.  
- Constructs a detailed prompt for GPT instructing it to parse the OCR text and fill the JSON schema accordingly, returning any missing fields as empty strings.  
- Cleans and validates GPT responses (removes triple backticks, checks for valid JSON) before returning the result.  
- Provides a mapping (`field_translation_map`) to convert from Hebrew-based keys to English-based keys for display or usage in an English interface.

#### **`evaluation.py`**
**Purpose**  
Evaluates the quality of the extracted JSON data under two scenarios:  
- **Supervised**: Compares the predicted JSON against a ground truth (golden label) for an exact field-by-field match.  
- **Unsupervised**: When ground truth is unavailable, performs rule-based checks (e.g., phone number or ID format, date plausibility) and summarizes OCR confidence metrics.

**Logic**  
- Uses a `flatten_json` function to simplify nested JSON structures for comparison.  
- **Supervised**: Calculates how many fields are correct, incorrect, missing, or falsely added, and computes an accuracy percentage.  
- **Unsupervised**: Checks for empty fields, validates certain fields (e.g., phone length, ID length, date range), and reports on overall OCR confidence (e.g., total words vs. words below a confidence threshold).  
- Saves the evaluation reports as JSON files for each processed form.

---

### **Part 2**

#### **1. Backend: `main.py`**
**Purpose**  
- Implements a **stateless microservice** using FastAPI to handle user queries about health funds (Maccabi, Meuhedet, Clalit).  
- Receives JSON payloads with user info, language, knowledge base snippet, and user question.  
- Uses **Azure OpenAI** to generate responses based on the user’s HMO and insurance tier.

**Key Logic**  
- **FastAPI App**  
  - A lightweight, stateless endpoint at `/chat` that accepts a `ChatRequest` (defined via Pydantic).  
  - A `/health` endpoint verifies the service is running.  
- **Azure OpenAI Integration**  
  - Creates a system prompt describing the rules for the chatbot (e.g., restrict answers to the user’s HMO, respond in the correct language).  
  - Merges the user question with any knowledge snippet to supply relevant context for GPT.  
- **Logging & Error Handling**  
  - Logs incoming requests, partial or absent knowledge context, and any errors from OpenAI API calls.  
  - Returns HTTP 500 if GPT fails or if there is an exception.

#### **2. Frontend: `app.py` (Streamlit)**
**Purpose**  
- Provides a client-side UI to collect user details (name, ID, age, HMO details) and then transitions to a Q&A phase, forwarding user queries to the FastAPI backend.  
- Manages all state client-side (in `st.session_state`) to keep the backend stateless.

**Key Logic**  
- **Session State**  
  - Maintains user information (first name, last name, ID, gender, HMO, etc.) and a chat history in the browser session.  
  - Guides the user through phases: greeting → collecting personal data → confirming details → Q&A.  
- **User Information Collection**  
  - Uses short prompts or function calls to validate user input (e.g., ensuring an ID is exactly 9 digits).  
  - If the user input is ambiguous, the system calls GPT to parse the correct format or to mark it invalid.  
- **Q&A Phase**  
  - Once user details are confirmed, the user can pose health-fund-related questions.  
  - The frontend performs semantic search over HTML content (from `phase2_data`) to retrieve the top relevant paragraphs, sending them as a “context” snippet to the `/chat` endpoint.  
- **Stateless Chat**  
  - The backend does not store past exchanges; the snippet plus the user question is posted each time.  
- **Multi-language Support**  
  - Detects Hebrew or English using `langdetect` during the initial greeting.  
  - Uses the discovered language to select appropriate messages (from a dictionary of prompts in both Hebrew and English).
- **Knowledge Base & Embeddings Logic**
   - Splits HTML files (from `phase2_data`) into paragraphs, then precomputes embeddings for each.  
   - At runtime, queries are embedded, and a cosine similarity search identifies the top relevant paragraphs to pass as context to GPT.
   - **`load_knowledge_base`**  
     - Iterates through all `.html` files in `phase2_data/`.  
     - Strips HTML tags, splits text into paragraph chunks, and stores them with metadata (filename, paragraph index).  
   - **`precompute_embeddings`**  
     - Calls Azure OpenAI’s embedding service on each paragraph.  
     - Stores the embeddings in a dictionary for fast retrieval.  
   - **`semantic_search_knowledge_base`**  
     - Given a user query, retrieves its embedding, compares it with each paragraph’s embedding, and selects the top **k** matches (e.g., 3 or 4).  
     - Returns a concatenated snippet to provide GPT with relevant references.

---
## 🔧 Setup & Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/GilatToker/KPMG.git
    cd KPMG
    ```

2. **Create and Activate a Virtual Environment**

    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3. **Install Dependencies**
     pip install -r requirements.txt


4. **Configure Environment Variables (`.env`)**

    At the root level (KPMG/), create a file called `.env` with the following keys (adapt as needed):
    ```env
    # Azure OpenAI
    AZURE_OPENAI_API_KEY=YOUR_OPENAI_KEY
    AZURE_OPENAI_ENDPOINT=YOUR_OPENAI_ENDPOINT
    AZURE_OPENAI_API_VERSION=YOUR_OPENAI_API_VERSION
    AZURE_OPENAI_DEPLOYMENT=YOUR_OPENAI_DEPLOYMENT_NAME

    # Azure Document Intelligence
    AZURE_DOC_INTELLIGENCE_ENDPOINT=YOUR_DOC_INTELLIGENCE_ENDPOINT
    AZURE_DOC_INTELLIGENCE_KEY=YOUR_DOC_INTELLIGENCE_KEY
    ```

---

## ▶️ How to Run

### **Part 1: OCR Form Extraction**

1. **Activate your venv** if not already:
    ```bash
    # Windows:
    .\.venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```
2. **Navigate to Part1**:
    ```bash
    cd Part1
    ```
3. **Launch Streamlit**:
    ```bash
    streamlit run app.py
    ```
4. **Open the app** in your browser: typically [http://localhost:8501](http://localhost:8501).  
   - Upload your PDF/image containing the ביטוח לאומי form.  
   - The app will display extracted fields as JSON and highlight any low-confidence text.

### **Part 2: HMO Chatbot**

You need to start both the **backend** (FastAPI) and the **frontend** (Streamlit).

1. **Backend**:
    ```bash
    cd Part2/backend
    python main.py
    ```
    - You should see a message like:  
      `INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

2. **Frontend**:
    ```bash
    cd ../frontend
    streamlit run app.py
    ```
    - Streamlit will start on [http://localhost:8501](http://localhost:8501) or another available port.  
    - Fill out your personal details, confirm, then ask health-fund related questions.
  
## Future Improvements & Additional Considerations

If given more time, here are several directions I would explore:

### **Part 1: OCR Form Extraction**
 **Detailed Unsupervised Evaluation**  
  - While the current approach displays confidence scores to the user, a future enhancement would involve developing a more comprehensive unsupervised evaluation module. This module would assess extraction quality on a per-field basis, automatically identifying fields that consistently perform poorly (e.g., the "signature" field often extracting only an "X") and quantifying overall extraction reliability.

### **Part 2: HMO Chatbot**
- **Comprehensive Language Support**  
  - Currently, some responses from the service are only in English. I would create a full mapping of all system and user messages in both Hebrew and English, ensuring every prompt and UI message honors the detected language.
- **Enhanced Retrieval**  
  - Right now, the chatbot selects top‑k paragraphs via cosine similarity. It would be interesting to test alternative ranking strategies or advanced retrieval methods to see if they improve answer accuracy.
- **Performance Monitoring**  
  - Analyze how long GPT responses typically take, track them over time, and ensure the overall solution’s latency remains acceptable for users.
- **Smart Chat History Integration**  
  Enhance the use of chat history by intelligently incorporating previous interactions and answers when they are relevant to the current query.
  
I truly enjoyed working on this project. I wish I had more time to delve deeper into every aspect, as it was important for me to deliver a strong POC Thank you for the opportunity.







