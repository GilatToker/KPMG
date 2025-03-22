# KPMG AI Project

This repository contains **two main parts**:

1. **Part 1: OCR Form Extraction**  
   Extracts structured data from National Insurance forms (ביטוח לאומי) using Azure Document Intelligence and Azure OpenAI.

2. **Part 2: HMO Chatbot**  
   Answers medical questions related to Israeli health funds (מכבי, מאוחדת, כללית) based on user info, using FastAPI (for backend) and Streamlit (for frontend).


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

**Enjoy building and exploring the HMO Chatbot & OCR Form Extraction!**


