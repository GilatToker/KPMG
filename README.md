# KPMG AI Project

This repository contains **two main parts**:

1. **Part 1: OCR Form Extraction**  
   Extracts structured data from National Insurance forms (ביטוח לאומי) using Azure Document Intelligence and Azure OpenAI.

2. **Part 2: HMO Chatbot**  
   Answers medical questions related to Israeli health funds (מכבי, מאוחדת, כללית) based on user info, using FastAPI (for backend) and Streamlit (for frontend).

---

## 🗂 Project Structure
KPMG/ ├── Part1/ │ ├── app.py # Streamlit frontend for OCR extraction │ ├── ocr_extraction.py # Azure Document Intelligence integration │ ├── parse_ocr_to_json.py # GPT-based field extraction logic │ └── requirements.txt # Dependencies for Part1 ├── Part2/ │ ├── backend/ │ │ ├── main.py # FastAPI server │ │ └── requirements.txt # Dependencies for backend │ └── frontend/ │ ├── app.py # Streamlit frontend for HMO Chatbot │ └── requirements.txt # Dependencies for frontend ├── phase2_data/ # HTML knowledge base for Chatbot ├── .env # Environment variables (not tracked in Git) └── README.md # This file

## 🔧 Setup & Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/<YourUserName>/KPMG.git
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

   - **Part 1** (OCR):
     ```bash
     cd Part1
     pip install -r requirements.txt
     cd ..
     ```
   - **Part 2** (Backend & Frontend):
     ```bash
     cd Part2/backend
     pip install -r requirements.txt
     cd ../frontend
     pip install -r requirements.txt
     cd ../../
     ```

    Make sure you remain in your virtual environment whenever you install or run the code.

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

---

## ℹ️ Additional Notes

- **.env**: Never commit `.env` to GitHub (it contains secrets).
- **requirements.txt**: Each subfolder has its own dependencies. If you prefer a single environment for all, just install them all together in one environment.
- **phase2_data**: Contains the HTML knowledge base for the HMO Chatbot’s semantic search.

---

## 🙌 Contributing

Feel free to open issues or pull requests. For major changes, please open an issue first to discuss your idea.

---

## 🏆 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details (if included).

---

**Enjoy building and exploring the HMO Chatbot & OCR Form Extraction!**


