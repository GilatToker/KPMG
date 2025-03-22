import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

# Azure Document Intelligence credentials
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT or not AZURE_OPENAI_API_VERSION:
    raise ValueError("Missing API Key or Endpoint in .env file!")

# Verify that the keys are loaded (optional, but useful for debugging)
if __name__ == "__main__":
    print("Config Loaded Successfully!")
    print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    print(f"Azure Document Intelligence Endpoint: {AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT}")
    print(f"GPT-4 Deployment: {AZURE_OPENAI_DEPLOYMENT}")
