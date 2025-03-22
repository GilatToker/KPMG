# backend/main.py
import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
import config  # This file holds configuration constants such as the Azure OpenAI deployment name

# Load environment variables from a .env file
load_dotenv()
# Configure logging to include the timestamp, level, and message for debugging purposes
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Create a FastAPI app instance. This is our stateless microservice.
app = FastAPI()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
# Define the data model for the chat request payload using Pydantic.
class ChatRequest(BaseModel):
    user_info: dict
    question: str
    language: str  # 'he' or 'en'
    context: str = ""  # knowledge base snippet
    conversation_history: list = []

# Health-check endpoint to verify that the service is running.
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Chat endpoint to process user queries.
@app.post("/chat")
async def chat(payload: ChatRequest):
    logging.info(f"Received chat request: {payload.question[:500]}...")
    logging.debug(f"Received knowledge snippet:\n{payload.context}")
    logging.debug(f"conversation_history: {payload.conversation_history}")

    # Warn if no context (knowledge snippet) was provided
    if not payload.context or payload.context.strip() == "":
        logging.warning("Warning: No knowledge snippet received!")

    # Build prompt
    system_prompt = (
        "You are a helpful chatbot that answers questions about Israeli health funds (Maccabi, Meuhedet, and Clalit). "
        "Your response should be based on the provided knowledge base, which includes general information shared across all health funds, "
        "followed by a table with specific details for each health fund and their respective insurance tiers. "
        "The file will also include contact numbers for each health fund. "
        f"Always provide an answer based solely on the user's selected health fund ({payload.user_info.get('hmo_name', '')}), "
        f"and their insurance tier ({payload.user_info.get('insurance_tier', '')}). "
        f"Answer in {'English' if payload.language == 'en' else 'Hebrew'}."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Note: The conversation history is not added to the prompt to ensure statelessness
    # and that each answer is based solely on the current query and provided context.

    logging.debug(f"Final messages sent to GPT:\n{messages}")

    # Add the context + last user question as a "user" role message
    # (this ensures the model sees the snippet + user question up front)
    context_message = (
        f"User Info: {payload.user_info}\n"
        f"Relevant Info:\n{payload.context}\n\n"
        f"User Question: {payload.question}"
    )
    messages.append({"role": "user", "content": context_message})

    try:
        # Send the messages to the Azure OpenAI chat completions API.
        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        # Extract the answer text from the response.
        answer_text = response.choices[0].message.content.strip()
        logging.info("Generated response")
        return {"answer": answer_text}

    except Exception as e:
        # Log any errors and return an HTTP 500 error to the client.
        logging.error(f"Error calling OpenAI: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

# Run the application using Uvicorn if this file is executed directly.
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
