from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

class UserInfo(BaseModel):
    first_name: str
    last_name: str
    id_number: str
    gender: str
    age: int
    hmo_name: str
    hmo_card_number: str
    insurance_tier: str
    language: str

class Query(BaseModel):
    user_info: UserInfo
    question: str

@app.post("/generate_answer/")
async def generate_answer(query: Query):
    system_message = (
        "You are a helpful chatbot providing information about Israeli HMO services. "
        f"Please answer in {'English' if query.user_info.language == 'en' else 'Hebrew'}."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query.question}
    ]
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
