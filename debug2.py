import openai
import config  # This is our config.py

# 1) Configure openai for Azure
openai.api_type = "azure"
openai.api_base = config.AZURE_OPENAI_ENDPOINT  # e.g. https://oai-lab-test-eastus-001.openai.azure.com/
openai.api_version = config.AZURE_OPENAI_API_VERSION
openai.api_key = config.AZURE_OPENAI_API_KEY

# 2) Make a request to the ChatCompletion endpoint
response = openai.ChatCompletion.create(
    engine=config.AZURE_OPENAI_DEPLOYMENT,  # e.g. "gpt-4o"
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello GPT-4O! How are you today?"}
    ]
)

# 3) Print the response
print("GPT-4O says:")
print(response.choices[0].message.content)
