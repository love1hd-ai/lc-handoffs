import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv()


client_local = Client(
    host=os.environ.get("OLLAMA_API_BASE_LOCAL"),
    headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY_LOCAL")},
)


client_remote = Client(
    host=os.environ.get("OLLAMA_API_BASE_REMOTE"),
    headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY_REMOTE")},
)
