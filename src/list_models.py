from groq import Groq
from utils import get_secret

client = Groq(api_key=get_secret("GROQ_API_KEY"))
for m in client.models.list().data:
    print(m.id)