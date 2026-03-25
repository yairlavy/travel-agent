import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load your GOOGLE_API_KEY from .env
load_dotenv()

# 2. Use the LangChain class 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # langchain-google-genai will automatically find GOOGLE_API_KEY in your env
)

# 3. Test the connection
try:
    print("--- Testing Connection ---")
    # LangChain uses .invoke(), not .generate_content()
    response = llm.invoke("Say 'System Online' if you can hear me.")
    print(f"Success! Response: {response.content}")
except Exception as e:
    print(f"--- Connection Failed ---")
    print(f"Error details: {e}")
