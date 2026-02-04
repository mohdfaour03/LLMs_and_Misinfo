
import os
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic
from openai import OpenAI

load_dotenv()

def log(msg):
    print(msg)
    with open("connectivity_results.txt", "a") as f:
        f.write(msg + "\n")

# Clear previous results
with open("connectivity_results.txt", "w") as f:
    f.write("--- DIAGNOSTIC TEST ---\n")

# 1. Test Gemini (User Request)
log("\nTesting GEMINI (gemini-flash-latest)...")
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-flash-latest')
    response = model.generate_content("Say 'Gemini OK'")
    log(f"SUCCESS: {response.text.strip()}")
except Exception as e:
    log(f"FAILED: {e}")

# 2. Test Claude (User Request)
log("\nTesting CLAUDE (User: claude-haiku-4-5)...")
try:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say 'Claude Output'"}]
    )
    log(f"SUCCESS (User String): {message.content[0].text}")
except Exception as e:
    log(f"FAILED (User String): {e}")

# 2b. Test Claude (Standard Hypothesis)
log("\nTesting CLAUDE (Standard: claude-3-5-haiku-20241022)...")
try:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say 'Claude Output'"}]
    )
    log(f"SUCCESS (Standard String): {message.content[0].text}")
except Exception as e:
    log(f"FAILED (Standard String): {e}")

# 3. Test OpenAI
log("\nTesting OPENAI (Control)...")
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say 'GPT OK'"}]
    )
    log(f"SUCCESS: {response.choices[0].message.content}")
except Exception as e:
    log(f"FAILED: {e}")
