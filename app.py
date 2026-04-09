import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Povolenie CORS pre tvoju doménu
CORS(app, origins=['https://eshop.marosko.sk', 'https://www.eshop.marosko.sk'])

# Načítanie API kľúča z prostredia Renderu
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# --- DEBUG: Vypíšeme do logov, či sa kľúč načítal ---
if DEEPSEEK_API_KEY:
    print(f"✅ DEEPSEEK_API_KEY loaded successfully. Starts with: {DEEPSEEK_API_KEY[:10]}...")
else:
    print("❌ FATAL: DEEPSEEK_API_KEY is NOT set in environment variables!")
# -------------------------------------------------

SYSTEM_PROMPT = """Si odborný poradca pre rezbárske náradie na e-shope marosko.sk.
Pomáhaš zákazníkom s výberom správneho náradia.
Tvoja odpoveď má byť stručná, užitočná a v slovenčine."""

conversations = {}

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "DeepSeek chatbot API beží. Použite /chat endpoint."})

@app.route('/chat', methods=['POST'])
def chat():
    # Ak nie je kľúč, vrátime zrozumiteľnú chybu
    if not DEEPSEEK_API_KEY:
        return jsonify({"success": False, "error": "Server not configured: API key missing. Please set DEEPSEEK_API_KEY in Render environment."}), 500

    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    conversations[session_id].append({"role": "user", "content": user_message})
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversations[session_id])
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]
        
        conversations[session_id].append({"role": "assistant", "content": ai_response})
        
        if len(conversations[session_id]) > 10:
            conversations[session_id] = conversations[session_id][-10:]
        
        return jsonify({"success": True, "response": ai_response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "api_key_configured": bool(DEEPSEEK_API_KEY)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
