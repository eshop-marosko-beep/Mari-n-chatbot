import os
import re
import uuid
import requests
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=['https://eshop.marosko.sk', 'https://www.eshop.marosko.sk'])

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    print("FATAL: DEEPSEEK_API_KEY is NOT set!")
else:
    print(f"DEBUG: API key loaded: {DEEPSEEK_API_KEY[:10]}...")

PRODUCT_XML_URL = "https://eshop.marosko.sk/erp/impexp/specialexport/heureka"

# ------------------ NAČÍTANIE PRODUKTOV Z XML ------------------
def load_products_from_xml():
    print("🔄 Sťahujem XML feed produktov...")
    try:
        resp = requests.get(PRODUCT_XML_URL, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"❌ Chyba pri sťahovaní XML: {e}")
        return []

    products = []
    for item in root.findall(".//SHOPITEM"):
        name = item.findtext("PRODUCTNAME", "").lower()
        manufacturer = item.findtext("MANUFACTURER", "").lower()
        price_vat = item.findtext("PRICE_VAT", "")
        url = item.findtext("URL", "")
        products.append({
            "name": name,
            "manufacturer": manufacturer,
            "price": price_vat,
            "url": url
        })
    print(f"✅ Načítaných {len(products)} produktov.")
    return products

products = load_products_from_xml()

# ------------------ JEDNODUCHÉ VYHĽADÁVANIE PRODUKTOV ------------------
def find_product(query):
    query_lower = query.lower()
    best_match = None
    best_score = 0
    
    for p in products:
        score = 0
        # Hľadá slová z otázky v názve alebo výrobcovi
        words = query_lower.split()
        for word in words:
            if len(word) < 3:
                continue
            if word in p['name']:
                score += 3
            if word in p['manufacturer']:
                score += 2
        if score > best_score:
            best_score = score
            best_match = p
    
    if best_score >= 2:
        return best_match
    return None

# ------------------ ENDPOINT /chat ------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    
    # 1. Skús nájsť produkt
    product = find_product(user_msg)
    if product:
        return jsonify({
            "success": True,
            "response": f"**{product['name'].title()}**\nCena: {product['price']} € s DPH\n\n👉 Kúpiť: {product['url']}"
        })
    
    # 2. Ak nenašiel, použijeme DeepSeek API
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Si odborný poradca pre rezbárske náradie. Odpovedaj stručne, užitočne, v slovenčine."},
            {"role": "user", "content": user_msg}
        ],
        "stream": False
    }
    try:
        resp = requests.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        ai_msg = resp.json()["choices"][0]["message"]["content"]
        return jsonify({"success": True, "response": ai_msg})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
