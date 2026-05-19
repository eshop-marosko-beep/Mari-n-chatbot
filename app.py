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
        name = item.findtext("PRODUCTNAME", "")
        manufacturer = item.findtext("MANUFACTURER", "")
        price_vat = item.findtext("PRICE_VAT", "")
        url = item.findtext("URL", "")
        description = item.findtext("DESCRIPTION", "")
        # Odstránime HTML tagy z popisu
        clean_desc = re.sub(r'<[^>]+>', ' ', description)
        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
        
        products.append({
            "name": name.lower(),
            "manufacturer": manufacturer.lower(),
            "price": price_vat,
            "url": url,
            "description": clean_desc[:1500]  # prvých 1500 znakov
        })
    print(f"✅ Načítaných {len(products)} produktov.")
    return products

products = load_products_from_xml()

# ------------------ VYHĽADÁVANIE PRODUKTOV ------------------
def find_product(query):
    query_lower = query.lower()
    best_match = None
    best_score = 0
    
    for p in products:
        score = 0
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
        # Ak ide o otázku na kúpu/cenu, odpovedz priamo
        lower_msg = user_msg.lower()
        if any(word in lower_msg for word in ["kúp", "cena", "koľko stojí", "objednať", "link"]):
            return jsonify({
                "success": True,
                "response": f"**{product['name'].title()}**\nCena: {product['price']} € s DPH\n\n👉 Kúpiť: {product['url']}"
            })
        
        # Inak – odborná otázka – použijeme DeepSeek s popisom produktu
        prompt = f"""Si odborný poradca pre rezbárske náradie. Na základe tohto popisu produktu odpovedz na otázku používateľa. Odpovedaj v slovenčine, stručne, odborne a užitočne.

POPIS PRODUKTU:
{product['description']}

OTÁZKA POUŽÍVATEĽA:
{user_msg}

TVOJA ODPOVEĎ (len na základe popisu, nepridávaj vlastné vedomosti):"""
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.3
        }
        try:
            resp = requests.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            ai_msg = resp.json()["choices"][0]["message"]["content"]
            # Na koniec pridáme odkaz na produkt
            final_response = f"{ai_msg}\n\n👉 **Produkt:** {product['name'].title()} – {product['price']} €\n🔗 **Kúpiť:** {product['url']}"
            return jsonify({"success": True, "response": final_response})
        except Exception as e:
            print(f"Chyba pri DeepSeek: {e}")
            # Fallback – aspoň základné info
            return jsonify({
                "success": True,
                "response": f"**{product['name'].title()}**\nCena: {product['price']} € s DPH\n\n👉 Kúpiť: {product['url']}\n\n(Podrobné info momentálne nedostupné, skúste neskôr.)"
            })
    
    # 2. Ak nenašiel produkt, použijeme všeobecný DeepSeek
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
