import os
import re
import uuid
import requests
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=['https://eshop.marosko.sk', 'https://www.eshop.marosko.sk'])

# ------------------ KONFIGURÁCIA ------------------
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    print("FATAL: DEEPSEEK_API_KEY is NOT set!")
else:
    print(f"DEBUG: API key loaded: {DEEPSEEK_API_KEY[:10]}...")

PRODUCT_XML_URL = "https://eshop.marosko.sk/erp/impexp/specialexport/heureka"

# 🆕 URL pre llms.txt (môžete zmeniť na https://eshop.marosko.sk/llms.txt ak chcete)
LLMS_TXT_URL = "https://marosko.sk/llms.txt"

# ------------------ NAČÍTANIE llms.txt (kontext webu) ------------------
def load_llms_context():
    """Stiahne llms.txt a vráti jeho obsah ako reťazec."""
    print("🔄 Sťahujem llms.txt...")
    try:
        resp = requests.get(LLMS_TXT_URL, timeout=10)
        resp.raise_for_status()
        print("✅ llms.txt načítaný.")
        return resp.text
    except Exception as e:
        print(f"❌ Chyba pri načítaní llms.txt: {e}")
        return ""   # ak sa nepodarí načítať, použije sa prázdny reťazec

# Globálna premenná – načíta sa pri štarte servera
llms_context = load_llms_context()

# ------------------ NAČÍTANIE PRODUKTOV Z XML (HEUREKA feed) ------------------
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
        # Vyčistenie HTML značiek z popisu
        clean_desc = re.sub(r'<[^>]+>', ' ', description)
        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
        
        products.append({
            "name": name.lower(),
            "original_name": name,
            "manufacturer": manufacturer.lower(),
            "price": price_vat,
            "url": url,
            "description": clean_desc[:1500]   # skrátenie kvôli tokenom
        })
    print(f"✅ Načítaných {len(products)} produktov.")
    return products

products = load_products_from_xml()

# ------------------ PRESNÉ VYHĽADÁVANIE PRODUKTU V OTÁZKE ------------------
def find_product(query):
    query_lower = query.lower()
    # Rozdelíme otázku na slová (odstránime krátke slová)
    words = [w for w in query_lower.split() if len(w) > 2]
    
    best_match = None
    best_score = 0
    best_match_type = "weak"  # "exact", "strong", "weak"
    
    for p in products:
        score = 0
        # 1. Skús nájsť presnú frázu (celý názov produktu v otázke)
        if p['name'] in query_lower:
            score += 100
            best_match_type = "exact"
        # 2. Hľadanie slov v názve (každé slovo)
        for word in words:
            if word in p['name']:
                score += 10
            if word in p['manufacturer']:
                score += 5
        # 3. Bonus za výrobcu, ak je spomenutý
        if any(word in p['manufacturer'] for word in words):
            score += 20
            
        if score > best_score:
            best_score = score
            best_match = p
            best_match_type = "strong" if score >= 20 else "weak"
    
    # Ak je skóre nízke (<15) a nie je to presná zhoda, nechceme produkt vrátiť
    if best_score >= 15 or best_match_type == "exact":
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
        # Ak je otázka na cenu/kúpu – odpovedz priamo
        lower_msg = user_msg.lower()
        is_price_question = any(word in lower_msg for word in ["cena", "koľko stojí", "kúp", "objednať", "link"])
        
                if is_price_question:
            # Odstránenie zátvoriek z URL
            clean_url = product['url'].rstrip(')').rstrip('(')
            return jsonify({
                "success": True,
                "response": f"**{product['original_name']}**\nCena: {product['price']} € s DPH\n\n👉 Kúpiť: {clean_url}"
            })
        
        # Odborná otázka – použijeme DeepSeek na základe popisu produktu
        prompt = f"""Si odborný poradca pre rezbárske náradie. Na základe nasledujúceho popisu produktu odpovedz na otázku používateľa. Odpovedaj v slovenčine, odborne, ale zrozumiteľne. Nepridávaj informácie, ktoré nie sú v popise.

POPIS PRODUKTU:
{product['description']}

OTÁZKA POUŽÍVATEĽA:
{user_msg}

TVOJA ODPOVEĎ (len z popisu, ak niečo nie je jasné, priznaj to):"""
        
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
            final_response = f"{ai_msg}\n\n---\n**Produkt:** {product['original_name']} – {product['price']} €\n🔗 **Kúpiť:** {product['url']}"
            return jsonify({"success": True, "response": final_response})
        except Exception as e:
            print(f"Chyba pri DeepSeek (produktová otázka): {e}")
            # Záložná odpoveď bez AI
            return jsonify({
                "success": True,
                "response": f"**{product['original_name']}**\nCena: {product['price']} € s DPH\n\n👉 Kúpiť: {product['url']}\n\n(pre podrobnosti nás kontaktujte)"
            })
    
    # 2. Žiadny relevantný produkt – použijeme všeobecný DeepSeek s kontextom z llms.txt
    # 🆕 Pripravíme systémový prompt, ktorý obsahuje llms.txt (ak je dostupný)
    if llms_context and llms_context.strip():
        system_prompt = f"""Si odborný poradca pre rezbárske náradie. Odpovedaj v slovenčine, užitočne a presne. Ak nepoznáš odpoveď, povedz to.

Tu máš informácie o e-shope Marosko (kategórie, dôležité stránky, blog, kontakty):

{llms_context}

Použi tieto informácie, ak sú relevantné k otázke používateľa. Neuvádzaj však priamo, že si čerpal z llms.txt. Odpovedaj prirodzene."""
    else:
        # Záložný prompt, ak sa llms.txt nepodarilo načítať
        system_prompt = "Si odborný poradca pre rezbárske náradie. Odpovedaj v slovenčine, užitočne a presne. Ak nepoznáš odpoveď, povedz to."

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "stream": False,
        "temperature": 0.5
    }
    try:
        resp = requests.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        ai_msg = resp.json()["choices"][0]["message"]["content"]
        return jsonify({"success": True, "response": ai_msg})
    except Exception as e:
        print(f"Chyba pri DeepSeek (všeobecná otázka): {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ------------------ HEALTH CHECK ------------------
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# ------------------ SPUSTENIE ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
