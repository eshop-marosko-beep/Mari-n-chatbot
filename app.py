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
LLMS_TXT_URL = "https://marosko.sk/llms.txt"

# ------------------ JAZYK ODPOVEDE ------------------
# Jazyk odpovede sa určuje z jazyka OTÁZKY (nie z locale frontendu):
# slovenčina/čeština/rumunčina → odpovie v tom istom jazyku, čokoľvek
# iné (napr. angličtina) → odpovie po slovensky.

# Štítky pre kartu s cenou/odkazom, ktorá sa pripája mechanicky za AI
# odpoveď (a pre núdzovú odpoveď pri zlyhaní DeepSeek) — tieto texty
# AI nevidí, takže sa musia prekladať samostatne.
LABELS = {
    "sk": {"product": "Produkt", "buy": "Kúpiť", "price": "Cena", "vat_suffix": "s DPH", "contact": "pre podrobnosti nás kontaktujte"},
    "cz": {"product": "Produkt", "buy": "Koupit", "price": "Cena", "vat_suffix": "s DPH", "contact": "pro podrobnosti nás kontaktujte"},
    "ro": {"product": "Produs", "buy": "Cumpără", "price": "Preț", "vat_suffix": "cu TVA", "contact": "pentru detalii ne puteți contacta"},
}

# AI dostane pokyn uviesť na prvom riadku svojej odpovede značku v tomto
# tvare (napr. "LANG:sk"), aby backend vedel, ktoré štítky použiť pre
# kartu s cenou/odkazom — túto značku parsujeme a z výslednej odpovede
# odstránime.
LANG_TAG_RE = re.compile(r'^[\s*_]*LANG:\s*(sk|cz|ro)[\s*_]*\n+', re.IGNORECASE)

LANGUAGE_INSTRUCTION = """DÔLEŽITÉ - JAZYK ODPOVEDE: Zisti, v akom jazyku je napísaná otázka zákazníka.
- Ak je v slovenčine, češtine alebo rumunčine, odpovedz v tom istom jazyku.
- Ak je v akomkoľvek inom jazyku (napríklad v angličtine), odpovedz po slovensky.
Prvý riadok svojej odpovede napíš presne v tvare "LANG:sk", "LANG:cz" alebo "LANG:ro" (podľa jazyka, v ktorom odpovedáš), za ním prázdny riadok a až potom samotnú odpoveď."""

def detect_locale_heuristic(text):
    """Núdzový odhad jazyka podľa charakteristických diakritických znakov —
    použije sa len keď AI odpoveď nie je k dispozícii (zlyhanie DeepSeek)."""
    t = (text or "").lower()
    if any(ch in t for ch in "ăâîșşțţ"):
        return "ro"
    if any(ch in t for ch in "ěřů"):
        return "cz"
    return "sk"

def extract_language(ai_text, fallback_source):
    """Vyparsuje značku LANG:xx z odpovede AI a vráti (locale, odpoveď bez značky).
    Ak AI značku nedodrží, jazyk sa odhadne z pôvodnej otázky."""
    match = LANG_TAG_RE.match(ai_text)
    if match:
        return match.group(1).lower(), ai_text[match.end():]
    return detect_locale_heuristic(fallback_source), ai_text

# ------------------ POMOCNÁ FUNKCIA NA ČISTENIE URL ------------------
def clean_url(url):
    """Odstráni zátvorky z URL."""
    if not url:
        return url
    # Odstrániť zátvorky z konca a začiatku
    url = url.rstrip(')').rstrip('(').lstrip('(').lstrip(')')
    # Odstrániť markdown syntax [text](url) - extrahuje iba URL
    match = re.search(r'\]\((https?://[^)\s]+)\)', url)
    if match:
        url = match.group(1)
    return url

def clean_ai_response(response_text):
    """Vyčistí AI odpoveď od zátvoriek v URL."""
    # Nájdi všetky URL v odpovedi
    url_pattern = r'https?://[^\s)]+'
    urls = re.findall(url_pattern, response_text)
    
    for url in urls:
        clean = clean_url(url)
        if clean != url:
            response_text = response_text.replace(url, clean)
    
    # Odstrániť markdown odkazy formátu [text](url) - premeniť na čistý text s URL
    response_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', response_text)
    
    return response_text

# ------------------ NAČÍTANIE llms.txt ------------------
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
        return ""

llms_context = load_llms_context()

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
        # Vyčistenie HTML značiek z popisu
        clean_desc = re.sub(r'<[^>]+>', ' ', description)
        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
        
        products.append({
            "name": name.lower(),
            "original_name": name,
            "manufacturer": manufacturer.lower(),
            "price": price_vat,
            "url": clean_url(url),  # Vyčisti URL už pri načítaní
            "description": clean_desc[:1500]
        })
    print(f"✅ Načítaných {len(products)} produktov.")
    return products

products = load_products_from_xml()

# ------------------ VYHĽADÁVANIE PRODUKTU ------------------
def find_product(query):
    query_lower = query.lower()
    words = [w for w in query_lower.split() if len(w) > 2]
    
    best_match = None
    best_score = 0
    best_match_type = "weak"
    
    for p in products:
        score = 0
        if p['name'] in query_lower:
            score += 100
            best_match_type = "exact"
        for word in words:
            if word in p['name']:
                score += 10
            if word in p['manufacturer']:
                score += 5
        if any(word in p['manufacturer'] for word in words):
            score += 20
            
        if score > best_score:
            best_score = score
            best_match = p
            best_match_type = "strong" if score >= 20 else "weak"
    
    if best_score >= 15 or best_match_type == "exact":
        return best_match
    return None

# ------------------ ENDPOINT /chat ------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")

    product = find_product(user_msg)

    if product:
        # Cena, výrobca aj odkaz idú do promptu vždy spolu s popisom — predtým sa
        # otázky na cenu rozpoznávali len podľa presného výskytu fráz ako
        # "koľko stojí" (bez diakritiky/preklepu sa to netrafilo), a keď sa
        # netrafilo, AI dostalo do kontextu len POPIS produktu bez ceny a
        # odpovedalo, že cenu nepozná — hoci bola k dispozícii a pripojila sa
        # v samostatnej karte pod odpoveďou. Jeden spoločný prompt so všetkými
        # údajmi to rieši bez ohľadu na presné znenie otázky.
        clean_url_link = clean_url(product['url'])
        prompt = f"""Si odborný a priateľský poradca pre rezbárske náradie v e-shope Marosko. Zákazník sa pýta na konkrétny produkt nižšie.

{LANGUAGE_INSTRUCTION}

Odpovedz prirodzene, vecne a v plných vetách, nie len strohým výpisom údajov. Použi cenu, výrobcu aj popis, ak sú pre otázku relevantné. Ak sa niečo v údajoch nenachádza, úprimne to priznaj namiesto vymýšľania. Keď zobrazuješ odkazy, používaj čisté URL bez zátvoriek. Údaje o produkte nižšie sú v slovenčine — ak odpovedáš v inom jazyku, preformuluj ich.

PRODUKT: {product['original_name']}
VÝROBCA: {product['manufacturer']}
CENA: {product['price']} € s DPH
ODKAZ NA KÚPU: {clean_url_link}
POPIS: {product['description']}

OTÁZKA ZÁKAZNÍKA:
{user_msg}

TVOJA ODPOVEĎ (začni riadkom LANG:xx):"""

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.4
        }
        try:
            resp = requests.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            ai_msg_raw = resp.json()["choices"][0]["message"]["content"]
            locale, ai_msg = extract_language(ai_msg_raw, user_msg)
            labels = LABELS[locale]
            # Vyčisti AI odpoveď od zátvoriek v URL
            ai_msg = clean_ai_response(ai_msg)
            final_response = f"{ai_msg}\n\n---\n**{labels['product']}:** {product['original_name']} – {product['price']} €\n🔗 **{labels['buy']}:** {clean_url_link}"
            return jsonify({"success": True, "response": final_response})
        except Exception as e:
            print(f"Chyba pri DeepSeek (produktová otázka): {e}")
            labels = LABELS[detect_locale_heuristic(user_msg)]
            return jsonify({
                "success": True,
                "response": f"**{product['original_name']}**\n{labels['price']}: {product['price']} € {labels['vat_suffix']}\n\n👉 {labels['buy']}: {clean_url_link}\n\n({labels['contact']})"
            })
    
    # Všeobecná otázka
    if llms_context and llms_context.strip():
        system_prompt = f"""Si odborný poradca pre rezbárske náradie. {LANGUAGE_INSTRUCTION} Buď užitočný a presný. Ak nepoznáš odpoveď, povedz to. Keď zobrazuješ odkazy, používaj čisté URL bez zátvoriek.

Tu máš informácie o e-shope Marosko (kategórie, dôležité stránky, blog, kontakty):

{llms_context}

Použi tieto informácie, ak sú relevantné k otázke používateľa. Neuvádzaj však priamo, že si čerpal z llms.txt. Odpovedaj prirodzene."""
    else:
        system_prompt = f"Si odborný poradca pre rezbárske náradie. {LANGUAGE_INSTRUCTION} Buď užitočný a presný. Ak nepoznáš odpoveď, povedz to. Keď zobrazuješ odkazy, používaj čisté URL bez zátvoriek."

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
        ai_msg_raw = resp.json()["choices"][0]["message"]["content"]
        _, ai_msg = extract_language(ai_msg_raw, user_msg)
        # Vyčisti AI odpoveď od zátvoriek v URL
        ai_msg = clean_ai_response(ai_msg)
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
