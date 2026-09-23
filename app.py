import os
import re
import time
import uuid
import requests
import xml.etree.ElementTree as ET
from collections import OrderedDict
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
LLMS_TXT_URL = "https://www.marosko.sk/llms.txt"  # canonical host; the bare domain 308-redirects here

# ------------------ KONVERZAČNÁ PAMÄŤ ------------------
# Predtým bola každá požiadavka úplne bez pamäti — session_id prišiel od
# frontendu, ale backend ho nikde nepoužil, takže bot si nič nepamätal medzi
# správami v tej istej konverzácii (napr. opravu od zákazníka o dva riadky
# vyššie). Uchovávame posledných pár výmen za session_id a vkladáme ich do
# promptu ako históriu, aby bot naozaj "pokračoval v konverzácii".
conversations = OrderedDict()  # session_id -> [{"role": "user"|"assistant", "content": str}, ...]
MAX_HISTORY_MESSAGES = 12  # posledných 6 výmen (user+assistant)
MAX_SESSIONS = 500  # ochrana proti neobmedzenému rastu pamäte na dlho bežiacom procese

def get_history(session_id):
    return conversations.get(session_id, [])

def append_to_history(session_id, user_msg, assistant_msg):
    history = conversations.setdefault(session_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    del history[:-MAX_HISTORY_MESSAGES]
    conversations.move_to_end(session_id)
    while len(conversations) > MAX_SESSIONS:
        conversations.popitem(last=False)

# ------------------ JAZYK ODPOVEDE ------------------
# Jazyk odpovede sa určuje z jazyka OTÁZKY (nie z locale frontendu):
# slovenčina/čeština/rumunčina → odpovie v tom istom jazyku, čokoľvek
# iné (napr. angličtina) → odpovie po slovensky.

# Štítky pre kartu s cenou/odkazom, ktorá sa pripája mechanicky za AI
# odpoveď (a pre núdzovú odpoveď pri zlyhaní DeepSeek) — tieto texty
# AI nevidí, takže sa musia prekladať samostatne.
LABELS = {
    "sk": {"product": "Produkt", "buy": "Kúpiť", "price": "Cena", "vat_suffix": "s DPH", "contact": "pre podrobnosti nás kontaktujte", "recommended": "Odporúčané produkty z ponuky"},
    "cz": {"product": "Produkt", "buy": "Koupit", "price": "Cena", "vat_suffix": "s DPH", "contact": "pro podrobnosti nás kontaktujte", "recommended": "Doporučené produkty z nabídky"},
    "ro": {"product": "Produs", "buy": "Cumpără", "price": "Preț", "vat_suffix": "cu TVA", "contact": "pentru detalii ne puteți contacta", "recommended": "Produse recomandate din ofertă"},
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
llms_context_loaded_at = time.time()
LLMS_REFRESH_SECONDS = 6 * 3600  # matches marosko-web's own Heureka-review cache window

def get_llms_context():
    """Returns llms_context, refreshing it once the TTL above expires so a
    long-running process (no restart for days) doesn't serve a stale site
    map forever. Keeps the last-known-good text if a refresh attempt fails."""
    global llms_context, llms_context_loaded_at
    if time.time() - llms_context_loaded_at > LLMS_REFRESH_SECONDS:
        fresh = load_llms_context()
        llms_context_loaded_at = time.time()
        if fresh and fresh.strip():
            llms_context = fresh
    return llms_context

# ------------------ DOPRAVA A PLATBA ------------------
# Táto informácia nie je ani v produktovom XML feede, ani v llms.txt — bot ju
# doteraz nemal k dispozícii nikde, a tak na otázku "koľko stojí doprava"
# odpovedal, že to nevie a treba nás kontaktovať, hoci ide o bežnú a vopred
# zverejnenú informáciu na e-shope. Stránka je celá plná menu/kategórií;
# zaujímavý text vyrežeme medzi dvoma stabilnými kotvami, ktoré sa na
# stránke opakujú okolo skutočného obsahu.
SHIPPING_PAYMENT_URL = "https://eshop.marosko.sk/mapa-nakup-rezbarskeho-naradia-online-obchod"
_SHIPPING_PAYMENT_START_RE = re.compile(r'ceny:\s*všetky zobrazené ceny', re.IGNORECASE)
_SHIPPING_PAYMENT_END_RE = re.compile(r'zvoľte kategóriu', re.IGNORECASE)

def load_shipping_payment_info():
    """Stiahne stránku Doprava a platba a vyrieže z nej len samotný text o
    cenách dopravy a spôsoboch platby (bez okolitého menu/kategórií)."""
    print("🔄 Sťahujem informácie o doprave a platbe...")
    try:
        resp = requests.get(SHIPPING_PAYMENT_URL, timeout=10)
        resp.raise_for_status()
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', resp.text, flags=re.S)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        start = _SHIPPING_PAYMENT_START_RE.search(text)
        end = _SHIPPING_PAYMENT_END_RE.search(text, start.end() if start else 0)
        if not start or not end:
            print("❌ Informácie o doprave a platbe: kotvy sa na stránke nenašli (zmenila sa štruktúra?).")
            return ""
        print("✅ Informácie o doprave a platbe načítané.")
        return text[start.start():end.start()].strip()
    except Exception as e:
        print(f"❌ Chyba pri načítaní informácií o doprave a platbe: {e}")
        return ""

shipping_payment_info = load_shipping_payment_info()
shipping_payment_info_loaded_at = time.time()
SHIPPING_PAYMENT_REFRESH_SECONDS = 24 * 3600  # ceny dopravy/platby sa menia zriedka

def get_shipping_payment_info():
    """Rovnaký TTL-refresh + last-known-good vzor ako get_llms_context()."""
    global shipping_payment_info, shipping_payment_info_loaded_at
    if time.time() - shipping_payment_info_loaded_at > SHIPPING_PAYMENT_REFRESH_SECONDS:
        fresh = load_shipping_payment_info()
        shipping_payment_info_loaded_at = time.time()
        if fresh and fresh.strip():
            shipping_payment_info = fresh
    return shipping_payment_info

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
        
        name_lower = name.lower()
        name_words = name_lower.split()
        products.append({
            "name": name_lower,
            # Susedné dvojice slov spojené bez medzery — zákazníci bežne
            # píšu viacslovné názvy ako jedno slovo (napr. "Orbicut" pre
            # feedom uvádzané "Orbi Cut"). Zámerne PRESNÁ zhoda len na
            # SUSEDNÝCH slovách (nie spojenie celého názvu do jedného
            # reťazca) — spájanie celého názvu vedelo omylom vytvoriť cudzie
            # slovo na hranici dvoch nesúvisiacich slov (napr. "hrana" +
            # "pr.9,5" → obsahuje aj "napr", bežné slovenské slovo "napr.").
            "name_bigrams": {name_words[i] + name_words[i + 1] for i in range(len(name_words) - 1)},
            "original_name": name,
            "manufacturer": manufacturer.lower(),
            "price": price_vat,
            "url": clean_url(url),  # Vyčisti URL už pri načítaní
            "description": clean_desc[:1500]
        })
    print(f"✅ Načítaných {len(products)} produktov.")
    return products

products = load_products_from_xml()
products_loaded_at = time.time()
PRODUCTS_REFRESH_SECONDS = 3600  # prices/stock change more often than the site map

def get_products():
    """Returns the product list, refreshing it once the TTL above expires —
    same stale-cache problem and fix as get_llms_context() above."""
    global products, products_loaded_at
    if time.time() - products_loaded_at > PRODUCTS_REFRESH_SECONDS:
        fresh = load_products_from_xml()
        products_loaded_at = time.time()
        if fresh:
            products = fresh
    return products

# ------------------ VYHĽADÁVANIE PRODUKTU ------------------
# Bežné všeobecné podstatné mená z produktového/obchodného slovníka, ktoré
# vedia byť dosť dlhé (6+ znakov) na to, aby ich find_product() nižšie
# považoval za "dosť špecifické slovo samo o sebe" — v skutočnosti sa ale
# vyskytujú v mnohých nesúvisiacich produktoch alebo bežnej konverzácii
# (napr. "aké priemery máte?" nesmie vrátiť náhodný produkt len preto, že
# slovo "priemery" je dlhé). Zoznam je zámerne len najčastejšie/najrizikovejšie
# prípady, nie kompletný slovník — chýbajúce slovo len vráti pôvodné
# (bezpečné) správanie, nikdy nespôsobí nový falošný zásah.
GENERIC_PRODUCT_WORDS = {
    "priemer", "priemery", "priemerov", "priemeru",
    "skrutka", "skrutky", "skrutiek",
    "rozmer", "rozmery", "rozmerov",
    "material", "materialu", "materiálu",
    "drevo", "dreva", "drevorezba", "drevorezbu", "drevorezby",
    "farba", "farby", "farieb",
    "naradie", "naradia", "nastroj", "nastroje", "nastroja",
    "sada", "sady", "sadu",
    "vyrobok", "vyrobky", "tovar", "tovaru", "tovary",
    # Bežné názvy celých kategórií náradia — príliš všeobecné na to, aby
    # samy osebe identifikovali KONKRÉTNY produkt (desiatky/stovky
    # produktov obsahujú tieto slová vo svojom názve).
    "brúska", "brúsky", "brusky", "bruska",
    "frézka", "frézky", "frezka", "frezky",
    "kotúč", "kotúče", "kotuc", "kotuce",
    "nástavec", "nástavce", "nastavec", "nastavce",
    "uhlová", "uhlovej", "uhlova", "uhlovou",
    "dláto", "dláta", "dlato", "dlata",
    "rašpľa", "rašple", "raspla", "rasple",
}


def find_product(query):
    """Nájde produkt, o ktorom sa zákazník pravdepodobne pýta.

    Zhoda ostáva na podreťazcoch (nie na celých slovách) zámerne — pri
    bohatej slovenskej/českej skloňovanej flektológii to funguje ako
    chudobná náhrada stemmingu (napr. "trojuholníkový" z otázky sa trafí
    do "trojuholníkovým" v názve produktu). Porovnáva sa navyše proti
    dvojiciam SUSEDNÝCH slov spojených bez medzery ("orbicut" zákazníka
    nájde feedom uvádzané "Orbi Cut" — inak by ich rozdelila medzera
    uprostred a bot by tvrdil, že cenu nepozná, hoci produkt v ponuke
    reálne má); zámerne len susedné dvojice, nie spojenie celého názvu do
    jedného reťazca, aby sa slovo neomylom netrafilo na hranici dvoch
    nesúvisiacich slov ďaleko od seba. Predtým vedela SAMOTNÁ zhoda vo
    výrobcovi (`manufacturer`) sama o sebe pretiahnuť prah a vrátiť celkom
    nesúvisiaci produkt, keď sa niektoré slovo z otázky náhodou vyskytlo
    ako podreťazec v poli výrobcu niektorého z ~1250 produktov — bez
    akejkoľvek súvislosti s tým, o čom sa reálne rozprávalo. Výrobca teda
    odteraz môže len PRIDAŤ body k produktu, ktorý už má aspoň jednu zhodu
    vo vlastnom názve, nikdy nie sám o sebe rozhodnúť."""
    query_lower = query.lower()
    # >3 (nie >2): trojpísmenové slová sú v slovenčine skoro vždy predložky
    # alebo spojky ("pre", "bez", "ako", "ale"...), nikdy nič, čo by
    # identifikovalo konkrétny produkt — a keď sa takéto slovo zhodou
    # okolností vyskytlo aj v názve nejakého produktu, dokázalo spolu s
    # jedným ďalším slabým zásahom pretiahnuť prah bez akejkoľvek reálnej
    # súvislosti s otázkou.
    words = [w for w in query_lower.split() if len(w) > 3]

    best_match = None
    best_score = 0

    for p in get_products():
        name_score = 0
        if p['name'] in query_lower:
            name_score += 100
        matched_words = [w for w in words if w in p['name'] or w in p['name_bigrams']]
        if not matched_words:
            continue
        name_score += 10 * len(matched_words)
        # Jedno dlhé/špecifické slovo (napr. "orbicut") je samo o sebe
        # dostatočný dôkaz — inak by pri jedinom zhodnom slove nikdy nedosiahlo
        # prah nižšie a bot by tvrdil, že produkt/cenu nepozná, hoci ho má.
        # Bežné všeobecné slová (pozri GENERIC_PRODUCT_WORDS) sú z tohto
        # bonusu vyňaté, aj keď sú rovnako dlhé.
        if any(len(w) >= 6 and w not in GENERIC_PRODUCT_WORDS for w in matched_words):
            name_score += 10

        score = name_score
        for word in words:
            if word in p['manufacturer']:
                score += 5

        if score > best_score:
            best_score = score
            best_match = p

    if best_score >= 15:
        return best_match
    return None


def find_products_mentioned(text, limit=3):
    """Keď bot pri odpovedi na VŠEOBECNÚ otázku (nie priamu otázku na jeden
    produkt) sám odporučí konkrétne nástroje/značky — napr. "skús Arbortech
    Turbo Plane alebo frézky Manpa" — zákazník dostal len holé mená bez
    ceny a odkazu na kúpu. Táto funkcia sa pokúsi k spomenutým značkám
    dohľadať konkrétny reálny produkt z ponuky, aby sa dal pod odpoveď
    pripojiť ako klikateľná karta (podobne ako pri priamej produktovej
    otázke). Zámerne vracia najviac `limit` produktov, jeden na značku —
    nie je to úplná zhoda, len najlepší odhad."""
    text_lower = text.lower()
    words = [w for w in re.findall(r'\w+', text_lower) if len(w) > 3]
    if not words:
        return []

    # Výrobca býva vo feede uložený ako zložený reťazec (napr. "saburrtooth
    # usa", "king arthur,usa") — zákazník/AI ale bežne spomenie len samotnú
    # značku ("Saburrtooth"), takže sa neporovnáva celá fráza, len jej
    # jednotlivé dosť dlhé (a teda dosť špecifické) slová.
    manufacturers = {p['manufacturer'].strip() for p in get_products() if p['manufacturer'].strip()}
    mentioned_manufacturers = set()
    for m in manufacturers:
        tokens = [t for t in re.findall(r'\w+', m) if len(t) >= 4]
        if any(re.search(r'\b' + re.escape(t) + r'\b', text_lower) for t in tokens):
            mentioned_manufacturers.add(m)
    if not mentioned_manufacturers:
        return []

    best_by_manufacturer = {}
    for p in get_products():
        manufacturer = p['manufacturer'].strip()
        if manufacturer not in mentioned_manufacturers:
            continue
        # Slová samotného výrobcu (napr. "saburrtooth") sa nepočítajú do
        # zhody na produkte — inak by sa pri čisto všeobecnom spomenutí
        # značky ("skús Saburrtooth alebo Manpa", bez konkrétneho modelu)
        # vždy vybral nejaký ľubovoľný SKU tej značky, hoci text nehovoril
        # o žiadnom konkrétnom z desiatok takmer identických variantov.
        manufacturer_tokens = set(re.findall(r'\w+', manufacturer))
        matched = [
            w for w in words
            if (w in p['name'] or w in p['name_bigrams'])
            and w not in GENERIC_PRODUCT_WORDS and w not in manufacturer_tokens
        ]
        if not matched:
            continue
        score = 10 * len(matched)
        current = best_by_manufacturer.get(manufacturer)
        if current is None or score > current[1]:
            best_by_manufacturer[manufacturer] = (p, score)

    ranked = sorted(best_by_manufacturer.values(), key=lambda item: -item[1])
    return [p for p, _ in ranked[:limit]]

# ------------------ ENDPOINT /chat ------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    session_id = data.get("session_id") or "default"
    history = get_history(session_id)

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
        current_shipping_payment_info = get_shipping_payment_info()
        shipping_payment_block = (
            f"\n\nDOPRAVA A PLATBA (použi len ak sa zákazník pýta na dopravu/platbu k tomuto produktu):\n{current_shipping_payment_info}"
            if current_shipping_payment_info else ""
        )
        system_prompt = f"""Si odborný a priateľský poradca pre rezbárske náradie v e-shope Marosko. Zákazník sa pýta na konkrétny produkt nižšie. Zohľadni pri odpovedi aj predchádzajúcu časť konverzácie nižšie, ak je k dispozícii — napríklad ak ťa zákazník už opravil alebo doplnil, neopakuj pôvodnú chybu.

{LANGUAGE_INSTRUCTION}

Odpovedz prirodzene, vecne a v plných vetách, nie len strohým výpisom údajov. Použi cenu, výrobcu aj popis, ak sú pre otázku relevantné. Ak sa niečo v údajoch nenachádza, úprimne to priznaj namiesto vymýšľania. Keď zobrazuješ odkazy, používaj čisté URL bez zátvoriek. Údaje o produkte nižšie sú v slovenčine — ak odpovedáš v inom jazyku, preformuluj ich.

PRODUKT: {product['original_name']}
VÝROBCA: {product['manufacturer']}
CENA: {product['price']} € s DPH
ODKAZ NA KÚPU: {clean_url_link}
POPIS: {product['description']}{shipping_payment_block}"""

        messages = [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": user_msg}
        ]

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
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
            append_to_history(session_id, user_msg, ai_msg)
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
    current_llms_context = get_llms_context()
    current_shipping_payment_info = get_shipping_payment_info()
    shipping_payment_block = (
        f"\n\nDOPRAVA A PLATBA (použi, ak sa zákazník pýta na dopravu, dodanie alebo platbu):\n{current_shipping_payment_info}\n"
        if current_shipping_payment_info else ""
    )
    if current_llms_context and current_llms_context.strip():
        system_prompt = f"""Si odborný poradca pre rezbárske náradie. {LANGUAGE_INSTRUCTION} Buď užitočný a presný. Ak nepoznáš odpoveď, povedz to. Keď zobrazuješ odkazy, používaj čisté URL bez zátvoriek. Zohľadni pri odpovedi aj predchádzajúcu časť konverzácie nižšie, ak je k dispozícii.

Tu máš informácie o e-shope Marosko (kategórie, dôležité stránky, blog, kontakty):

{current_llms_context}
{shipping_payment_block}
Použi tieto informácie, ak sú relevantné k otázke používateľa. Neuvádzaj však priamo, že si čerpal z llms.txt. Odpovedaj prirodzene."""
    else:
        system_prompt = f"Si odborný poradca pre rezbárske náradie. {LANGUAGE_INSTRUCTION} Buď užitočný a presný. Ak nepoznáš odpoveď, povedz to. Keď zobrazuješ odkazy, používaj čisté URL bez zátvoriek.{shipping_payment_block}"

    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_msg}
    ]

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": False,
        "temperature": 0.5
    }
    try:
        resp = requests.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        ai_msg_raw = resp.json()["choices"][0]["message"]["content"]
        locale, ai_msg = extract_language(ai_msg_raw, user_msg)
        # Vyčisti AI odpoveď od zátvoriek v URL
        ai_msg = clean_ai_response(ai_msg)
        append_to_history(session_id, user_msg, ai_msg)

        mentioned_products = find_products_mentioned(ai_msg)
        if mentioned_products:
            labels = LABELS[locale]
            cards = "\n".join(
                f"- {p['original_name']} – {p['price']} € | {clean_url(p['url'])}"
                for p in mentioned_products
            )
            final_response = f"{ai_msg}\n\n---\n**{labels['recommended']}:**\n{cards}"
        else:
            final_response = ai_msg
        return jsonify({"success": True, "response": final_response})
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
