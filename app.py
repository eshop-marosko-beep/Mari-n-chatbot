import os
import glob
import uuid
import re
import requests
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

app = Flask(__name__)
CORS(app, origins=['https://eshop.marosko.sk', 'https://www.eshop.marosko.sk'])

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    print("FATAL: DEEPSEEK_API_KEY is NOT set!")
else:
    print(f"DEBUG: API key loaded: {DEEPSEEK_API_KEY[:10]}...")

PRODUCT_XML_URL = "https://eshop.marosko.sk/erp/impexp/specialexport/heureka"
PRODUCT_CHROMA_PATH = "/tmp/product_chroma"

print("🔄 Načítavam embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_and_index_products():
    print("🔄 Sťahujem XML feed produktov...")
    try:
        resp = requests.get(PRODUCT_XML_URL, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"❌ Chyba pri sťahovaní/parsovaní XML: {e}")
        return None

    documents = []
    for item in root.findall(".//SHOPITEM"):
        item_id = item.findtext("ITEM_ID", "")
        name = item.findtext("PRODUCTNAME", "")
        manufacturer = item.findtext("MANUFACTURER", "")
        price_vat = item.findtext("PRICE_VAT", "")
        url = item.findtext("URL", "")
        description = item.findtext("DESCRIPTION", "")
        clean_desc = re.sub(r'<[^>]+>', ' ', description)
        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()

        content = f"""
Názov: {name}
Výrobca: {manufacturer}
Cena: {price_vat} € s DPH
Popis: {clean_desc[:800]}
Link na produkt: {url}
        """.strip()

        metadata = {
            "item_id": item_id,
            "name": name,
            "price": price_vat,
            "url": url,
            "manufacturer": manufacturer
        }
        documents.append((content, metadata))

    if not documents:
        print("⚠️ Žiadne produkty neboli nájdené v XML.")
        return None

    texts = [doc[0] for doc in documents]
    metadatas = [doc[1] for doc in documents]
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=PRODUCT_CHROMA_PATH
    )
    vectordb.persist()
    print(f"✅ Indexovaných {len(documents)} produktov.")
    return vectordb

DOCS_FOLDER = "docs"
GENERAL_CHROMA_PATH = "/tmp/general_chroma"

def load_and_index_general_docs():
    if not os.path.exists(DOCS_FOLDER):
        print("📁 Priečinok 'docs' neexistuje, preskakujem všeobecné dokumenty.")
        return None

    all_docs = []
    for pdf in glob.glob(os.path.join(DOCS_FOLDER, "*.pdf")):
        try:
            loader = PyPDFLoader(pdf)
            all_docs.extend(loader.load())
            print(f"   Načítaný PDF: {pdf}")
        except Exception as e:
            print(f"   Chyba pri načítaní PDF {pdf}: {e}")
    for txt in glob.glob(os.path.join(DOCS_FOLDER, "*.txt")):
        try:
            loader = TextLoader(txt, encoding="utf-8")
            all_docs.extend(loader.load())
            print(f"   Načítaný TXT: {txt}")
        except Exception as e:
            print(f"   Chyba pri načítaní TXT {txt}: {e}")

    if not all_docs:
        print("⚠️ Žiadne všeobecné dokumenty neboli načítané.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=GENERAL_CHROMA_PATH)
    vectordb.persist()
    print(f"✅ Indexovaných {len(chunks)} všeobecných častí dokumentov.")
    return vectordb

product_db = fetch_and_index_products()
general_db = load_and_index_general_docs()

qa_product = None
if product_db:
    product_retriever = product_db.as_retriever(search_kwargs={"k": 3})
    qa_product = RetrievalQA.from_chain_type(
        llm=OpenAI(
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
            model_name="deepseek-chat",
            temperature=0.3
        ),
        chain_type="stuff",
        retriever=product_retriever
    )
    print("🧠 Produktový RAG pripravený.")

qa_general = None
if general_db:
    general_retriever = general_db.as_retriever(search_kwargs={"k": 4})
    qa_general = RetrievalQA.from_chain_type(
        llm=OpenAI(
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
            model_name="deepseek-chat",
            temperature=0.7
        ),
        chain_type="stuff",
        retriever=general_retriever
    )
    print("🧠 Všeobecný RAG pripravený.")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    session_id = data.get("session_id", str(uuid.uuid4()))

    product_answer = None
    if qa_product:
        try:
            product_result = qa_product.invoke({"query": user_msg})
            product_answer = product_result["result"]
        except Exception as e:
            print("Chyba pri produktovom RAG:", e)

    lower_msg = user_msg.lower()
    is_buy_question = any(word in lower_msg for word in ["kúp", "cena", "koľko stojí", "objednať", "link", "odkaz", "kde kúpim"])

    if is_buy_question and product_answer:
        match = re.search(r'(https?://[^\s]+)', product_answer)
        if match:
            link = match.group(1)
            final = f"{product_answer}\n\n👉 **Kúpiť si ho môžeš tu:** {link}"
        else:
            final = product_answer
        return jsonify({"success": True, "response": final})

    if product_answer:
        return jsonify({"success": True, "response": product_answer})

    if qa_general:
        try:
            general_answer = qa_general.invoke({"query": user_msg})
            return jsonify({"success": True, "response": general_answer["result"]})
        except Exception as e:
            print("Chyba pri všeobecnom RAG:", e)

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
