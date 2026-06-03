# ==========================================
# Legal Advisor - Constitutional Compliance
# ==========================================

# To install dependencies locally, run this in your terminal:
# pip install faiss-cpu PyPDF2 sentence-transformers groq

import os
import re
import json
import pickle
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# Handle Colab vs Local environment

# ── CONFIG ───────────────────────────────
GROQ_API_KEY  = "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"   # ← replace with your actual key
GROQ_MODEL    = "llama-3.3-70b-versatile"
PDF_PATH      = "Constitution.pdf"
FAISS_INDEX   = "constitution.index"
CHUNKS_FILE   = "chunks.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 3
# ─────────────────────────────────────────


# ==========================================
# 1. Text Extraction & Chunking Helpers
# ==========================================
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    print(f"  Extracted {len(pages)} pages")
    return "\n".join(pages)

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text.strip()

def make_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, start, cid = [], 0, 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        last_period = max(chunk_text.rfind(". "), chunk_text.rfind("? "), chunk_text.rfind("! "))
        if last_period != -1 and last_period > len(chunk_text) * 0.5:
            chunk_text = chunk_text[:last_period + 1]
        chunks.append({"id": cid, "text": chunk_text.strip()})
        cid += 1
        start += chunk_size - overlap
    print(f"  Created {len(chunks)} chunks")
    return chunks


# ==========================================
# 2. Ingest Constitution PDF (Run once)
# ==========================================
print("[1/4] Reading PDF...")
raw = clean_text(extract_text(PDF_PATH))

print("[2/4] Chunking...")
chunks = make_chunks(raw)

print("[3/4] Embedding...")
embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode([c["text"] for c in chunks], batch_size=64,
                             show_progress_bar=True, convert_to_numpy=True).astype("float32")

print("[4/4] Building FAISS index...")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"\n✅ Done — {index.ntotal} vectors saved")


# ==========================================
# 3. Load Index & Initialize Groq Client
# ==========================================
client = Groq(api_key=GROQ_API_KEY)
index = faiss.read_index(FAISS_INDEX)

with open(CHUNKS_FILE, "rb") as f:
    const_chunks = pickle.load(f)

print(f"✅ FAISS index loaded  — {index.ntotal} vectors")
print(f"✅ Constitution chunks — {len(const_chunks)} chunks")
print(f"✅ Groq client ready   — {GROQ_MODEL}")


# ==========================================
# 4. Analysis Helpers
# ==========================================
def embed_texts(texts):
    vecs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

def search_constitution(query_vec, top_k=TOP_K):
    scores, ids = index.search(query_vec, top_k)
    return [{"chunk_id": int(idx), "score": round(float(s), 4),
              "text": const_chunks[idx]["text"]}
            for s, idx in zip(scores[0], ids[0])]

def build_prompt(user_clause, const_articles):
    articles_text = "\n\n---\n\n".join(
        [f"[Chunk {i+1} | score: {a['score']}]\n{a['text']}"
         for i, a in enumerate(const_articles)]
    )
    return f"""You are a Pakistani legal analyst. Analyze this clause against the Constitution of Pakistan.

CLAUSE:
\"\"\"{user_clause}\"\"\"

RELEVANT CONSTITUTIONAL ARTICLES:
\"\"\"{articles_text}\"\"\"

Respond ONLY in this exact JSON format:
{{
   "verdict": "COMPLIANT" | "VIOLATION" | "POTENTIAL CONFLICT" | "UNRELATED",
   "affected_articles": ["Article X", "Article Y"],
   "explanation": "2-3 sentence explanation.",
   "risk_level": "LOW" | "MEDIUM" | "HIGH"
}}"""

def call_llm(prompt):
    """Calls Groq (Llama 3.3 70B) — the only LLM used here."""
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            response_format={"type": "json_object"},
            temperature=0.1      # lower = faster + more consistent
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"verdict": "ERROR", "affected_articles": [],
                 "explanation": str(e), "risk_level": "HIGH"}

print("✅ All helpers ready")


# ==========================================
# 5. Process User Document
# ==========================================
user_pdf_path = input("📂 Enter the path to your legal document PDF: ")

raw = clean_text(extract_text(user_pdf_path))
user_chunks = make_chunks(raw)
print(f"📄 {user_pdf_path} → {len(user_chunks)} chunks\n")

# ── Batch embed ALL user chunks at once (fast) ──
print("⚡ Embedding all chunks at once...")
all_vecs = embed_texts([c["text"] for c in user_chunks])

# ── Analyze ──
results = []
for i, (chunk, vec) in enumerate(zip(user_chunks, all_vecs)):
    print(f"  Analyzing chunk {i+1}/{len(user_chunks)} ... ", end="\r")
    const_articles = search_constitution(vec.reshape(1, -1))
    prompt = build_prompt(chunk["text"], const_articles)
    verdict = call_llm(prompt)
    results.append({"chunk_id": i, "user_clause": chunk["text"],
                     "constitution_matches": const_articles, "analysis": verdict})

print(f"\n✅ Done — {len(results)} chunks analyzed")


# ==========================================
# 6. Generate Reports
# ==========================================
RISK_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", " ": "⚪"}

with open("report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open("report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("       CONSTITUTIONAL COMPLIANCE REPORT\n")
    f.write(f"       Document: {user_pdf_path}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total chunks    : {len(results)}\n")
    f.write(f"Violations      : {len([r for r in results if r['analysis'].get('verdict') == 'VIOLATION'])}\n")
    f.write(f"Potential issues: {len([r for r in results if r['analysis'].get('verdict') == 'POTENTIAL CONFLICT'])}\n\n")

    for r in results:
        a = r["analysis"]
        verdict = a.get("verdict", "N/A")
        if verdict == "COMPLIANT":
            continue
        risk = a.get("risk_level", " ")
        emoji = RISK_EMOJI.get(risk, "⚪")
        f.write("-" * 70 + "\n")
        f.write(f"CHUNK #{r['chunk_id']+1}  |  {emoji} {verdict}  |  Risk: {risk}\n")
        f.write(f"Articles : {', '.join(a.get('affected_articles', []))}\n")
        f.write(f"Clause   : {r['user_clause'][:300]}...\n")
        f.write(f"Analysis : {a.get('explanation', '')}\n\n")

print("✅ Reports saved! (report.json and report.txt)")

