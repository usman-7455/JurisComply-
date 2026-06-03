# ==========================================
# Constitutional Compliance Analyzer
# Professional Legal Review Interface
# ==========================================
import os
import re
import json
import pickle
import io
import faiss
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


GROQ_MODEL = "llama-3.3-70b-versatile"
FAISS_INDEX = "constitution.index"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
SECTION_SIZE = 400      # Renamed from CHUNK_SIZE for layman understanding
SECTION_OVERLAP = 80    # Renamed from CHUNK_OVERLAP
TOP_K = 3
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Constitutional Compliance Analyzer", 
    page_icon="⚖️", 
    layout="centered" # Centered layout feels more like a formal document
)

# Custom CSS for a clean, professional, legal-tech aesthetic
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 600; color: #1f2937; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4b5563; margin-bottom: 2rem; font-weight: 400; }
    .stExpander { border: 1px solid #e5e7eb; border-radius: 8px; }
    .metric-label { font-size: 0.9rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #111827; }
    div[data-testid="stStatus"] { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Secure Resource Loading
# ==========================================
@st.cache_resource
def load_resources():
    if not os.path.exists(FAISS_INDEX) or not os.path.exists(CHUNKS_FILE):
        st.error("System configuration error: Reference database files are missing.")
        st.stop()
    
    with st.spinner("Initializing secure analysis environment..."):
        embedder = SentenceTransformer(EMBED_MODEL)
        index = faiss.read_index(FAISS_INDEX)
        with open(CHUNKS_FILE, "rb") as f:
            const_sections = pickle.load(f)
    return embedder, index, const_sections

embedder, index, const_sections = load_resources()
client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# 2. Core Logic (Hidden from UI)
# ==========================================
def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text.strip()

def split_into_sections(text):
    words = text.split()
    sections, start, sid = [], 0, 0
    while start < len(words):
        end = min(start + SECTION_SIZE, len(words))
        section_text = " ".join(words[start:end])
        last_period = max(section_text.rfind(". "), section_text.rfind("? "), section_text.rfind("! "))
        if last_period != -1 and last_period > len(section_text) * 0.5:
            section_text = section_text[:last_period + 1]
        sections.append({"id": sid, "text": section_text.strip()})
        sid += 1
        start += SECTION_SIZE - SECTION_OVERLAP
    return sections

def get_embeddings(texts):
    vecs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

def find_relevant_constitution(query_vec):
    scores, ids = index.search(query_vec, TOP_K)
    return [{"score": round(float(s), 4), "text": const_sections[idx]["text"]}
            for s, idx in zip(scores[0], ids[0])]

def build_prompt(clause_text, relevant_articles):
    articles_text = "\n\n---\n\n".join(
        [f"Reference {i+1}:\n{a['text']}" for i, a in enumerate(relevant_articles)]
    )
    return f"""You are an expert Pakistani legal analyst. Review the following contract clause against the Constitution of Pakistan.

CLAUSE UNDER REVIEW:
\"\"\"{clause_text}\"\"\"

RELEVANT CONSTITUTIONAL REFERENCES:
\"\"\"{articles_text}\"\"\"

Respond ONLY in this exact JSON format:
{{
   "verdict": "COMPLIANT" | "VIOLATION" | "POTENTIAL CONFLICT" | "UNRELATED",
   "affected_articles": ["Article X", "Article Y"],
   "explanation": "A clear, plain-English explanation of the finding in 2-3 sentences.",
   "risk_level": "LOW" | "MEDIUM" | "HIGH"
}}"""

def analyze_with_ai(prompt):
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"verdict": "ERROR", "affected_articles": [], "explanation": str(e), "risk_level": "HIGH"}

# ==========================================
# 3. Professional UI Layout
# ==========================================
st.markdown('<div class="main-header">Constitutional Compliance Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a legal document to automatically review its clauses against the Constitution of Pakistan.</div>', unsafe_allow_html=True)

# Sidebar: Clean, high-level info only
with st.sidebar:
    st.markdown("### System Status")
    st.markdown(f"**Reference Database:** Constitution of Pakistan")
    st.markdown(f"**Sections Indexed:** {len(const_sections)}")
    st.markdown(f"**Analysis Engine:** Secure AI (Llama-3-70B)")
    st.markdown("---")
    st.caption("All processing is done securely. No documents are permanently stored on our servers.")

# File Upload
uploaded_file = st.file_uploader("Upload Legal Document", type=["pdf"], help="Supported format: PDF")

if uploaded_file is not None:
    st.markdown("---")
    
    # Modern Step-by-Step Status Tracker
    with st.status("Processing Document", expanded=True) as status:
        
        st.write("Extracting text from the document...")
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        raw_text = "\n".join([p.extract_text().strip() for p in reader.pages if p.extract_text()])
        cleaned_text = clean_text(raw_text)
        document_sections = split_into_sections(cleaned_text)
        
        st.write(f"Document divided into {len(document_sections)} reviewable sections.")
        st.write("Cross-referencing sections with constitutional database...")
        
        results = []
        all_vecs = get_embeddings([s["text"] for s in document_sections])
        
        for i, (section, vec) in enumerate(zip(document_sections, all_vecs)):
            # Update status text dynamically but keep it professional
            status.update(label=f"Reviewing section {i+1} of {len(document_sections)}...")
            
            relevant_articles = find_relevant_constitution(vec.reshape(1, -1))
            prompt = build_prompt(section["text"], relevant_articles)
            analysis = analyze_with_ai(prompt)
            
            results.append({
                "section_id": i + 1, 
                "original_text": section["text"],
                "references": relevant_articles, 
                "analysis": analysis
            })
        
        status.update(label="Analysis Complete", state="complete", expanded=False)

    # ==========================================
    # 4. Results Dashboard
    # ==========================================
    st.markdown("### Review Summary")
    
    # Calculate metrics
    total_sections = len(results)
    compliant_count = sum(1 for r in results if r["analysis"].get("verdict") == "COMPLIANT")
    issue_count = total_sections - compliant_count
    
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="metric-label">Total Sections Reviewed</div>', unsafe_allow_html=True)
    col1.markdown(f'<div class="metric-value">{total_sections}</div>', unsafe_allow_html=True)
    
    col2.markdown('<div class="metric-label">Fully Compliant</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-value" style="color: #059669;">{compliant_count}</div>', unsafe_allow_html=True)
    
    col3.markdown('<div class="metric-label">Requiring Attention</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-value" style="color: #dc2626;">{issue_count}</div>', unsafe_allow_html=True)

    if issue_count > 0:
        st.markdown("### Detailed Findings")
        st.markdown("The following sections contain potential conflicts or violations that require manual legal review.")
        
        issues = [r for r in results if r["analysis"].get("verdict") != "COMPLIANT"]
        
        for r in issues:
            a = r["analysis"]
            verdict = a.get("verdict", "Unknown")
            risk = a.get("risk_level", "Unknown")
            
            # Color code the expander title based on risk
            if risk == "HIGH":
                title_prefix = "[High Risk] "
            elif risk == "MEDIUM":
                title_prefix = "[Medium Risk] "
            else:
                title_prefix = "[Low Risk / Unrelated] "
                
            with st.expander(f"Section {r['section_id']}: {title_prefix}{verdict}"):
                st.markdown("**Analyst Finding:**")
                st.info(a.get("explanation", "No explanation provided by the analysis engine."))
                
                st.markdown("**Relevant Constitutional References:**")
                for ref in r["references"]:
                    st.markdown(f"> *{ref['text'][:250]}...*")
                
                st.markdown("**Original Document Text:**")
                st.markdown(f"\"{r['original_text'][:400]}...\"")

    else:
        st.success("The analysis found no potential conflicts or violations in the uploaded document.")

    # ==========================================
    # 5. Export Options
    # ==========================================
    st.markdown("---")
    st.markdown("### Export Documentation")
    
    # Clean TXT Report Generation
    txt_lines = [
        "CONSTITUTIONAL COMPLIANCE REVIEW REPORT",
        f"Document: {uploaded_file.name}",
        "=" * 50,
        f"Total Sections Reviewed: {total_sections}",
        f"Compliant Sections: {compliant_count}",
        f"Sections Requiring Attention: {issue_count}\n"
    ]
    
    for r in results:
        a = r["analysis"]
        if a.get("verdict") == "COMPLIANT":
            continue
        txt_lines.append(f"SECTION {r['section_id']} - {a.get('verdict')} (Risk: {a.get('risk_level')})")
        txt_lines.append(f"Affected Articles: {', '.join(a.get('affected_articles', ['None']))}")
        txt_lines.append(f"Finding: {a.get('explanation', '')}")
        txt_lines.append(f"Original Text: {r['original_text'][:300]}...\n")
        txt_lines.append("-" * 50)
        
    txt_report = "\n".join(txt_lines)
    
    st.download_button(
        label="Download Formal Review Report (TXT)",
        data=txt_report,
        file_name=f"Compliance_Report_{uploaded_file.name.replace('.pdf', '')}.txt",
        mime="text/plain",
        use_container_width=True
    )