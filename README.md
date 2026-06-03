# Constitutional Compliance Analyzer

A professional, Streamlit-powered legal-tech application designed to automatically review and analyze legal documents (PDFs) against the Constitution of Pakistan. 

By leveraging Retrieval-Augmented Generation (RAG), this tool semantically matches document clauses with relevant constitutional articles and uses a powerful LLM to generate clear, plain-English compliance assessments.

##  Key Features

- **Automated Document Processing**: Instantly parses and intelligently chunks uploaded PDF documents.
- **Semantic Constitutional Search**: Uses FAISS vector search to find the most relevant constitutional articles for every clause.
- **AI-Powered Legal Analysis**: Powered by Llama-3.3-70B (via Groq) to evaluate compliance, identify violations, and assess risk levels.
- **Professional UI/UX**: Clean, jargon-free interface designed for legal professionals, compliance officers, and laymen. No overwhelming technical details.
- **Exportable Reports**: Generate and download formal compliance summaries in both JSON and TXT formats.

##  Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (CPU-optimized)
- **LLM**: Groq API (`llama-3.3-70b-versatile`)
- **Document Parsing**: `PyPDF2`

##  Getting Started

### Prerequisites
- Python 3.9 or higher
- A valid [Groq API Key](https://console.groq.com/keys)

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/constitutional-compliance-analyzer.git
   cd constitutional-compliance-analyzer
