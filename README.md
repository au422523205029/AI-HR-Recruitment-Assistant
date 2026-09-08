# AI HR Recruitment Assistant — Gemini Free Tier

Agent + Tools + RAG application for:
1. PDF resume ingestion
2. Gemini embeddings + FAISS semantic retrieval
3. Job-description matching
4. Candidate scoring/ranking
5. Evidence-backed strengths and gaps
6. Tailored interview-question generation

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set your Google AI Studio API key as `GOOGLE_API_KEY`, then:

```bash
streamlit run main.py
```

## Streamlit Cloud Secrets

```toml
GOOGLE_API_KEY = "YOUR_GOOGLE_AI_STUDIO_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```

Do not commit your API key to GitHub.

Human review is required for employment decisions. Do not use protected characteristics or proxies.
