# Resume Screener

A Streamlit app that lets you ask natural-language questions about a folder of resumes (PDFs). It uses a Retrieval-Augmented Generation (RAG) pipeline built with LangChain and Google's Gemini models to search resume content and answer questions with source attribution.

---

## What is RAG, and why use it here?

**Retrieval-Augmented Generation (RAG)** is a pattern for grounding an LLM's answers in your own private data, instead of relying only on what the model memorized during training. An LLM by itself:

- Has never seen your resumes — it has no idea who your candidates are.
- Has a limited context window — you can't just paste hundreds of resumes into every prompt.
- Can hallucinate facts if asked about things it doesn't actually know.

RAG solves this with a **retrieve-then-generate** pattern:

1. **Index (offline, done once):** Break your documents into chunks, convert each chunk into a numerical vector ("embedding") that captures its meaning, and store those vectors in a searchable index.
2. **Retrieve (per question):** Convert the user's question into a vector too, then find the chunks whose vectors are closest in meaning — this is *semantic search*, not keyword matching, so "salary expectations" can match a chunk that says "compensation requirements" even without shared words.
3. **Augment:** Insert those retrieved chunks into the prompt as context.
4. **Generate:** Send the augmented prompt to the LLM, which now answers using the actual resume text rather than guessing.

This is why the app can answer "Which candidates have AWS experience?" accurately — the LLM is reading real excerpts from your resumes at answer time, not recalling anything from training.

## Why LangChain?

**LangChain** is an orchestration framework that provides standardized, swappable building blocks for LLM pipelines like this one, so you don't have to hand-write the plumbing between each stage. In this app, LangChain provides:

- **Document loaders** (`DirectoryLoader`, `PyPDFLoader`) — a consistent interface for pulling text out of many file types, so `.pdf`, `.docx`, `.txt`, etc. could all be swapped in without changing the rest of the pipeline.
- **Text splitters** (`RecursiveCharacterTextSplitter`) — reusable chunking logic tuned for how LLMs consume text.
- **Vector store integrations** (`FAISS`) — a common interface so you could swap FAISS for Pinecone, Chroma, or Weaviate later with minimal code changes.
- **Embeddings & chat model wrappers** (`GoogleGenerativeAIEmbeddings`, `ChatGoogleGenerativeAI`) — standardized interfaces so swapping providers (Google, OpenAI, Anthropic, etc.) doesn't require rewriting your chain logic.
- **Chains** (`RetrievalQA`) — pre-built orchestration that wires a retriever and an LLM together (retrieve → stuff into prompt → generate → return sources) so you don't have to write that control flow by hand.

In short: LangChain doesn't add intelligence — the LLM and embeddings do that — it standardizes how the pieces of a RAG pipeline snap together, and makes it easy to swap any one piece later.

## The two models doing the actual work

This app uses **two separate Gemini models for two different jobs** — a common and important RAG pattern:

| Model | Role | Why it's separate |
|---|---|---|
| `gemini-embedding-001` | **Embedding model** — converts text into vectors for semantic search | Optimized purely to represent meaning as numbers; doesn't generate text |
| `gemini-3.5-flash` | **Chat/generation model (the LLM)** — reads retrieved chunks and writes a natural-language answer | Optimized for reasoning and fluent text generation, not for producing embeddings |

They live in different "spaces" — you can't compare an embedding-model vector to anything the chat model produces. The embedding model's only job is retrieval (step 2 above); the chat model's only job is generation (step 4 above). This separation of concerns is exactly what `RetrievalQA` orchestrates for you.

---

## How `app.py` Works

The app builds a **RAG (Retrieval-Augmented Generation) pipeline** that lets an LLM answer questions grounded in your resume PDFs, instead of relying on the model's general knowledge. Here's what each step in `build_chain()` does, mapped back to the retrieve → augment → generate stages above:

### 1. Load environment variables
```python
load_dotenv()
```
Reads your `.env` file (containing `GOOGLE_API_KEY`) and loads it into the environment so LangChain's Google integrations can authenticate automatically.

### 2. Load resume PDFs — *(preparing data for indexing)*
```python
loader = DirectoryLoader("resumes", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
```
Scans the `resumes/` directory for every `.pdf` file and extracts their text content using `PyPDFLoader`. Each page becomes a LangChain `Document` object with the text plus metadata (like the source filename) — that `source` metadata is what later lets the app tell you *which resume* an answer came from.

### 3. Split documents into chunks — *(preparing data for indexing)*
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
```
Large documents are broken into smaller ~1000-character chunks. This matters for two RAG-specific reasons:
- **Embedding quality:** embedding models represent meaning best over focused passages, not entire multi-page documents — a giant blob of text produces a vague, averaged-out vector that's harder to match precisely.
- **Prompt budget:** the LLM only sees the *top-k retrieved chunks* at answer time (see step 6), not entire resumes, so chunks need to be small enough that several of them fit comfortably in the prompt.

The 150-character **overlap** between consecutive chunks prevents a sentence or fact from being awkwardly cut in half at a chunk boundary and losing meaning in either half.

### 4. Generate embeddings and build a vector store — *(the "Index" stage)*
```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document",
)
vectordb = FAISS.from_documents(chunks, embeddings)
```
This is where each chunk becomes a searchable vector. `task_type="retrieval_document"` tells the embedding model these vectors will be *searched against* later (as opposed to being search queries themselves) — Gemini's embedding model optimizes the vector slightly differently depending on which side of a search it's used for.

**FAISS** (Facebook AI Similarity Search) then indexes all these vectors in memory, enabling fast **nearest-neighbor search**: given any query vector, it can quickly find which stored vectors are closest to it in the embedding space — i.e., which chunks are most semantically similar to the query.

### 5. Initialize the LLM — *(the "Generate" stage)*
```python
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
```
This is the actual **large language model** that reads context and writes answers — everything before this step was just preparing data for it to read. `temperature=0` removes randomness from its output, so the same question against the same resumes should produce a consistent, repeatable answer each time — important for a screening tool where reproducibility matters more than creative variety.

### 6. Build the retrieval QA chain — *(wires Retrieve → Augment → Generate together)*
```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
)
```
This is where LangChain assembles the full RAG pipeline into one callable object:

1. **Retrieve:** `vectordb.as_retriever(search_kwargs={"k": 4})` turns the FAISS index into a retriever that returns the **top 4** most relevant chunks for any given question (embedding the question first, then running the same nearest-neighbor search described in step 4).
2. **Augment:** Behind the scenes, `RetrievalQA` uses a `"stuff"` chain type by default — meaning it literally "stuffs" all 4 retrieved chunks directly into the prompt template alongside the user's question, before sending anything to the LLM.
3. **Generate:** The augmented prompt (question + 4 chunks of real resume text) goes to `gemini-3.5-flash`, which reads that context and composes an answer grounded in it.
4. **Attribution:** `return_source_documents=True` makes the chain also hand back the raw chunk objects (with their `source` metadata) that were used — this is what powers the "Source Resumes" list in the UI, so you can verify *which* resume(s) informed the answer rather than just trusting the LLM blindly.

**Why `k=4`?** This is a tunable tradeoff: a higher `k` gives the LLM more context (potentially better recall across more candidates) but costs more tokens and can dilute relevance if weaker matches get included; a lower `k` is cheaper and more focused but risks missing a relevant chunk that ranked 5th or 6th.

### 7. Cache the chain
```python
@st.cache_resource
def build_chain():
    ...
```
This decorator ensures the (expensive) steps above — loading PDFs, embedding, and building the FAISS index — only run **once** per app session, not on every user interaction or page rerun.

### 8. Streamlit UI and query flow
```python
question = st.text_input("Ask anything about resumes")
if st.button("Ask") and question:
    result = qa.invoke({"query": question})
    st.write(result["result"])
    for doc in result["source_documents"]:
        st.write("*", os.path.basename(doc.metadata["source"]))
```
- Takes a question from a text input box.
- On button click, runs it through the QA chain.
- Displays the generated answer.
- Lists the unique source resume filenames used to produce that answer, so you can verify which candidates the answer is based on.

## Pipeline at a glance

```
resumes/*.pdf
     │
     ▼
PyPDFLoader ─── extracts raw text per page
     │
     ▼
RecursiveCharacterTextSplitter ─── breaks text into ~1000-char chunks
     │
     ▼
GoogleGenerativeAIEmbeddings (gemini-embedding-001) ─── chunk → vector
     │
     ▼
FAISS index ─── stores vectors, enables similarity search      ┐
                                                                 │  built once,
                                                                 │  cached via
   ── user question ──                                          │  @st.cache_resource
          │                                                     │
          ▼                                                     │
GoogleGenerativeAIEmbeddings ─── question → vector              │
          │                                                     │
          ▼                                                     ┘
FAISS similarity search ─── returns top k=4 closest chunks
          │
          ▼
Prompt = question + retrieved chunks  ("stuffed" together)
          │
          ▼
ChatGoogleGenerativeAI (gemini-3.5-flash) ─── generates grounded answer
          │
          ▼
Answer + source resume filenames ─── shown in Streamlit UI
```

---

## Project Structure

```
project/
├── app.py
├── requirements.txt
├── .env                  # not committed — contains GOOGLE_API_KEY
└── resumes/              # put your candidate PDF resumes here
    ├── candidate1.pdf
    ├── candidate2.pdf
    └── ...
```

---

## Setup & Running Instructions

### 1. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> Once activated, your terminal prompt will show `(venv)` at the beginning of the line, confirming you're inside the virtual environment.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the project root (same folder as `app.py`) with:
```
GOOGLE_API_KEY=your_google_api_key_here
```
Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Add resumes
Place your candidate PDF files inside the `resumes/` directory (create it if it doesn't already exist):
```bash
mkdir -p resumes
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```
This starts a local server and should automatically open the app in your browser, typically at:
```
http://localhost:8501
```
If it doesn't open automatically, copy that URL into your browser manually.

### 6. Stop the app
In the terminal running Streamlit, press:
```
Ctrl + C
```

### 7. Deactivate the virtual environment
Once you're done working, exit the virtual environment with:
```bash
deactivate
```
This works the same way across macOS, Linux, and Windows — just run `deactivate` from within the activated environment.

---

## Notes

- The FAISS index and PDF loading happen once per session thanks to `@st.cache_resource`. If you add or change resumes in the `resumes/` folder, restart the Streamlit app (or clear the cache via Streamlit's menu) to rebuild the index with the updated files.
- Make sure the Gemini model names used in `app.py` (`gemini-3.5-flash`, `gemini-embedding-001`) are still current — Google periodically deprecates older model versions. Check [Google's model list](https://ai.google.dev/gemini-api/docs/models) if you hit a `404 NOT_FOUND` error.