# Insurance RAG Assistant

A Python notebook prototype for answering questions using information retrieved from documents. It combines **sentence-transformer embeddings**, **ChromaDB semantic search**, and **Google Gemini** to generate answers with supporting document references.

The intended application is insurance policy question answering. The current notebook demonstrates a general document question-answering pipeline with health-related example questions; insurance-specific performance has not been evaluated.

## Purpose

Long documents can make it difficult to locate information quickly. This project retrieves relevant passages for a question and supplies them to a language model as context for its answer.

This technique is called **retrieval-augmented generation (RAG)**. Instead of relying only on the language model's general knowledge, the application searches the documents provided by the user. For an insurance use case, those documents could contain coverage descriptions, exclusions, or claims procedures.

The prompt instructs the model to acknowledge when the retrieved context does not contain an answer. This helps guide responses but does not guarantee correctness; users should verify answers against the source documents.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Document processing and retrieval/generation workflow |
| PyPDF2 | Extracting text from PDF documents |
| Sentence Transformers | Creating embeddings with `all-MiniLM-L6-v2` |
| ChromaDB | Persisting document chunks and embeddings; retrieving similar passages |
| Google Gemini | Generating answers from retrieved context |
| Google Gen AI SDK | Calling Gemini directly |
| OpenAI Python SDK | Demonstrating Gemini access through its OpenAI-compatible endpoint |
| Jupyter Notebook / VS Code | Interactive execution and inspection |
| Python `os` module | File discovery, paths, and source metadata |

The notebook uses Gemini for response generation through both SDK approaches. It does not require LangChain or LlamaIndex.

## How the Workflow Works

1. **Read documents:** extract text from PDF or plain-text files.
2. **Split text:** group sentences into chunks with a target size of approximately 500 characters. Individual long sentences can exceed that target.
3. **Create metadata:** attach the source filename and chunk number to each passage.
4. **Index documents:** use sentence-transformer embeddings and store the passages in a persistent ChromaDB collection.
5. **Retrieve context:** search for passages whose embeddings are similar to the user's question.
6. **Generate an answer:** send the question and retrieved passages to Gemini with instructions to answer from that context.
7. **Return references:** return the answer together with source filenames and chunk numbers.

In plain language: **find the relevant document passages, then ask the model to explain them in response to the question.**

## Implemented Features

- PDF and TXT text extraction.
- Sentence-based text chunking.
- Persistent local vector storage in `./chroma_db`.
- Batch insertion of chunks, up to 100 per batch.
- Semantic retrieval with source metadata and similarity distances.
- Context-based answer generation with Gemini.
- A helper that returns an answer and supporting source references.

The notebook title refers to conversational RAG, but the current implementation processes each question independently. Conversation history and follow-up question resolution are future extensions.

## Configuration

| Setting | Notebook value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` |
| Generation model identifier | `gemini-2.5-flash` |
| Chunk-size target | 500 characters |
| ChromaDB location | `./chroma_db` |
| Collection name | `documents_collection` |
| Default semantic search results | 4 passages |
| Default `rag_query()` retrieval | 2 passages |

The generation model identifier is taken from the notebook. Access depends on the API account and model availability.

## Suggested Repository Files

| File or folder | Purpose |
|---|---|
| `insurance_rag_assistant.ipynb` | Document processing, indexing, retrieval, and generation notebook |
| `docs/` | Documents supplied locally for indexing |
| `README.md` | Project overview and setup instructions |
| `.gitignore` | Excludes credentials and generated local data |

The source documents referenced in the uploaded notebook were not included with the project uploads. Supply documents you are permitted to use. Keep confidential policy documents out of a public repository.

## Setup and Running

Install the packages in your Python environment:

```bash
python -m pip install chromadb sentence-transformers PyPDF2 google-genai openai jupyter ipykernel
```

These dependencies are not version-pinned. Open the notebook in VS Code or Jupyter and select the Python environment where they are installed.

### 1. Configure credentials

Replace both hardcoded API-key values with an environment variable lookup:

```python
api_key = os.environ["GEMINI_API_KEY"]
```

Use it when initializing the clients:

```python
gemini_client = genai.Client(api_key=api_key)

openai_compatible_client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
```

Update the corresponding generation functions to use these client names. Set `GEMINI_API_KEY` in the environment that starts the notebook kernel. Do not place an actual key in the notebook or commit it. If a real key has already been exposed publicly, revoke and replace it.

### 2. Supply documents

Create a `docs` folder and add a text-based PDF or TXT file. Replace the notebook's machine-specific `H:` drive path with the path to your document, for example:

```python
text = read_document("./docs/sample_policy.pdf")
```

The indexing function reads files from `./docs`. DOCX support is incomplete: the dispatcher includes a DOCX branch, but its reader is commented out. Use PDF or TXT with the current code. Scanned PDFs require a separate OCR step.

### 3. Choose one generation route

The original `rag_query()` calls both generation functions and overwrites the first answer with the second. Keep one call to avoid issuing two API requests per question:

```python
def rag_query(collection, query: str, n_chunks: int = 2):
    results = semantic_search(collection, query, n_chunks)
    context, sources = get_context_with_sources(results)
    response = generate_response_gemini_new(query, context)
    return response, sources
```

This uses the direct Gemini SDK route.

### 4. Run and inspect answers

Run the setup, document-processing, and indexing cells in order. Then ask a question relevant to your documents:

```python
answer, sources = rag_query(
    collection,
    "What exclusions are described in this policy?",
)

print(answer)
for source in sources:
    print(source)
```

This is an illustrative insurance question, not a recorded project result. The answer depends on the documents indexed. Enable source printing in the final notebook cell; it is commented out in the uploaded version.

Index a document once in a fresh collection. The current insertion function uses deterministic IDs with `collection.add()` and does not implement a document-update workflow.

## Evaluation and Limitations

No measured retrieval accuracy, answer-quality score, or latency benchmark is reported. The current project is a notebook prototype with no deployed UI or API, conversation memory, or automated evaluation suite.

References identify the retrieved chunks, rather than verifying individual statements in the generated answer. Chunking has no overlap, PDF extraction has no OCR fallback, and metadata does not preserve page numbers. Only retrieved text is sent as answer context, but that text is transmitted to the external Gemini API.

## Potential Improvements

- Demonstrate the workflow with an appropriate sample insurance document.
- Add chat history and follow-up question handling.
- Track page numbers and display clearer supporting quotations.
- Add overlapping chunks, document updates, and retrieval-quality evaluation.
- Test questions with known answers and questions unsupported by the documents.
- Package the validated workflow behind an API or user interface.

## Author

**Ritika Dharamkar**

- [GitHub](https://github.com/RitikaDharamkarJ)
- [LinkedIn](https://www.linkedin.com/in/ritikadharamkar/)
