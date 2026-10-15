# Clinic Context

Notebook that builds a local medical Q&A stack: chunk documents, embed them, retrieve with FAISS, then generate with LangChain. Early runs used Flan-T5 Small to prove the pipe; a stronger generator (Gemini/GPT-class) is the intended production head.

Not clinical advice. Answers are only as good as the source corpus.

## Pipeline

1. Load PDFs / text
2. Split with overlap (typical 300–500 tokens, 50–100 overlap)
3. Embed (e.g. MiniLM or mpnet, no API key)
4. Store vectors
5. Query → top-k chunks → prompt → LLM

LangChain shape:

```
{"context": retriever | format_docs, "question": RunnablePassthrough()}
| RAG_PROMPT | llm | StrOutputParser()
```

## Open this project

Open `clinic_context.ipynb` in Jupyter or VS Code.

## Design notes

Small generators tend to ramble or repeat. Keep the prompt tight (“answer only from context”). Cosine similarity is the default retrieval metric. Swap the vector store (Chroma, Qdrant, pgvector) without changing the rest of the chain if you outgrow in-notebook FAISS.
