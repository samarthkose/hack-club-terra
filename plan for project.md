step 1
set up ai rig
step 2
start actually makign the rocm for rx580
step 3 start working o na rag pipeline, need to incorporate
* Embed model
* reranker
* decision model
* large llm to be able to process it all, 256 (prerffered), 512 (holy smokes), 1M (bonkers) context sizes.



look of it

                    ┌─────────────────────────────────────────────┐
                    │            TERRARIUM (Ubuntu)                │
  your PC ──eth──►  │  FastAPI+UI ──► RAG pipeline (Python)        │
  (SSH + browser)   │       │                                      │
                    │       ├── embed  ──► llama-server :8081  (GPU2)│
                    │       ├── rerank ──► llama-server :8082  (GPU3)│
                    │       └── chat   ──► llama-server :8080 (GPU0,1)│
                    │  vector index + BM25 (numpy/sqlite)           │
                    └─────────────────────────────────────────────┘

                    