ROCm pipeline, whole local ai project. here is the plan:

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

Week by week plan

1           Build, flash VBIOS, POST, Ubuntu, SSH over ethernet

2           Docker + ROCm gfx803 images, rocminfo sees 4 GPUs

3           Build llama.cpp with HIP (Vulkan fallback), 4-GPU split, tok/s benchmark

4           Models downloaded, 3 llama-server instances as systemd/compose services

5-6         Chunker + vector store + BM25 + RRF, ingest CLI

7-8         Reranker + prompt/answer pipeline + citations

9-10        FastAPI + web UI, remote demo from your main PC

11          Hybrid-search tuning, eval set, benchmark page

12          Docs, screenshots, README, demo video

13          Polish + submit



