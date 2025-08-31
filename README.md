# ArtRAG
This repository contains the code for the paper titled "ArtRAG: Retrieval-Augmented Generation with Structured Context for Visual Art Understanding".

**ArtRAG** is a Retrieval-Augmented Generation (RAG) framework tailored for multimodal visual art undterstanding. It integrates large language models (LLMs) and vision-language processing to generate and evaluate art-related content.

---

## 📁 Project Structure

```
ArtRAG_software/
│
├── lightrag/                   
│   ├── base.py                  # Base functionality
│   ├── clip_score.py            # CLIP-based scoring for images
│   ├── generation_eval_utils.py # Utilities for evaluation
│   ├── lightrag.py              # RAG implementation of lightrag
│   ├── llm.py                   # LLM interaction wrapper
│   ├── operate.py               # Core execution logic
│   ├── prompt_*.py              # Prompt templates and generators
│   ├── prunning.py              # Pruning logic for candidate outputs
│   ├── shared_storage.py        # Shared memory/storage handler
│   ├── storage.py               # Storage utilities
│   └── utils.py                 # General utilities
│
├── built_graph/                # Prebuilt knowledge/graph resources
│   └── All_gpt_4o_mini_clean    # Example graph data
│
├── scripts/                    # Evaluation and setup scripts
│   ├── build_graph.py
│   ├── clean_graph.py
│   ├── inference_eval_artpedia.py
│   ├── inference_eval.py
│   ├── run_eval.sh
│   └── art_evaluation_data.csv
│
└── README.md                   # You're reading it!
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip` or `conda`
- Access to OpenAI or HuggingFace API 

---

## 🧠 Core Concepts

This framework is designed around the concept of **retrieval-augmented generation** applied to **art and multimodal tasks**.

- **Prompting Strategies:** Configurable in `prompt_*.py`
- **Model Execution:** Controlled via `llm.py` and `operate.py`
- **Evaluation:** Scripts in `/scripts` help benchmark output against datasets like **Artpedia**

---

## 🛠 Example Usage

### Build a Knowledge Graph

```bash
python scripts/build_graph.py
```

### Run Inference

```bash
python scripts/inference_eval.py
```

### Evaluate Against Artpedia

```bash
python scripts/inference_eval_artpedia.py
```

Or use the shell script:

```bash
bash scripts/run_eval.sh
```

The code base is built on lightrag project. Thanks for the great work.
