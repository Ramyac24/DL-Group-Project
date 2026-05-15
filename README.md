# Multimodal RAG for Chest X-ray Report Generation
CSE 676 — Spring 2026 Group Project.

## Notebooks (run in order)
- `notebooks/01_classifier.ipynb` — ViT-B/16 fine-tuned classifier + TorchXRayVision baseline (NIH ChestX-ray14).
- `notebooks/02_iu_prep.ipynb` — Indiana University X-ray prep + patient-level splits.
- `notebooks/03_rag_pipeline.ipynb` — Knowledge base, hybrid retriever (Dense + BM25 + label-conditioned + RRF), local Qwen2.5-7B generator, negation-aware verifier.
- `notebooks/04_evaluate.ipynb` — BLEU/ROUGE/BERTScore + rule-based hallucination + LLM-as-judge + bootstrap CIs.

## Results
See `results/` for the headline tables.

## Datasets
- **NIH ChestX-ray14** — classifier training (via Kaggle).
- **Indiana University Chest X-rays (OpenI)** — RAG eval (via Kaggle, see notebook 02).
