# Multimodal RAG for Chest X-ray Report Generation
CSE 676 — Spring 2026 Group Project.

## Notebooks (run in order)
- `notebooks/01_classifier.ipynb` — ViT-B/16 fine-tuned classifier + TorchXRayVision baseline (NIH ChestX-ray14).
- `notebooks/02_iu_prep.ipynb` — Indiana University X-ray prep + patient-level splits.
- `notebooks/03_rag_pipeline.ipynb` — Knowledge base, hybrid retriever (Dense+BM25+label-conditioned+RRF), local Qwen2.5-7B generator, negation-aware verifier.
- `notebooks/04_evaluate.ipynb` — BLEU/ROUGE/BERTScore + rule-based hallucination + LLM-as-judge + bootstrap CIs.

## Results
See `results/` for the headline tables:
- `final_ablation_table.csv` — full ablation across 3 variants × all metrics
- `metrics_with_ci.csv` — bootstrap 95% CIs
- `judge_scores.csv` — LLM-as-judge scores on 4 clinical axes
- `classifier_compare.csv` — TorchXRayVision vs fine-tuned ViT

## Datasets
- **NIH ChestX-ray14** — classifier training (downloaded via Kaggle).
- **Indiana University Chest X-rays (OpenI)** — RAG eval (downloaded via Kaggle, see notebook 02).
