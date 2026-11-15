Get your core ML pipeline running end-to-end in a fully reproducible way. No Docker, cloud deployment, REST APIs, or monitoring yet.

Requirements:
GitHub Repository: Clean, organized repo with a proper README (description, setup steps, how to run, folder structure). All group members must have commits. No raw data or model weights committed — set up .gitignore.
Data Pipeline & Versioning: Ingest data from its source and version with DVC using a remote (DagsHub or AWS S3). Run basic validation checks before training. Must be fully re-runnable.
                          1) Standard ML / CV / Time-Series: Dataset ingested, validated (schema, nulls, ranges), and versioned in DVC.
                          2) RAG / LLM Systems: Documents ingested, chunked, embedded, and indexed into a vector DB (Pinecone, Qdrant,                                      Chroma, etc.). Version ingestion configs and prompt templates in DVC.
                          3) RL / Agentic / Prompt-Based: Version environment configs, simulator parameters, or prompt templates in DVC                                           instead.
Model / System Core: At least one baseline and one main model/agent/system, both evaluated with appropriate metrics.
                          1) Standard / Hugging Face Models: Preprocessing, training, and evaluation scripts as repeatable pipeline runs. One                                     baseline + one main model.
                          2) RAG Systems: Full pipeline end-to-end (query → retrieve → generate), evaluated on a labeled question set.
                          3) Agentic Workflows: Orchestration scripts for multi-agent loops with evaluation on a labeled split (accuracy, latency,                                 etc.).
                          4) Reinforcement Learning: Training loop with at least one baseline policy. Reward curves and key metrics logged per                                  episode.
Experiment Tracking with MLflow: Log every run — hyperparameters, metrics, and artifacts. Register at least one model/checkpoint in the MLflow Model Registry (status: Staging or Production). Must be hosted on DagsHub or a remote server, not local.
                          1) Standard ML / CV / RL: MLflow for training runs, metrics, and model registration.
                          2) LLM / Agent Projects: Langfuse or MLflow for agent traces, token usage, latency, and evaluation accuracy.
DVC Pipeline: Define dvc.yaml with named stages. dvc repro must reproduce the full pipeline from scratch. 


Note on Non-Standard Projects: The same components apply regardless of project type — adapt them to fit. DVC must version something meaningful, MLflow must log something meaningful, and the pipeline must be reproducible. Reach out before the deadline if unsure.

Submission Instructions:
Submit a PDF (3–4 pages) that includes:
Your GitHub repo link and DagsHub/MLflow tracking server link.
Screenshots of your MLflow experiment runs (min. 2), registered model in Staging, and DVC pipeline diagram (dvc dag).
A short write-up (a few sentences per component) on your implementation.