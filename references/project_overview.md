### Project Overview: End-to-End MLOps Pipeline for Real-Time YouTube Sentiment Analysis

This project will develop a robust MLOps pipeline to collect YouTube video comments, perform real-time sentiment analysis, and display results via a Chrome extension. The pipeline will leverage the provided Reddit sentiment dataset as a proxy for initial training, transitioning to YouTube-specific data collection. We will emphasize clean, modular code with comprehensive documentation (e.g., docstrings, README updates) to facilitate learning and iteration.

The approach integrates modern ML tools for reproducibility (DVC for data versioning), experiment tracking (MLflow), containerization (Docker), automation (GitHub Actions for CI/CD), and cloud deployment (AWS for scalable inference). Package management will use uv for fast, dependency-resolved environments, ensuring reproducibility across local and cloud setups. To encourage innovation, we'll design modular components allowing easy swaps (e.g., advanced models like transformers) while prioritizing practicality through iterative testing.

The workflow follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, adapted for MLOps. This ensures a structured lifecycle from problem definition to production, with feedback loops for continuous improvement.

### CRISP-DM Workflow Adapted to the Project

| Phase              | Key Activities | Tools/Practices | Alignment with Requirements |
|--------------------|----------------|---------------|-----------------------------|
| **Business Understanding** | Define objectives: Real-time sentiment scoring (-1 to 1) for YouTube comments via extension. Identify success metrics (e.g., accuracy >85%, latency <2s). Scope: Proxy training on Reddit data; live YouTube ingestion. | Stakeholder alignment via README; requirements in .env. | **Reliability**: Clear KPIs prevent scope creep. **Scalability**: Design for comment volumes (e.g., 1000+/video). **Maintainability**: Document assumptions. **Adaptability**: Modular for future metrics (e.g., toxicity). |
| **Data Understanding** | Explore Reddit dataset (37k samples, imbalanced categories: ~60% neutral/positive). Analyze distributions, missing values (~0.3% in comments). Prototype YouTube data fetch via API. | Pandas in notebooks; DVC for versioning raw/interim data. Run `download_dataset.py` to ingest Reddit CSV. | **Reliability**: Validate data quality early. **Scalability**: Batch processing scripts. **Maintainability**: Scripts in `src/data`. **Adaptability**: Hooks for YouTube API integration. |
| **Data Preparation** | Clean text (e.g., remove URLs, normalize); engineer features (e.g., TF-IDF, embeddings); split train/test (80/20). Version processed data. Augment with synthetic YouTube-like comments for domain shift. | DVC pipelines; `make_dataset.py` and `feature_engineering.py` in `src`. uv for deps like scikit-learn, NLTK. | **Reliability**: Automated validation (e.g., schema checks). **Scalability**: Parallel processing with Dask if needed. **Maintainability**: Reproducible via DVC tracks. **Adaptability**: Configurable preprocessors for new data sources. |
| **Modeling** | Train baselines (e.g., Logistic Regression on TF-IDF, LSTM) on sentiment; log experiments (params, metrics), progress to advanced (e.g., BERT via Hugging Face for better sentiment nuance). Select the best model via cross-validation. Prototype inference endpoint. | MLflow for tracking; `train_model.py`/`predict_model.py` in `src/models`. Dockerize training env. | **Reliability**: A/B testing in MLflow. **Scalability**: GPU support via AWS EC2. **Maintainability**: Serialized models in `models/` with metadata. **Adaptability**: Experiment with Hugging Face for zero-shot models. |
| **Evaluation** | Assess on holdout/YouTube test set (precision, recall, F1). Monitor drift (e.g., sentiment shifts). Simulate real-time via extension mockups. | MLflow UI; custom metrics in `visualize.py`. GitHub Actions for automated eval on PRs. | **Reliability**: Threshold-based alerts. **Scalability**: Batch eval scripts. **Maintainability**: Reports in `reports/`. **Adaptability**: Feedback loop to retrain on new data. |
| **Deployment** | Containerize pipeline (Docker); deploy inference to AWS Lambda/EC2. Integrate Chrome extension (JS frontend calling API). CI/CD for builds/tests. | Dockerfiles; GitHub Actions workflows; AWS CDK for infra-as-code. Real-time: YouTube API polling + queue (SQS). | **Reliability**: Health checks, rollbacks. **Scalability**: Auto-scaling groups. **Maintainability**: Blue-green deploys. **Adaptability**: Serverless for extension updates. |

This workflow includes iterative loops (e.g., back to Data Preparation post-evaluation) and MLOps layers: version control (Git/DVC), automation (CI/CD), monitoring (MLflow + AWS CloudWatch).

### Expected Deliverables

- **Codebase**: Updated folder structure with scripts (e.g., `src/chrome_extension/` for JS bundle; `docker-compose.yml` for local stack).
- **Documentation**: Enhanced README.md (setup, run instructions); API docs (e.g., Swagger for inference); architecture diagram in `reports/figures/`.
- **Artifacts**: Versioned datasets (DVC); MLflow runs DB; Docker images; Deployed AWS resources (e.g., API Gateway endpoint); Functional Chrome extension (.crx file).
- **Reports**: Jupyter notebooks for exploration; evaluation summary (HTML/PDF) with metrics tables/ROC curves.
- **CI/CD Pipeline**: GitHub Actions YAML for linting, testing, building, deploying.

### Potential Starting Points for Next Steps

1. **Environment Setup**: Install uv (`pip install uv`), then `uv init` in project root; add deps to `pyproject.toml` (e.g., pandas, mlflow). Run `uv sync` and test `download_dataset.py`.
2. **Data Ingestion Prototype**: Extend `download_dataset.py` for YouTube API (via .env keys); commit to Git and track with DVC (`dvc init`, `dvc add data/raw`).
3. **Exploratory Notebook**: Create `notebooks/1.0-initial-exploration.ipynb` to load Reddit data, visualize distributions (e.g., category histogram), and baseline clean.
4. **Tool Onboarding**: Spin up local MLflow (`mlflow ui`); draft a simple GitHub Actions workflow for data download on push.
