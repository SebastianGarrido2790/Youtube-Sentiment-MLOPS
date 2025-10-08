### Project Environment Setup Guide

To establish a reproducible development environment using uv (a fast, Rust-based Python package manager), follow these steps. This setup assumes Python 3.10+ is installed globally. uv will manage virtual environments, dependencies, and locks for consistency across local, CI/CD, and AWS deployments.

#### Step 1: Install uv
Run the following in your terminal (macOS/Linux/Windows with pip):
```
pip install uv
```
Verify installation:
```
uv --version
```
Expected output: `uv x.y.z` (as of October 2025, latest is 0.4+).

#### Step 2: Initialize the Project
Navigate to your project root directory (where the folder structure resides). Initialize uv to create a `pyproject.toml` and `.venv`:
```
uv init --python 3.12
```
This generates:
- `pyproject.toml`: For dependencies and project metadata.
- `.venv`: Virtual environment (auto-activated in supported shells).

Activate the environment (if not auto-activated):
- macOS/Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

#### Step 3: Configure Project Metadata
Edit `pyproject.toml` to include basic project info. Add/replace the `[project]` section:
```toml
[project]
name = "youtube-sentiment-mlops"
version = "0.1.0"
description = "End-to-end MLOps pipeline for real-time YouTube sentiment analysis"
authors = [{name = "Your Name", email = "your.email@example.com"}]
dependencies = [
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "nltk>=3.8",
    "mlflow>=2.8",
    "dvc>=3.0",
    "docker>=7.0",  # For local containerization
    "boto3>=1.28",  # AWS integration
    "requests>=2.31",  # API calls (e.g., YouTube)
]
requires-python = ">=3.10"
```
This starts with core dependencies for data handling, ML, tracking, versioning, and cloud. We'll add more (e.g., transformers) iteratively.

Install dependencies and generate lockfile:
```
uv sync
```
Verify: `uv pip list` should show installed packages.

#### Step 4: Initialize Git and DVC
For version control:
```
git init
git add .
git commit -m "Initial project structure"
```
For data versioning (DVC):
```
uv run dvc init
dvc add data/raw  # After running download_dataset.py
git add .dvc
git commit -m "Initialize DVC"
```

#### Step 5: Test the Setup
Run the provided `src/data/download_dataset.py` to validate:
```
uv run python src/data/download_dataset.py
```
Expected output: "Data saved to data/raw/reddit.csv". Inspect the file:
```
head data/raw/reddit.csv
```
If issues arise (e.g., network), ensure no firewall blocks the GitHub URL.

#### Best Practices for Maintenance
- Always use `uv run <command>` for scripts to ensure environment isolation.
- Update deps: `uv add <package>` or `uv sync --upgrade`.
- For innovation: Experiment with uv's workspaces for sub-projects (e.g., extension JS).
- .env: Add secrets (e.g., `YOUTUBE_API_KEY=your_key`) and load via `python-dotenv` (add to deps if needed).

This setup ensures reliability (locked deps), scalability (uv's speed for large ML libs), maintainability (TOML config), and adaptability (easy dep swaps).