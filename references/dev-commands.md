# Development Commands

## Dependencies

### Update requirements.txt 
```bash
# 1. Generate the standard freeze
python -m pip freeze > requirements.txt
```

```bash
# 2. Scrub the "build-killers" (for Linux deployment env)
(Get-Content requirements.txt) | Where-Object { $_ -notmatch 'pywinpty|win32|debugpy|jupyter|notebook|ipython|-e git' } | Set-Content requirements.txt
```

### Install dependencies
```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Install package in editable mode (for development)
```bash
pip install -e .
```

### Show outdated packages
```bash
pip list --outdated
```

### Upgrade a specific package
```bash
pip install --upgrade package_name
```

## Virtual Environment

### Create virtual environment
```bash
python -m venv .venv
```

### Activate virtual environment
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### Deactivate virtual environment
```bash
deactivate
```

### Delete virtual environment
```bash
# PowerShell
Remove-Item -Recurse -Force .venv

# Bash/Linux
rm -rf .venv
```

## Code Quality

### Lint code
```bash
ruff format --check
ruff check
```

### Format code
```bash
ruff check --fix
ruff format
```

### Clean compiled files
```bash
# PowerShell
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse

# Bash/Linux
find . -type f -name "*.py[co]" -delete
find . -type d -name "__pycache__" -delete
```

### Type checking (if using mypy)
```bash
mypy .
```

### Run tests
```bash
# pytest
pytest

# with coverage
pytest --cov=emoji_sentiment_analysis

# verbose mode
pytest -v

# specific test file
pytest tests/test_dataset.py

# specific test function
pytest tests/test_dataset.py::test_function_name
```

## Git

### Check status
```bash
git status
```

### Stage all changes
```bash
git add .
```

### Commit with message
```bash
git commit -m "Your commit message"
```

### Push to remote
```bash
git push origin main
```

### Create new branch
```bash
git checkout -b feature/branch-name
```

### Switch branches
```bash
git checkout branch-name
```

### Pull latest changes
```bash
git pull origin main
```

### Undo last commit (keep changes)
```bash
git reset --soft HEAD~1
```

### Discard all local changes
```bash
git reset --hard HEAD
```

### View commit history
```bash
git log --oneline
```

## Project

### Generate dataset
```bash
python emoji_sentiment_analysis/dataset.py
```

### Run main script
```bash
python -m emoji_sentiment_analysis
```

### Start Jupyter notebook
```bash
jupyter notebook
```

### Start Jupyter lab
```bash
jupyter lab
```

## Python REPL & Debugging

### Start Python interactive shell
```bash
python
```

### Start IPython (enhanced shell)
```bash
ipython
```

### Run script with debugger on error
```bash
python -m pdb script.py
```

### Profile code performance
```bash
python -m cProfile -s cumtime script.py
```

## File Operations

### Find files by name
```bash
# PowerShell
Get-ChildItem -Recurse -Filter "*.py"

# Bash/Linux
find . -name "*.py"
```

### Search text in files
```bash
# PowerShell
Select-String -Path "*.py" -Pattern "search_term"

# Bash/Linux
grep -r "search_term" --include="*.py"
```

### Count lines of code
```bash
# PowerShell
(Get-ChildItem -Recurse -Filter "*.py" | Get-Content | Measure-Object -Line).Lines

# Bash/Linux
find . -name "*.py" -exec wc -l {} + | tail -1
```

## Useful Aliases

### Create PowerShell alias (add to $PROFILE)
```powershell
# Edit profile
notepad $PROFILE

# Add these lines:
function activate { .\.venv\Scripts\Activate.ps1 }
function freeze { python -m pip list --format=freeze --not-required > requirements.txt }
function run-tests { pytest -v }
```

### Reload PowerShell profile
```powershell
. $PROFILE
```

## Package Publishing (if needed)

### Build package
```bash
python -m build
```

### Upload to PyPI
```bash
twine upload dist/*
```

### Upload to Test PyPI
```bash
twine upload --repository testpypi dist/*
```

## Troubleshooting

### Clear pip cache
```bash
pip cache purge
```

### Reinstall all packages
```bash
pip freeze > temp_requirements.txt
pip uninstall -r temp_requirements.txt -y
pip install -r requirements.txt
```

### Fix SSL certificate errors
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name
```

### Check Python version
```bash
python --version
```

### Check pip version
```bash
pip --version
```

### Show installed package info
```bash
pip show package_name
```

## Cloud & Deployment (Google Cloud Platform)

### Deployment

```bash
# Submit a manual build to Cloud Build and deploy to Cloud Run
# This is the 'Nuclear Option' that pushes your local code live
gcloud builds submit --config cloudbuild.yaml .

```

### Log Analysis & Debugging

```bash
# 1. View logs of a specific build (if it fails during Step 0 or 1)
gcloud builds log [BUILD_ID]

# 2. View recent CRASH logs from the container (if it fails during Step 2)
# This is the "Smoking Gun" finder
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=emoji-sentiment-app" --limit 30

# 3. Stream LIVE logs from production
# Run this in a separate terminal to watch users interact with your app in real-time
gcloud alpha logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=emoji-sentiment-app"

```

### Service Management

```bash
# List all Cloud Run services and their public URLs
gcloud run services list

# Check the health/status of your specific service
gcloud run services describe emoji-sentiment-app --region us-central1

# Get the latest 5 revisions (useful for rollbacks)
gcloud run revisions list --service=emoji-sentiment-app --limit=5

```

### Cleanup & Resource Control

```bash
# Delete all untagged (old) images to save on storage costs
gcloud artifacts docker images list us-central1-docker.pkg.dev/hybrid-sentiment-analysis/emoji-repo/emoji-app --filter="NOT tags:*" --format="get(digest)" | ForEach-Object { gcloud artifacts docker images delete "us-central1-docker.pkg.dev/hybrid-sentiment-analysis/emoji-repo/emoji-app@$_" --quiet }

# Stop/Delete the service (if you want to pause the project)
gcloud run services delete emoji-sentiment-app --region us-central1

```

### Local Container Testing

```bash
# Build the image locally (requires Docker Desktop)
docker build -t emoji-app-local .

# Run the image locally to test paths before pushing to Cloud
# Maps your local port 8000 to the container's 8080
docker run -p 8000:8080 -e PORT=8080 emoji-app-local

```