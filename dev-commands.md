# Development Commands

## Dependencies

### Update requirements.txt with top-level packages only
```bash
python -m pip list --format=freeze --not-required > requirements.txt
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