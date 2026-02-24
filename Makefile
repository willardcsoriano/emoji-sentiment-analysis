#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = emoji_sentiment_analysis
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Update requirements.txt and scrub Windows "build killers"
.PHONY: freeze
freeze:
	$(PYTHON_INTERPRETER) -m pip freeze > requirements.txt
	@$(PYTHON_INTERPRETER) -c "import re; \
	lines = open('requirements.txt').readlines(); \
	filtered = [l for l in lines if not re.search('pywinpty|win32|debugpy|jupyter|notebook|ipython|-e git', l)]; \
	open('requirements.txt', 'w').writelines(filtered)"
	@echo ">>> requirements.txt updated and scrubbed for Linux deployment."

## Delete all compiled Python files (Cross-platform version)
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	$(PYTHON_INTERPRETER) -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"

## Lint using ruff
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

#################################################################################
# CLOUD & DEPLOYMENT                                                            #
#################################################################################

## Deploy the application to Google Cloud Run
.PHONY: deploy
deploy:
	gcloud builds submit --config cloudbuild.yaml .

## View recent Cloud Run runtime logs
.PHONY: logs
logs:
	gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=emoji-sentiment-app" --limit 20

## Stream live logs from the cloud
.PHONY: stream-logs
stream-logs:
	gcloud alpha logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=emoji-sentiment-app"

## Train the model and update artifacts
.PHONY: train
train:
	$(PYTHON_INTERPRETER) src/emoji_sentiment_analysis/modeling/train_model.py
	@echo ">>> Artifacts updated in src/models/"

## Deploy WITH a fresh training run
.PHONY: deploy-full
deploy-full: train deploy

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Run the local FastAPI dev server
.PHONY: dev
dev:
	uvicorn app:app --reload

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) emoji_sentiment_analysis/dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)