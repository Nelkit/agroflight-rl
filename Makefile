# agroflight-rl -- shared setup so all three of us get an identical environment.
#
# Normal first-time flow:
#     make setup
#     make menagerie
#     make test
#
.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV        := .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
PYTEST      := $(VENV)/bin/pytest
MENAGERIE   := mujoco_menagerie
DRONE_DIR   := $(MENAGERIE)/skydio_x2
DRONE_XML   := $(DRONE_DIR)/x2.xml
MENAGERIE_URL := https://github.com/google-deepmind/mujoco_menagerie.git

# MuJoCo's interactive viewer must own the main thread on macOS, which needs
# `mjpython` instead of `python`. Pick automatically.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
VIEWER_PY := $(VENV)/bin/mjpython
else
VIEWER_PY := $(PY)
endif

export AGRO_DRONE_XML ?= $(CURDIR)/$(DRONE_XML)

.PHONY: help setup menagerie test viewer train evaluate tensorboard clean clean-runs

help: ## Show this help
	@echo "agroflight-rl -- available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Model path (AGRO_DRONE_XML): $(AGRO_DRONE_XML)"

setup: $(VENV)/.installed ## Create the venv and install the package (editable) + dev deps

$(VENV)/.installed: pyproject.toml requirements.txt
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[train,dev]"
	@touch $@
	@echo ""
	@echo "venv ready. Next: make menagerie"

menagerie: $(DRONE_XML) ## Sparse-clone the Skydio X2 model from mujoco_menagerie

$(DRONE_XML):
	@if [ ! -d $(MENAGERIE) ]; then \
		git clone --depth 1 --filter=blob:none --sparse $(MENAGERIE_URL) $(MENAGERIE); \
	fi
	cd $(MENAGERIE) && git sparse-checkout set skydio_x2
	@test -f $(DRONE_XML) && echo "Model ready at $(DRONE_XML)"

test: setup menagerie ## Run the pytest sanity suite
	$(PYTEST) -v

viewer: setup menagerie ## Open the live desktop viewer (baseline controller, not a trained policy)
	$(VIEWER_PY) scripts/viewer.py $(ARGS)

train: setup menagerie ## Run PPO training (UNTUNED -- see the TODOs in the training module)
	$(PY) -m agroflight.training.train_ppo_cleanrl

evaluate: setup menagerie ## Evaluate the baselines (and your checkpoint, once wired in)
	$(PY) -m agroflight.evaluation.evaluate_policy

tensorboard: setup ## Open TensorBoard on the training runs
	$(VENV)/bin/tensorboard --logdir runs

clean-runs: ## Delete training runs and checkpoints (keeps the venv and the model)
	rm -rf runs/ checkpoints/ wandb/

clean: ## Remove the venv, the menagerie clone, and all generated artefacts
	rm -rf $(VENV) $(MENAGERIE) runs/ checkpoints/ wandb/ \
		.pytest_cache/ src/*.egg-info build/ dist/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
