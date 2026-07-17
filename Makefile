# Introspection-leakage: common tasks. See README for the full picture.
# Two venvs: .venv (CPU analysis) and .venv-driver (the labkit GPU driver).
.PHONY: help setup setup-driver test analyze summary reanalyze collect clean

MODEL ?= qwen2.5-3b
GPU   ?= RTX_4090
EXP1  ?= experiments/exp1_epistemic_privilege
EXP2  ?= experiments/exp2_output_monitorability
EXP3  ?= experiments/exp3_induction_and_scale

help:
	@echo "make setup         # build .venv (CPU analysis: numpy/scipy/sklearn/matplotlib/torch)"
	@echo "make setup-driver  # build .venv-driver (the private labkit GPU driver)"
	@echo "make test          # run all CPU unit tests (exp1/exp2/exp3)"
	@echo "make analyze MODEL=qwen2.5-3b   # exp1 offline analysis suite for one model (needs the .pt)"
	@echo "make summary       # exp1 cross-model summaries (verify numbers, B-C significance, matched control)"
	@echo "make reanalyze     # exp2+exp3 reader analysis on a rented box (heavy nested-CV; gated)"
	@echo "make collect MODEL=qwen2.5-3b GPU=RTX_4090   # GPU collect via labkit (spends money)"
	@echo "make clean         # remove __pycache__"

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

setup-driver:
	python3 -m venv .venv-driver && .venv-driver/bin/pip install -r requirements-driver.txt

test:
	@for t in $(EXP1)/tests/test_injection_hook.py $(EXP1)/tests/test_calibrate_floor.py $(EXP1)/tests/test_sweep_probe.py $(EXP2)/tests/test_*.py $(EXP3)/tests/test_*.py; do \
		echo "=== $$t ==="; .venv/bin/python $$t || exit 1; done
	@echo "=== $(EXP1)/tests/test_driver_wiring.py (driver venv) ===" ; .venv-driver/bin/python $(EXP1)/tests/test_driver_wiring.py

analyze:
	.venv/bin/python $(EXP1)/analysis/run_all.py --model $(MODEL)

summary:
	.venv/bin/python $(EXP1)/analysis/verify_v1_numbers.py
	.venv/bin/python $(EXP1)/analysis/check_bmc_significance.py
	.venv/bin/python $(EXP1)/analysis/concept_matched_control.py

reanalyze:   # exp2+exp3 readers on a rented box -- heavy nested-CV, never a laptop; gated via experimentfactory
	.venv-driver/bin/python harness/run_reanalysis.py --gpu RTX4090

collect:
	.venv-driver/bin/python harness/run_labkit.py $(MODEL) --gpu $(GPU)

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
