# Reproduction Makefile for Section 7 of
# Barnfield, Burke, Friedlander, Hoheisel (2026).
#
# Usage:
#   make figures   # regenerate all four Section 7 figures
#   make test      # run smoke test
#   make all       # test + figures
#   make clean     # remove generated figures + build artifacts
#   make lock      # regenerate uv.lock and requirements.txt
#
# Requires:
#   - uv (https://docs.astral.sh/uv/)  OR  a Python 3.11 environment with
#     the packages in requirements.txt already installed.
#
# Override RUN= to use plain python instead of `uv run`:
#   make figures RUN=python

RUN ?= uv run python

FIGURES := figures/problem_data.pdf \
           figures/trajectory_a.pdf \
           figures/trajectory_b.pdf \
           figures/trajectory_c.pdf \
           figures/trajectory_d.pdf \
           figures/trajectory_e.pdf \
           figures/trajectory_f.pdf \
           figures/trajectory_legend_row1.pdf \
           figures/trajectory_legend_row2.pdf \
           figures/regularization_path_rel.pdf

.PHONY: all figures test clean lock sync help

help:
	@echo "Targets:"
	@echo "  figures   regenerate all four Section 7 figures into figures/"
	@echo "  test      run the smoke test"
	@echo "  all       run test then regenerate figures"
	@echo "  clean     remove generated figures + __pycache__"
	@echo "  lock      regenerate uv.lock and requirements.txt"
	@echo "  sync      uv sync (install pinned environment)"

all: test figures

figures: $(FIGURES)

figures/problem_data.pdf: scripts/fig_7_1_problem_data.py
	$(RUN) scripts/fig_7_1_problem_data.py

TRAJECTORY_PANELS := figures/trajectory_a.pdf \
                     figures/trajectory_b.pdf \
                     figures/trajectory_c.pdf \
                     figures/trajectory_d.pdf \
                     figures/trajectory_e.pdf \
                     figures/trajectory_f.pdf \
                     figures/trajectory_legend_row1.pdf \
                     figures/trajectory_legend_row2.pdf

$(TRAJECTORY_PANELS): scripts/fig_7_iteration_trajectories.py
	$(RUN) scripts/fig_7_iteration_trajectories.py

figures/regularization_path_rel.pdf: scripts/fig_7_4_regularization_path.py
	$(RUN) scripts/fig_7_4_regularization_path.py

test:
	$(RUN) -m pytest

sync:
	uv sync

lock:
	uv lock
	uv pip compile pyproject.toml -o requirements.txt

clean:
	rm -f figures/*.pdf
	rm -rf src/selfscaling/__pycache__ scripts/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
