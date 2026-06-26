.PHONY: install install-all test bench demo dataset dash clean

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,kuzu,dashboard]"

test:
	python -m pytest -q

bench:
	python -m agent_memory_lab.cli bench --backends all --episodes 30 --out results.json

demo:
	python -m agent_memory_lab.cli demo

dataset:
	python -m agent_memory_lab.cli dataset --name longmemeval

dash:
	streamlit run dashboard/app.py

clean:
	rm -f results.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
