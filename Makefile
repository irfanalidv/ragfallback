.PHONY: test test-unit test-integration test-all

test: test-unit

test-unit:
	python -m pytest tests/unit -q --tb=short

test-integration:
	python -m pytest tests/integration -m integration -q --tb=short

test-all:
	python -m pytest tests -q --tb=short
