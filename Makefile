.PHONY: help clean setup test

ENV_NAME ?= AMARL
.DEFAULT: help
help:
	@echo "make clean"
	@echo "	clean all python build/compilation files and directories"
	@echo "make setup"
	@echo "	install dependencies in active python environment"
	@echo "make test"
	@echo " run all tests and coverage"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +
	rm --force .coverage
	rm --force --recursive .pytest_cache
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

amarl/_version.py:
	python meta.py $(shell git describe --tags --abbrev=0 --always)


third_party/.install.done:
	pip install --upgrade pip setuptools
	cd third_party/sequential_social_dilemma_games; pip install -e .
	touch third_party/.install.done

.install.done: amarl/_version.py third_party/.install.done
	pip install --upgrade pip setuptools
	pip install -e .
	touch .install.done

setup: .install.done

.install.test.done: amarl/_version.py third_party/.install.done
	pip install --upgrade pip setuptools
	pip install -e .[test]
	touch .install.test.done

test: .install.test.done
	pytest --verbose --color=yes --cov=amarl --cov-report term-missing tests
	pytest --verbose --color=yes third_party/sequential_social_dilemma_games/tests/test_envs.py
