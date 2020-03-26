.PHONY: help meta clean setup test

ENV_NAME ?= AMARL
.DEFAULT: help
help:
	@echo "make meta"
	@echo " update version number and meta data"
	@echo "make clean"
	@echo "	clean all python build/compilation files and directories"
	@echo "make setup"
	@echo "	create virtual environment and install dependencies"
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

env_done:
	conda env update -n ${ENV_NAME} -f external/sequential_social_dilemma_games/environment.yml
	cd external/sequential_social_dilemma_games/; python setup.py develop
	cd external/ray/python/ray/rllib/; python setup-rllib-dev.py --yes
	cd external/sequential_social_dilemma_games; python -m pytest
	conda env update -n ${ENV_NAME} -f environment.yml
	pip install -e .
	touch env_done

setup: env_done

env_test_done: env_done
	pip install -e .[test]
	touch env_test_done

test: env_test_done
	pytest --verbose --color=yes --cov=amarl --cov-report term-missing tests