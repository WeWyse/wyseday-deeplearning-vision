# Makefile for wyseday-deeplearning-vision project

.PHONY: clean venv install-requirements install format

#######################
# PREPARE ENVIRONMENT #
#######################

clean:
	rm -rf .venv

venv:
	python3 -m venv .venv

install-requirements:
	./.venv/bin/pip install -r requirements.txt

install: clean venv install-requirements

###############
# FORMAT CODE #
###############

format:
	black ./*.py & black ./*/*.py & isort ./*.py & isort ./*/*.py 


#####################
# LAUNCH THE SCRIPT #
#####################

run:
	python3 main.py
