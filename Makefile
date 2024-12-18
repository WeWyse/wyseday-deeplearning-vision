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

run-model:
	rm -rf runs/ & sleep 1 & python3 main.py

run-tensorboard:
	tensorboard --logdir=runs

# Run both model and tensorboard in parallel,
# CTRL+C will stop both at the same time
# By default the Tensorboard will be on:
#       http://localhost:6006/
run:
	bash -c 'trap "" INT; (make -j run-model & make -j run-tensorboard) & wait'
