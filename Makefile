<<<<<<< HEAD
autoformat:
	black src/ checkpoints/ models/
	isort --atomic src/ checkpoints/ models/
	docformatter --in-place --recursive src checkpoints models

lint:
	isort -c src/ checkpoints/ models/
	black src/ checkpoints/ models/ --check
	flake8 src/ checkpoints/ models/

dev:
	pip install -r requirements-dev.txt
=======
autoformat:
	black src/ checkpoints/ models/
	isort --atomic src/ checkpoints/ models/
	docformatter --in-place --recursive src checkpoints models

lint:
	isort -c src/ checkpoints/ models/
	black src/ checkpoints/ models/ --check
	flake8 src/ checkpoints/ models/

dev:
	pip install -r requirements-dev.txt
>>>>>>> 2a5624c (Add project fiels)
