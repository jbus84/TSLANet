SHELL = /bin/bash
.DEFAULT_GOAL := help

.PHONY: bootstrap
bootstrap: ## Bootstrap local repo checkout
	@echo Installing Python deps into Poetry managed venv
ifeq (, $(shell which poetry))
	@echo No \'Poetry\' in \$$PATH. Please install
else

	@poetry env use python3.10
	@poetry install
	@rm -f .venv
	@ln -s $$(poetry env info --path) .venv
endif
