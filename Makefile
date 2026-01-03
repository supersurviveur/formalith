COVERAGE=cargo llvm-cov --branch
coverage:
	@$(COVERAGE) | bat --wrap=never

coverage-html:
	@$(COVERAGE) --open

coverage-diff:
	@$(COVERAGE) --text --color=always
