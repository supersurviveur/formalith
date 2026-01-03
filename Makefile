COVERAGE=cargo llvm-cov --branch --doc
coverage:
	@$(COVERAGE) | bat --wrap=never

coverage-html:
	@$(COVERAGE) --open

coverage-diff:
	@$(COVERAGE) --text --color=always
