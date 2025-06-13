.PHONY: docs
docs:
	@hugo -s docs server --bind 0.0.0.0 --port 1313 --disableFastRender --buildDrafts --buildFuture --buildExpired

# get hugo dependencies
mod:
	@hugo mod get ./...
