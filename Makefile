.PHONY: docs
docs:
	@hugo -s docs server --bind 0.0.0.0 --port 1313 --disableFastRender --buildDrafts --buildFuture --buildExpired --baseURL=http://0.0.0.0:1313/

# get hugo dependencies
mod:
	@hugo mod get ./...
