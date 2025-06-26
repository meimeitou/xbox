.PHONY: docs
docs:
	@echo "Hugo server starting on:"
	@ip route get 1.1.1.1 | awk '{printf "\033[31m  http://%s:1313/xbox\033[0m\n", $$7}' 2>/dev/null || printf "\033[31m  http://localhost:1313/xbox\033[0m\n"
	@printf "\033[31m  http://localhost:1313/xbox\033[0m\n"
	@hugo -s docs server --bind 0.0.0.0 --port 1313 --disableFastRender --buildDrafts --buildFuture --buildExpired --baseURL=http://0.0.0.0:1313/xbox

# get hugo dependencies
mod:
	@hugo mod get ./...