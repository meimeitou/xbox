.PHONY: docs
docs:
	@echo "Hugo server starting on:"
	@LOCAL_IP=$$(ipconfig getifaddr en0 2>/dev/null || ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $$2}' | head -n1); \
	if [ -n "$$LOCAL_IP" ]; then \
		printf "\033[31m  http://%s:1313/xbox\033[0m\n" "$$LOCAL_IP"; \
	fi
	@printf "\033[31m  http://localhost:1313/xbox\033[0m\n"
	@hugo -s docs server --bind 0.0.0.0 --port 1313 --disableFastRender --buildDrafts --buildFuture --buildExpired --baseURL=http://0.0.0.0:1313/xbox

# get hugo dependencies
mod:
	@hugo mod get ./...