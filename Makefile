# Blast Radius -- every operation has one documented command.
#
# `make help` lists them. A reviewer should never have to read this file to run the project, and
# `make demo` takes a clean checkout to a working demo in one step.

SHELL := /bin/bash
# Pinned, not `latest`: this engine's behaviour differs between 6.0.1 and 4.22.x in ways
# that change RESULTS rather than raising (see tests/test_regressions.py). 4.22.x is also
# what FalkorDB Cloud's free tier runs.
FALKOR_IMAGE := falkordb/falkordb:v4.22.1
PY := .venv/bin/python
PIP := .venv/bin/pip

# Which installer to use. Detected rather than assumed: `uv venv` creates a virtualenv with no
# pip inside it, so a hardcoded `$(PIP)` turns a working environment into a broken `make install`.
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
INSTALL := $(PIP) install --quiet
else
INSTALL := uv pip install --quiet --python $(PY)
endif
PORT ?= 8099
TENANT ?= acme
CONTAINER ?= blastradius-falkordb
CYTOSCAPE_VERSION ?= 3.30.2

.DEFAULT_GOAL := help
.PHONY: help venv install db db-stop vendor ingest verify cloud-verify ask run algo memory provenance \
        api test bench memory-proof doctor mcp mcp-verify demo clean fmt

help: ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# -- setup ----------------------------------------------------------------------------------

venv: ## create the virtualenv
	@test -d .venv || { \
	  if [ -n "$(UV)" ]; then uv venv .venv > /dev/null; else python3 -m venv .venv; fi; \
	}
	@# Deliberately no `pip install --upgrade pip` here. A `.venv` created by `uv venv` has no
	@# pip at all, and on some interpreters `ensurepip` is stripped too -- so insisting on pip
	@# fails on a virtualenv that is otherwise perfectly good. The detected installer above
	@# handles both cases.

install: venv ## install dependencies and the package
	@$(INSTALL) -r requirements.txt
	@$(INSTALL) -e .
	@echo "installed. next: make db && make ingest"

db: ## start FalkorDB (reuses the container if it exists)
	@if ! docker info > /dev/null 2>&1; then \
	  echo "Docker is not running. Start Docker Desktop, then re-run."; exit 1; fi
	@if docker ps -a --format '{{.Names}}' | grep -qx $(CONTAINER); then \
	  docker start $(CONTAINER) > /dev/null && echo "started $(CONTAINER)"; \
	else \
	  docker run -d --name $(CONTAINER) \
	    -p 127.0.0.1:6379:6379 -p 127.0.0.1:3100:3000 \
	    $(FALKOR_IMAGE) > /dev/null && echo "created $(CONTAINER)"; \
	fi
	@echo "  bolt-ish : 127.0.0.1:6379"
	@echo "  browser  : http://127.0.0.1:3100"

db-stop: ## stop FalkorDB (keeps the data)
	@docker stop $(CONTAINER) > /dev/null 2>&1 && echo "stopped" || echo "not running"

vendor: ## fetch the vendored browser asset (one network call, then offline forever)
	@mkdir -p web/vendor
	@curl -sSL -o web/vendor/cytoscape.min.js \
	  https://unpkg.com/cytoscape@$(CYTOSCAPE_VERSION)/dist/cytoscape.min.js
	@echo "vendored cytoscape $(CYTOSCAPE_VERSION) -- the demo now needs no CDN"

# -- data -----------------------------------------------------------------------------------

ingest: ## build both tenants' graphs
	@$(PY) -m blast_radius.cli ingest --tenant acme
	@$(PY) -m blast_radius.cli ingest --tenant globex

cloud-verify: ## run ingest + verify against FalkorDB Cloud (needs .env.cloud; never the default)
	@test -f .env.cloud || { echo "no .env.cloud -- put FALKOR_URL=falkor://user:pass@host:port in it"; exit 1; }
	@echo "target: $$(grep -o '@[^@]*$$' .env.cloud | head -1)"
	@set -a; . ./.env.cloud; set +a; \
	  $(PY) -m blast_radius.cli doctor --tenant acme && \
	  $(PY) -m blast_radius.cli ingest --tenant acme && \
	  $(PY) -m blast_radius.cli verify --tenant acme

ingest-globex-redis: ## load globex from a DIFFERENT real repo (needs ~19 API calls)
	@$(PY) -m blast_radius.cli ingest --tenant globex --repo redis/redis

ingest-offline: ## rebuild from cache only, zero network calls
	@$(PY) -m blast_radius.cli ingest --tenant acme --offline

verify: ## assert the graph can actually answer the demo
	@$(PY) -m blast_radius.cli verify --tenant acme,globex

provenance: ## what is real and what is synthetic (Rule 6, as a query)
	@$(PY) -m blast_radius.cli --tenant $(TENANT) provenance

# -- the demo -------------------------------------------------------------------------------

ask: ## all six questions, with the Cypher for each
	@$(PY) -m blast_radius.cli --tenant $(TENANT) ask all --cypher

run: ## the three-agent incident loop
	@$(PY) -m blast_radius.cli --tenant $(TENANT) run

algo: ## every graph algorithm
	@$(PY) -m blast_radius.cli --tenant $(TENANT) algo all

memory: ## the remembered episode timeline
	@$(PY) -m blast_radius.cli --tenant $(TENANT) memory

memory-proof: ## prove memory survives, in THREE separate OS processes
	@$(PY) -m blast_radius.proof --tenant $(TENANT)

api: ## serve the UI + API on 127.0.0.1 only
	@echo "  http://127.0.0.1:$(PORT)"
	@$(PY) -m uvicorn blast_radius.api:app --host 127.0.0.1 --port $(PORT)

mcp: ## run the MCP server (needs fastmcp)
	@$(PY) -m blast_radius.mcp_server

mcp-verify: ## prove the MCP transport end to end: spawn the server, speak the protocol
	@$(PY) tests/mcp_probe.py

# -- checks ---------------------------------------------------------------------------------

test: ## the regression suite
	@$(PY) -m pytest tests/ -q

bench: ## concurrency benchmark
	@$(PY) -m bench.concurrency --tenant $(TENANT) --workers 60 --seconds 10

bench-light: ## smaller benchmark for a loaded machine
	@$(PY) -m bench.concurrency --tenant $(TENANT) --workers 30 --seconds 8

doctor: ## diagnose the setup
	@$(PY) -m blast_radius.cli --tenant $(TENANT) doctor

# -- one-shot -------------------------------------------------------------------------------

demo: install db vendor ingest verify ## clean checkout -> working demo
	@echo ""
	@echo "ready. now run:  make api   then open http://127.0.0.1:$(PORT)"

clean: ## remove build artefacts (keeps .venv and the GitHub cache)
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache *.egg-info src/*.egg-info
	@echo "cleaned"
