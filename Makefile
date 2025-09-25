# Makefile for Insurance Claims Fraud Detection System
# Common development tasks and automation

.PHONY: help install install-dev test test-unit test-integration test-performance lint format type-check security-check
.PHONY: build run run-dev stop clean logs docker-build docker-push docker-clean
.PHONY: db-up db-down db-reset db-migrate db-seed deploy-staging deploy-prod rollback
.PHONY: docs docs-serve monitoring-up monitoring-down backup restore

# Variables
PROJECT_NAME := insurance-claims
PYTHON_VERSION := 3.10
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
DOCKER_REGISTRY := ghcr.io
DOCKER_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)
VERSION := $(shell git describe --tags --always --dirty)
COMMIT_SHA := $(shell git rev-parse HEAD)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

## Display help information
help:
	@echo "$(CYAN)Insurance Claims Fraud Detection System$(NC)"
	@echo "$(CYAN)========================================$(NC)"
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run mypy type checking"
	@echo "  security-check   Run security analysis"
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@echo "  build            Build development environment"
	@echo "  run              Start all services"
	@echo "  run-dev          Start development environment"
	@echo "  stop             Stop all services"
	@echo "  clean            Clean up containers and volumes"
	@echo "  logs             View service logs"
	@echo "  docker-build     Build production Docker image"
	@echo "  docker-push      Push Docker image to registry"
	@echo "  docker-clean     Clean up Docker resources"
	@echo ""
	@echo "$(GREEN)Database Commands:$(NC)"
	@echo "  db-up            Start database services"
	@echo "  db-down          Stop database services"
	@echo "  db-reset         Reset database with fresh data"
	@echo "  db-migrate       Run database migrations"
	@echo "  db-seed          Seed database with sample data"
	@echo ""
	@echo "$(GREEN)Deployment Commands:$(NC)"
	@echo "  deploy-staging   Deploy to staging environment"
	@echo "  deploy-prod      Deploy to production environment"
	@echo "  rollback         Rollback to previous version"
	@echo ""
	@echo "$(GREEN)Monitoring Commands:$(NC)"
	@echo "  monitoring-up    Start monitoring stack"
	@echo "  monitoring-down  Stop monitoring stack"
	@echo ""
	@echo "$(GREEN)Utility Commands:$(NC)"
	@echo "  docs             Generate documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo "  backup           Backup database and models"
	@echo "  restore          Restore from backup"

# Development Environment Setup
$(VENV):
	@echo "$(YELLOW)Creating virtual environment...$(NC)"
	python$(PYTHON_VERSION) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

## Install production dependencies
install: $(VENV)
	@echo "$(YELLOW)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

## Install development dependencies
install-dev: $(VENV)
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,performance,monitoring]"
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

# Code Quality and Testing
## Run all tests
test: install-dev
	@echo "$(YELLOW)Running test suite...$(NC)"
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ All tests completed$(NC)"

## Run unit tests only
test-unit: install-dev
	@echo "$(YELLOW)Running unit tests...$(NC)"
	$(PYTEST) tests/unit/ -v --cov=src --cov-report=term-missing
	@echo "$(GREEN)✓ Unit tests completed$(NC)"

## Run integration tests only
test-integration: install-dev db-up
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(PYTEST) tests/integration/ -v
	@echo "$(GREEN)✓ Integration tests completed$(NC)"

## Run performance tests
test-performance: install-dev
	@echo "$(YELLOW)Running performance tests...$(NC)"
	$(PYTEST) tests/performance/ -v --benchmark-only
	@echo "$(GREEN)✓ Performance tests completed$(NC)"

## Run linting checks
lint: install-dev
	@echo "$(YELLOW)Running linting checks...$(NC)"
	$(VENV)/bin/flake8 src tests
	$(VENV)/bin/black --check src tests
	$(VENV)/bin/isort --check-only src tests
	@echo "$(GREEN)✓ Linting checks passed$(NC)"

## Format code with black and isort
format: install-dev
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(VENV)/bin/black src tests
	$(VENV)/bin/isort src tests
	@echo "$(GREEN)✓ Code formatted$(NC)"

## Run mypy type checking
type-check: install-dev
	@echo "$(YELLOW)Running type checking...$(NC)"
	$(VENV)/bin/mypy src
	@echo "$(GREEN)✓ Type checking passed$(NC)"

## Run security analysis
security-check: install-dev
	@echo "$(YELLOW)Running security checks...$(NC)"
	$(VENV)/bin/safety check
	$(VENV)/bin/bandit -r src
	@echo "$(GREEN)✓ Security checks passed$(NC)"

# Docker Commands
## Build development environment
build:
	@echo "$(YELLOW)Building development environment...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Development environment built$(NC)"

## Start all services
run:
	@echo "$(YELLOW)Starting all services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ All services started$(NC)"
	@echo "$(CYAN)Access the application at: http://localhost:8000$(NC)"
	@echo "$(CYAN)Grafana dashboard: http://localhost:3000 (admin/admin)$(NC)"
	@echo "$(CYAN)Kibana: http://localhost:5601$(NC)"

## Start development environment with logs
run-dev:
	@echo "$(YELLOW)Starting development environment...$(NC)"
	docker-compose up

## Stop all services
stop:
	@echo "$(YELLOW)Stopping all services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ All services stopped$(NC)"

## Clean up containers and volumes
clean:
	@echo "$(YELLOW)Cleaning up containers and volumes...$(NC)"
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

## View service logs
logs:
	docker-compose logs -f

## Build production Docker image
docker-build:
	@echo "$(YELLOW)Building production Docker image...$(NC)"
	docker build \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT_SHA=$(COMMIT_SHA) \
		--build-arg BUILD_DATE=$(shell date -u +"%Y-%m-%dT%H:%M:%SZ") \
		-t $(DOCKER_IMAGE):$(VERSION) \
		-t $(DOCKER_IMAGE):latest \
		.
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE):$(VERSION)$(NC)"

## Push Docker image to registry
docker-push: docker-build
	@echo "$(YELLOW)Pushing Docker image to registry...$(NC)"
	docker push $(DOCKER_IMAGE):$(VERSION)
	docker push $(DOCKER_IMAGE):latest
	@echo "$(GREEN)✓ Docker image pushed$(NC)"

## Clean up Docker resources
docker-clean:
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker image prune -f
	docker container prune -f
	docker volume prune -f
	docker network prune -f
	@echo "$(GREEN)✓ Docker cleanup completed$(NC)"

# Database Commands
## Start database services
db-up:
	@echo "$(YELLOW)Starting database services...$(NC)"
	docker-compose up -d postgres redis elasticsearch
	@echo "$(GREEN)✓ Database services started$(NC)"

## Stop database services
db-down:
	@echo "$(YELLOW)Stopping database services...$(NC)"
	docker-compose stop postgres redis elasticsearch
	@echo "$(GREEN)✓ Database services stopped$(NC)"

## Reset database with fresh data
db-reset: db-down
	@echo "$(YELLOW)Resetting database...$(NC)"
	docker-compose rm -f postgres redis elasticsearch
	docker volume rm -f insurance-claims_postgres_data insurance-claims_redis_data insurance-claims_elasticsearch_data
	$(MAKE) db-up
	sleep 10
	$(MAKE) db-seed
	@echo "$(GREEN)✓ Database reset completed$(NC)"

## Run database migrations
db-migrate: install-dev db-up
	@echo "$(YELLOW)Running database migrations...$(NC)"
	$(PYTHON) -m src.database.migrate
	@echo "$(GREEN)✓ Database migrations completed$(NC)"

## Seed database with sample data
db-seed: install-dev db-up
	@echo "$(YELLOW)Seeding database with sample data...$(NC)"
	$(PYTHON) -m src.database.seed
	@echo "$(GREEN)✓ Database seeded$(NC)"

# Deployment Commands
## Deploy to staging environment
deploy-staging: test docker-build
	@echo "$(YELLOW)Deploying to staging...$(NC)"
	# Add your staging deployment commands here
	@echo "$(GREEN)✓ Deployed to staging$(NC)"

## Deploy to production environment
deploy-prod: test docker-build
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@echo "$(RED)⚠️  This will deploy to production. Are you sure? [y/N]$(NC)"
	@read -r CONFIRM && [ "$$CONFIRM" = "y" ] || (echo "Deployment cancelled" && exit 1)
	# Add your production deployment commands here
	@echo "$(GREEN)✓ Deployed to production$(NC)"

## Rollback to previous version
rollback:
	@echo "$(YELLOW)Rolling back to previous version...$(NC)"
	# Add your rollback commands here
	@echo "$(GREEN)✓ Rollback completed$(NC)"

# Monitoring Commands
## Start monitoring stack
monitoring-up:
	@echo "$(YELLOW)Starting monitoring stack...$(NC)"
	docker-compose up -d prometheus grafana jaeger
	@echo "$(GREEN)✓ Monitoring stack started$(NC)"
	@echo "$(CYAN)Prometheus: http://localhost:9090$(NC)"
	@echo "$(CYAN)Grafana: http://localhost:3000$(NC)"
	@echo "$(CYAN)Jaeger: http://localhost:16686$(NC)"

## Stop monitoring stack
monitoring-down:
	@echo "$(YELLOW)Stopping monitoring stack...$(NC)"
	docker-compose stop prometheus grafana jaeger
	@echo "$(GREEN)✓ Monitoring stack stopped$(NC)"

# Utility Commands
## Generate documentation
docs: install-dev
	@echo "$(YELLOW)Generating documentation...$(NC)"
	$(VENV)/bin/sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)✓ Documentation generated$(NC)"

## Serve documentation locally
docs-serve: docs
	@echo "$(YELLOW)Serving documentation...$(NC)"
	@echo "$(CYAN)Documentation available at: http://localhost:8080$(NC)"
	cd docs/_build/html && python -m http.server 8080

## Backup database and models
backup:
	@echo "$(YELLOW)Creating backup...$(NC)"
	mkdir -p backups
	docker-compose exec postgres pg_dump -U claims_user claims_db > backups/db_$(shell date +%Y%m%d_%H%M%S).sql
	tar -czf backups/models_$(shell date +%Y%m%d_%H%M%S).tar.gz models/
	@echo "$(GREEN)✓ Backup completed$(NC)"

## Restore from backup
restore:
	@echo "$(YELLOW)Restoring from backup...$(NC)"
	@echo "$(RED)⚠️  This will overwrite current data. Are you sure? [y/N]$(NC)"
	@read -r CONFIRM && [ "$$CONFIRM" = "y" ] || (echo "Restore cancelled" && exit 1)
	# Add restore commands here
	@echo "$(GREEN)✓ Restore completed$(NC)"

# CI/CD Helpers
.PHONY: ci-setup ci-test ci-build ci-deploy
ci-setup:
	pip install --upgrade pip
	pip install -e ".[dev,performance,monitoring]"

ci-test: ci-setup
	pytest tests/ --cov=src --cov-report=xml --junitxml=test-results.xml

ci-build:
	docker build -t $(PROJECT_NAME):ci .

ci-deploy: ci-test ci-build
	# CI deployment logic here
	@echo "CI deployment completed"