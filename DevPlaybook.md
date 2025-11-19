```markdown
# GitHub Actions CI/CD Guide — Python projects (with Web-Scale Optimization)

This document describes a recommended GitHub Actions CI/CD setup for Python services and libraries, plus advanced optimization techniques for web-scale systems (frontend, business layer, backend) and best practices for Test‑Driven Development (TDD), refactoring, and agile software craftsmanship.

Goals
- Prevent regressions with formatters, linters, type checks, and tests on PRs.
- Ensure reproducible builds and cached dependencies for speed.
- Automate releases and controlled deployments with environment protection.
- Surface security and dependency issues early.
- Measure and enforce performance budgets and optimizations as part of CI/CD.
- Encourage craftsmanship: TDD, continuous refactoring, pair review, and learning culture.

Key principles
- Fail fast for style/type issues; run expensive checks after fast gate checks pass.
- Measure before optimizing — use profiling and metrics to identify real hotspots.
- Prefer simple, maintainable solutions; optimize where it matters using data.
- Make performance and scalability part of PR review and CI (profiling, perf tests).
- Practice TDD where practical; integrate refactoring and tech‑debt pay‑down into regular cadences.
- Keep feedback loops short (fast tests, pre-commit hooks, small PRs).

Required repo settings
- Branch protection rules for main: require CI checks (lint/type/tests/perf where applicable), require PR reviews, require up-to-date branch before merge.
- Repository Secrets for publish tokens, cloud credentials, and environment-specific secrets.
- Environments with required reviewers for staging/production.
- Performance baselines and test-data artifacts stored or reproducible.
- CODEOWNERS, PR/ISSUE templates, CONTRIBUTING.md and SECURITY.md.

Recommended CI checks
- Formatting: black, isort
- Linting: ruff/flake8
- Type checking: mypy/pyright
- Unit tests: pytest with coverage; fast unit tests for PR feedback
- Integration/smoke tests for critical flows
- Dependency/supply-chain scans: Dependabot + CodeQL/Snyk
- Performance/regression tests and profiling jobs
- Mutation testing (scheduled or on major PRs) to improve test quality

TDD, Refactoring & Agile Software Craftsmanship

1) Test-Driven Development (TDD)
- Philosophy: Red → Green → Refactor. Write a failing test that describes desired behavior, implement minimal code to pass, then refactor.
- Practical TDD rules:
  - Prefer unit tests for fast-feedback behavior validation. Keep unit tests <100ms where possible.
  - Write tests that assert behavior, not implementation details. Use public APIs for tests.
  - Keep tests deterministic and independent; avoid shared global state.
  - Maintain a clear test pyramid: many fast unit tests, fewer integration tests, even fewer E2E tests.
- CI enforcement:
  - Require tests to run and pass on PRs.
  - Require coverage for new code (e.g., ensure new modules have tests).
  - Integrate mutation testing (mutmut or similar) in scheduled workflows or on major releases to increase confidence.
- Tooling & practices:
  - Use pytest with parametrization, fixtures, and markers (unit/integration/smoke).
  - Use test doubles (mocks/fakes) at unit level; use contract tests for real integrations.
  - Pair or mob on test-writing for complex flows to spread knowledge.

2) Refactoring
- Continuous refactoring is part of daily work, not a separate phase.
- Small, safe refactors:
  - Keep PRs small and focused: one refactor per PR where possible.
  - Use automated formatting (black/isort) and lint auto-fixes (ruff) to remove noise.
  - Add unit tests before refactoring if behavior is not already covered.
- Detection & automation:
  - Use static analysis (mypy, ruff), complexity checks (radon), duplication detection, and code-quality tools (SonarCloud/CodeClimate) in CI.
  - Add a scheduled "code health" job that reports complexity, duplicated code, and trendlines.
- Policies & governance:
  - Boy Scout Rule: leave the codebase cleaner than you found it.
  - Track tech debt as first-class backlog items; schedule periodic pay-down and review.
  - Use CODEOWNERS and small module ownership to manage large refactors.
- Safe rollout:
  - Run full test suite and perf baselines after refactors touching hot paths.
  - Use feature flags when refactors cross public API boundaries.

3) Agile Software Craftsmanship
- Team practices:
  - Definition of Done (DoD): code compiles, tests pass, type checks, lint passed, docs/README updated, and performance considerations noted for critical paths.
  - Pair programming & mobbing on high-risk or knowledge-transfer tasks.
  - Regular code reviews focused on correctness, readability, and design — not just formatting.
  - Encourage coding katas and lunch-and-learns to build shared skillsets (TDD workshops, design-pattern sessions).
- Process & delivery:
  - Trunk-based development or short-lived feature branches with frequent merges.
  - Use feature flags and canary releases for incremental rollouts.
  - Keep user stories small and implementable within a sprint; include acceptance tests as part of the ticket.
  - Maintain a blameless postmortem culture and continuous improvement via retrospectives.
- Documentation & mentoring:
  - Maintain onboarding docs, architectural overviews, and runbooks.
  - Rotate on-call and review duties to spread operational knowledge.

CI/CD Integration for TDD & Refactoring

1) Pre-commit & local parity
- Ensure developers run the same linters/formatters locally:
  - .pre-commit-config.yaml runs black, isort, ruff, detect-secrets, mypy, and basic tests.
- Enforce pre-commit on CI (CI should re-run linters and fail fast if pre-commit would have rejected).

2) PR checks & gating
- Required jobs for PRs:
  - Formatting check (black --check, isort --check-only)
  - Linting (ruff/flake8)
  - Type checks (mypy/pyright)
  - Unit tests (fast subset)
  - Full tests & coverage (dependent job)
- Labels / templates:
  - PR template should include checkboxes for TDD/refactor practices:
    - Tests added? (Y/N)
    - Behavior covered by tests? (unit/integration)
    - Complexity considerations documented?
    - Rollout/flagging plan for risky changes?

3) Mutation testing (improve test quality)
- Run mutation testing in scheduled workflow or nightly for main branch (tools: mutmut, cosmic-ray).
- Use results to identify weak tests and add coverage where necessary.
- Do not block trivial PRs with mutation testing, but triage high-impact mutation escapes.

4) Performance & refactor safety
- For refactors touching hot paths, require:
  - Microbenchmarks or profiling evidence that change is safe.
  - Perf tests in CI/staging with baseline comparison (scheduled or on-demand).
- Upload profiler flamegraphs to PR for reviewer inspection when relevant.

Example PR checklist (add to .github/PULL_REQUEST_TEMPLATE.md)
- [ ] Small, focused change (one logical purpose)
- [ ] Tests: unit/integration added/updated
- [ ] TDD followed where feasible (link to failing test if used)
- [ ] CI: linters, type checks, tests passed
- [ ] Performance impact noted (N/A if not applicable)
- [ ] Documentation and changelog updated
- [ ] Code owner review requested

Sample GitHub Actions additions

A) Fast "TDD gate" job for PRs (conceptual snippet)
```yaml
name: TDD-Gate
on: [pull_request]
jobs:
  tdd-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          pip install -r requirements-dev.txt
      - name: Run format/lint/type
        run: |
          black --check .
          isort --check-only .
          ruff .
          mypy src
      - name: Run unit tests (fast)
        run: |
          pytest tests/unit -q --maxfail=1 -k "not integration"
```

B) Scheduled mutation testing job (nightly/main)
```yaml
name: Mutation-Test
on:
  schedule:
    - cron: '0 3 * * *' # nightly
jobs:
  mutation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps and mutmut
        run: |
          pip install -r requirements-dev.txt
          pip install mutmut
      - name: Run mutation tests (mutmut)
        run: |
          mutmut run --paths-to-mutate src
      - name: Upload mutmut report
        uses: actions/upload-artifact@v4
        with:
          name: mutmut-report
          path: .mutmut-cache
```

Best practices summary (practical rules)
- TDD: write failing test first where reasonable; keep tests fast & deterministic.
- Refactor: one refactor per PR, small commits, add tests before modifying behavior; run complexity/duplication checks.
- Agile craftsmanship: DoD, pair programming, trunk-based development, feature flags, keep PRs small, mentor juniors.
- CI: enforce tests, type checks, linters; schedule heavier quality jobs (mutation, perf, code-health) nightly.
- Observability: require telemetry for new code paths; add dashboards and alerting for critical SLIs.

Where to put these items in your repo
- CI workflows: .github/workflows/ (add tdd-gate.yml, mutation.yml, perf.yml)
- PR template: .github/PULL_REQUEST_TEMPLATE.md (add TDD/refactor checklist)
- Contributing & style: CONTRIBUTING.md, CODE_OF_CONDUCT.md
- Runbooks & SLOs: docs/SLO.md, docs/runbooks/
```