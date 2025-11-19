## Contributing to Advanced RAG (Milvus)

Thank you for your interest in contributing! This project follows a few simple principles:

- **Tests first** where practical (TDD-style: Red → Green → Refactor).
- **Small, focused PRs**: one logical change per PR.
- **Quality gates**: lint, type checks, tests, and security scans must pass.

### How to develop

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests:

```bash
pytest -q
```

4. (Optional) Install and run pre-commit:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Pull Requests

- Ensure tests pass locally before opening a PR.
- Update or add tests for any behavior changes.
- Update README/docs when you change public behavior or APIs.
- Use the PR template and fill in the checklist.


