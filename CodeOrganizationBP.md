<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Code organization best practices for python

Well-organized Python code keeps projects maintainable, testable, and scalable as they grow. Aim for a consistent project layout, modular boundaries, clear naming, and standards-based packaging, testing, and tooling across the lifecycle.[^1][^2][^3]

### Project layout

- Prefer a src/ layout to avoid accidental imports from the working directory and to mirror how packages are installed in production.[^4][^5]
- A pragmatic baseline: top-level files for pyproject.toml, README, LICENSE; src/<package>/ for code; tests/ for test modules; docs/ for documentation; scripts/ for CLI or dev utilities.[^5][^1]
- Keep configuration in code-friendly formats and centralize metadata in pyproject.toml using PEP 621, which standardizes project metadata and dependencies.[^6][^3]


### Packages and modules

- Keep module names short, lowercase, and avoid special symbols; use CapWords for classes and snake_case for functions and variables per PEP 8.[^2][^1]
- Group related functionality into packages and subpackages; split large modules rather than accumulating long files that mix concerns.[^7][^1]
- For multi-distribution namespace packages, use PEP 420 implicit namespaces (no __init__.py) when targeting modern Python, and avoid mixing pkg_resources-style namespaces with others due to incompatibilities.[^8][^9]


### Imports and boundaries

- Design for explicit, stable import paths; avoid deep circular imports by factoring shared types/utilities into lower-level modules.[^1][^2]
- Use __all__ sparingly to curate public APIs; keep internal modules private via leading underscores and document supported import surfaces in package __init__ when appropriate.[^2][^1]
- Adopt a layered structure (e.g., domain, services, adapters) to separate pure logic from I/O and frameworks, enabling easier testing and swaps.[^10][^1]


### Configuration and metadata

- Centralize build, dependencies, and tool config in pyproject.toml (PEP 621), which modern tools (PDM, Hatch, Poetry) consume; keep Python version and classifiers explicit.[^3][^6]
- Prefer environment variables or config files for runtime settings; avoid hardcoding secrets in code or VCS.[^7][^1]
- Pin application runtime dependencies appropriately and separate dev/test extras in optional dependency groups in pyproject.toml.[^6][^3]


### Testing strategy

- Place tests/ at the project root, mirroring the src/ package structure; keep unit tests isolated from network and filesystem unless using fakes.[^5][^1]
- Name tests consistently, e.g., test_<module>.py, and use pytest conventions for discovery; target public APIs and critical invariants first.[^1][^5]
- Maintain fixtures/factories under tests/ to avoid duplication and encourage deterministic, hermetic tests.[^5][^1]


### Documentation

- Use module, class, and function docstrings to describe purpose, arguments, returns, and exceptions; keep CLI usage in README and help text.[^11][^1]
- Maintain a top-level README with how to install, develop, test, and release; link to docs/ for extended guides and architecture notes.[^7][^1]
- Keep changelogs and release notes versioned alongside code for traceability.[^1][^7]


### Code style and quality

- Follow PEP 8 for formatting, naming, and whitespace; enforce with linters/formatters to keep reviews focused on logic.[^12][^2]
- Use type hints to clarify interfaces and enable static checks; pair with mypy/pyright and keep runtime fast by minimizing heavy typing at hot paths.[^12][^2]
- Add logging with consistent structure and levels; prefer dataclasses/enums to make state explicit and reduce ad-hoc dicts.[^10][^12]


### Dependency and environment management

- Standardize on a modern toolchain via pyproject.toml; PDM/Poetry/Hatch support PEP 517/518 builds and lockfiles for reproducibility.[^3][^6]
- Keep local development isolated in virtual environments; document minimum Python version and constraints to prevent drift.[^6][^3]
- Separate runtime dependencies from dev/test tooling, and keep extras grouped for optional features.[^3][^6]


### Version control hygiene

- Commit only source, config, and lockfiles; exclude caches, virtualenvs, and build artifacts with a clean .gitignore.[^7][^1]
- Keep small, cohesive commits with descriptive messages; enforce pre-commit hooks for formatting, linting, and basic static checks.[^12][^7]
- Maintain branch protection and CI to run tests and checks on every change to preserve quality gates.[^12][^7]


### Example minimal layout

- project-root/ with pyproject.toml, README.md, LICENSE and .gitignore at the top-level.[^5][^1]
- src/my_package/ for modules and subpackages; tests/ mirroring src/ packages; scripts/ for task entry points.[^4][^5]
- Use PEP 420 only when intentionally spanning a namespace across distributions; otherwise include __init__.py for regular packages.[^13][^9]


### When to use namespaces (PEP 420)

- Use implicit namespaces to split a logical package across multiple distributions or repos without __init__.py, available since Python 3.3+.[^9][^13]
- Do not mix pkg_resources-style namespaces with native or pkgutil styles; the packaging guide warns they are incompatible across the same namespace.[^14][^8]
- If transitioning, ensure all parts of the namespace adopt one method to avoid path recomputation and discovery issues.[^8][^9]
<span style="display:none">[^15][^16][^17][^18][^19][^20]</span>

<div align="center">⁂</div>

[^1]: https://docs.python-guide.org/writing/structure/

[^2]: https://peps.python.org/pep-0008/

[^3]: https://peps.python.org/pep-0621/

[^4]: https://discuss.python.org/t/python-project-structure/36119

[^5]: https://stackoverflow.com/questions/193161/what-is-the-best-project-structure-for-a-python-application

[^6]: https://rudeigerc.dev/posts/structuring-a-python-project/

[^7]: https://dev.to/olgabraginskaya/12-steps-to-organize-and-maintain-your-python-codebase-for-beginners-18bb

[^8]: https://github.com/zopefoundation/meta/issues/194

[^9]: https://peps.python.org/pep-0420/

[^10]: https://realpython.com/python-script-structure/

[^11]: https://stackoverflow.com/questions/1849311/how-should-i-organize-python-source-code

[^12]: https://realpython.com/python-code-quality/

[^13]: https://www.oreilly.com/library/view/expert-python-programming/9781789808896/f1f1803e-37c5-40a0-b3d2-f382da2ea76f.xhtml

[^14]: https://projects.gentoo.org/python/guide/concept.html

[^15]: https://dagster.io/blog/python-project-best-practices

[^16]: https://www.reddit.com/r/Python/comments/vtevk9/organize_python_code_like_a_pro/

[^17]: https://www.reddit.com/r/Python/comments/18qkivr/what_is_the_optimal_structure_for_a_python_project/

[^18]: https://peps.python.org/pep-0001/

[^19]: https://peps.python.org/pep-0402/

[^20]: https://www.appacademy.io/blog/python-coding-best-practices/

