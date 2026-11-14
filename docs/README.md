# Project guides

Start with these documents:

- [Project explainer](project-explainer.html) — visual architecture, agent roles,
  protocols, guardrails, evaluation, and deployment.
- [Faculty demo comparison](faculty-demo-concept-comparison.md) — maps the
  implementation to the course examples and evaluation concepts.
- [Minimum-requirements audit](minimum-requirements-compliance-audit.md) —
  evidence for each required capability.
- [Architecture alignment](architecture-alignment-verification.md) — verifies
  that the diagrams and implementation describe the same system.

Runnable instructions live in the repository [README](../README.md). The
canonical command-line demonstration is:

```powershell
.\.venv\Scripts\python.exe scripts\run_e2e.py
```

Generated evaluation evidence belongs in [`reports/`](../reports/); runtime
models, databases, uploads, and traces belong in the gitignored `artifacts/`
directory.
