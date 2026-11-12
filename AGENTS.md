# BMED365-2027 – instructions for AI coding agents

The working rules for this repository are in [CLAUDE.md](./CLAUDE.md). Read that file before making any change, and follow it regardless of which agent or model you are (Cursor, Grok, Codex, Claude Code, or other).

The rules are kept in one file so they do not drift apart. Do not copy them here; edit `CLAUDE.md` instead.

The most important ones, in short:

- The repo is public. Never commit secrets, exam material, unreleased solutions or non-shareable data.
- Work on `dev`, never directly on `main`. Commit and push only when the user asks.
- Do not commit notebooks that were only run as a test, and do not commit changes to a notebook's `kernelspec` metadata.
- Use `uv` (`uv sync`, `uv run`, `uv add`). Never `pip install` into `.venv`.
