"""
Server entry point for IndicScriptureQA — OpenEnv compatible.

Exposes the FastAPI app and a `main()` callable for the `server` script.
"""

import uvicorn

from main import app  # noqa: F401 — re-export for openenv discovery


def main() -> None:
    """Entry point used by `[project.scripts] server`."""
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
