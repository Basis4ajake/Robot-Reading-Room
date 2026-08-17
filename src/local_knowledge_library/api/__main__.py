from __future__ import annotations

import os

import uvicorn

from .app import create_app

app = create_app()


def main() -> None:
    host = os.environ.get("LKL_API_HOST", "127.0.0.1")
    port = int(os.environ.get("LKL_API_PORT", "8000"))
    uvicorn.run("local_knowledge_library.api.__main__:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
