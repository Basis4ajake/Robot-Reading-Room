from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import chat, health, libraries, models, sources
from .state import AppState


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_app(data_dir: str | None = None, force_dummy: bool | None = None) -> FastAPI:
    resolved_data_dir = data_dir or os.environ.get("LKL_DATA_DIR", "./data/libraries")
    resolved_force_dummy = force_dummy if force_dummy is not None else _env_bool("LKL_FORCE_DUMMY", False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.lkl = AppState(data_dir=resolved_data_dir, force_dummy=resolved_force_dummy)
        yield
        app.state.lkl.close()

    app = FastAPI(title="Robot Reading Room API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(libraries.router)
    app.include_router(sources.router)
    app.include_router(chat.router)

    return app
