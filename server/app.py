"""Run with one worker: uvicorn app:app --host 127.0.0.1 --port 8094."""
import hashlib
import hmac
import os
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException, Request, Depends
from protocol import DecisionRequest
from scheduler import Scheduler, Busy


def create_app(backend_factory=None):
    @asynccontextmanager
    async def lifespan(app):
        if not os.environ.get("JEV_MEMORY_TOKEN") or len(os.environ["JEV_MEMORY_TOKEN"]) < 24:
            raise RuntimeError("Set JEV_MEMORY_TOKEN to a random token of at least 24 characters")
        if backend_factory is None:
            from backend import MemoryBackend
            factory = MemoryBackend
        else:
            factory = backend_factory
        backend = await asyncio.to_thread(factory)
        app.state.scheduler = Scheduler(backend)
        app.state.scheduler.start()
        yield
        await app.state.scheduler.close()

    app = FastAPI(title="Jev Memory Decisions", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def payload_limit(request, call_next):
        from fastapi.responses import JSONResponse
        if request.method == "POST":
            try:
                if int(request.headers.get("content-length", "0")) > 65536:
                    return JSONResponse({"error": "Payload too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"error": "Invalid content length"}, status_code=400)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 65536:
                    return JSONResponse({"error": "Payload too large"}, status_code=413)
            request._body = bytes(body)
        return await call_next(request)

    def authorized(request: Request):
        token = os.environ.get("JEV_MEMORY_TOKEN", "")
        supplied = request.headers.get("authorization", "")
        if not token or not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            raise HTTPException(401, "Unauthorized")
        return hashlib.sha256(token.encode()).hexdigest()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(owner=Depends(authorized)):
        scheduler = getattr(app.state, "scheduler", None)
        if not scheduler or not scheduler.task or scheduler.task.done():
            raise HTTPException(503, "Not ready")
        return {"status": "ready", "model": scheduler.backend.identity}

    @app.get("/capabilities")
    def capabilities(owner=Depends(authorized)):
        return {"protocol": 1, "operations": ["retain", "relevance", "relationship"],
                "model": app.state.scheduler.backend.identity, "maxCandidates": 32,
                "maxSequenceTokens": 512, "prefixCache": False}

    @app.post("/v1/memory/decide")
    async def decide(body: DecisionRequest, owner=Depends(authorized)):
        try:
            return await app.state.scheduler.submit(body, owner)
        except ValueError:
            raise HTTPException(422, "Candidate or batch token limit exceeded")
        except Busy:
            raise HTTPException(503, "Inference unavailable or deadline exceeded")

    return app


app = create_app()
