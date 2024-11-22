from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

from vox_box import __version__
from vox_box.server.routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="vox-box",
    lifespan=lifespan,
    response_model_exclude_unset=True,
    version=__version__,
)
app.include_router(router)


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "API endpoint not found"},
    )


@app.get("/")
async def read_root():
    return {"message": "Welcome"}
