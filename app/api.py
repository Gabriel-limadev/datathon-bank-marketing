import logging
import time

import psutil
from fastapi import FastAPI, HTTPException, Request

from app.schemas import Customer
from app.services import recommend_service, train_service

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bank Marketing - Contextual Bandit API",
    description="API para recomendação de canal de contato usando Thompson Sampling.",
    version="1.0.0",
)


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    """Registra tempo de resposta e uso de CPU e memória da API."""
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    cpu = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory().percent

    logger.info(
        f"{request.client.host} | "
        f"{request.method} {request.url.path} | "
        f"Status={response.status_code} | "
        f"Tempo={duration:.3f}s | "
        f"CPU={cpu}% | "
        f"RAM={memory}%"
    )

    return response


@app.get("/")
def root():
    return {
        "title": "Bank Marketing - Contextual Bandit API",
        "description": "API para recomendação de canal de contato usando Thompson Sampling.",
        "version": "1.0.0",
    }


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "online",
        "version": app.version,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
    }


@app.post("/train", tags=["Training"])
def train():
    """Retreina os modelos do Thompson Sampling."""

    try:
        return train_service()
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/recommend", tags=["Recommendation"])
def recommend(customer: Customer):
    """Retorna o canal de contato recomendado pelo Thompson Sampling."""
    try:
        return recommend_service(customer.model_dump())
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
