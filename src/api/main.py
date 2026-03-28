"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.utils.database import initialize_database
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Readmission Prevention Signal Agent API",
    description="AI-powered 30-day readmission risk scoring and care transition management",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    initialize_database()
    logger.info("Readmission Prevention Agent API started")


if __name__ == "__main__":
    import uvicorn
    from src.utils.config import config
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
