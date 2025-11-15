from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from datetime import datetime
import logging

from ..pipeline.processor import NewsProcessor
from ..utils.logging_config import setup_logging
from ..config import settings

setup_logging()
logger = logging.getLogger(__name__)

#Initialize FastAPI app
app = FastAPI(
    title = "NewsScope API",
    description = "Production-grade NLP pipeline for event summarization",
    version = "1.0.0",
)

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#Global processor instance
processor = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global processor
    logger.info("Initializing NewsProcessor...")
    processor = NewsProcessor()
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("API shudown")


#Pydantic models
class ArticleInput(BaseModel):
    """Input article schema"""
    id: Optional[str] = None
    title: str = Field(..., min_length = 1, max_length = 500) #... is required pydantic helps setting minlength and maxlength
    content: str = Field(..., min_length = 10, max_length = 10000)
    url: HttpUrl
    published_at: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "title": "AI Breakthrough in Natural Language Processing",
                "content": "Researchers have acjeived a major breakthrough...",
                "url": "https://example.com/article",
                "published_at": "2025-01-15T10:30:00Z"
            }
        }

class BatchProcessRequest(BaseModel):
    """Batch processing request"""
    articles: List[ArticleInput]
    deduplicate: bool = True
    summarize: bool = True


class ProcessedArticleResponse(BaseModel):
    """Processed Article Response"""
    id: str
    title: str
    summary: str
    url: str
    published_at: str
    is_duplicate: bool
    processing_time_ms: float

class BatchProcessResponse(BaseModel):
    """Batch Processsing Response"""
    total_articles: int
    unique_articles: int
    duplicate_articles: int
    processing_time_ms: float
    articles: List[ProcessedArticleResponse]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool

class MetricsResponse(BaseModel):
    """Metrics Response"""
    total_processed: int
    total_duplicates: int
    avg_processing_time_ms: float
    uptime_seconds: float

#API Routes
# API Routes
@app.get("/", response_model=dict)  # ❌ Added root endpoint
async def root():
    """Root endpoint."""
    return {
        "service": "NewsScope",
        "version": "1.0.0",
        "status": "operational",
    }


@app.get("/health", response_model = HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status = "healthy",
        timestamp = datetime.utcnow().isoformat(),
        models_loaded = processor is not None,
    )

@app.post("/api/v1/summarize", response_model = ProcessedArticleResponse)
async def summarize_article(article: ArticleInput):
    """
    Summarize a single article.
    
    Args:
        article: Input article
        
    Returns:
        Processed article with summary
    """
    try:
        article_dict = article.dict()
        if not article_dict.get('id'):
            # Generate ID from URL if not provided
            article_dict['id'] = str(hash(article_dict['url']))
        result = processor.process_single_article(article_dict)

        return ProcessedArticleResponse(
            id = result.id,
            title = result.title,
            summary = result.summary,
            url = str(result.url),
            published_at = result.published_at or '',
            is_duplicate = result.is_duplicate,
            processing_time_ms = result.processing_time_ms
        )
    except Exception as e:
        logger.error(f"Error processing article: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))
    
@app.post("/api/v1/batch", response_model = BatchProcessResponse)
async def batch_process(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
):
    """
    Process multiple articles in batch.
    
    Args:
        request: Batch processing request
        background_tasks: FastAPI background tasks
        
    Returns:
        Batch processing results
    """

    try:
        start_time = datetime.now()

        #Convert to dict
        articles = [a.dict() for a in request.articles]

        #process batch
        results = processor.process_articles(
            articles,
            deduplicate = request.deduplicate,
            summarize = request.summarize,
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        #Separate unique and duplicates
        unique = [r for r in results if not r.is_duplicate]
        duplicates = [r for r in results if r.is_duplicate]

        article_responses = [
            ProcessedArticleResponse(
                id = r.id or str(hash(r.url)),
                title = r.title,
                summary = r.summary,
                url = str(r.url),
                published_at = r.published_at or '',
                is_duplicate = r.is_duplicate,
                processing_time_ms = r.processing_time_ms
            )
            for r in results
        ]

        return BatchProcessResponse(
            total_articles = len(request.articles),
            unique_articles = len(unique),
            duplicate_articles = len(duplicates),
            processing_time_ms = processing_time,
            articles = article_responses
        )
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))
    
@app.get("/api/v1/metrics", response_model = MetricsResponse)
async def get_metrics():
    """Get Processing metrics"""
    try:
        metrics = processor.get_metrics_summary()

        return MetricsResponse(
            total_processed = metrics.get('total_processed', 0),
            total_duplicates = metrics.get("total_duplicates", 0),
            avg_processing_time_ms = metrics.get('avg_processing_time_ms', 0),
            uptime_seconds = metrics.get('uptime_seconds', 0),
        )
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host = settings.API_HOST,
        port = settings.API_PORT,
        reload = True,
    )

    









    

