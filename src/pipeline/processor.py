from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from ..models.summarizer import Summarizer
from .deduplicator import SemanticDeduplicator, Article
from ..utils.metrics import MetricsCollector


logger = logging.getLogger(__name__)

@dataclass
class ProcessedArticle:
    """Processed Article with summary"""

    id: str
    title: str
    content: str
    summary: str
    url: str
    published_at: str
    is_duplicate: bool = False
    processing_time_ms: float = 0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
class NewsProcessor:
    """Main pipeline for processing news articles"""

    def __init__(
        self,
        summarizer: Optional[Summarizer] = None,
        deduplicator: Optional[SemanticDeduplicator] = None,
        enable_metrics: bool = True
    ):
        """
        Initialize the news processor.
        
        Args:
            summarizer: Summarizer instance
            deduplicator: Deduplicator instance
            enable_metrics: Whether to collect metrics
        """

        self.summarizer = summarizer or Summarizer()
        self.deduplicator = deduplicator or SemanticDeduplicator()
        self.metrics = MetricsCollector() if enable_metrics else None

        logger.info("NewsProcessor initialized")


    def process_single_article(
            self,
            article: Dict,
    ) -> ProcessedArticle:
        """
        Process a single article.
    
        Args:
        article: Article dictionary
        
        Returns:
        Processed article
        """

        results = self.process_articles(
            [article],
            deduplicate = False,
            summarize = True,
        )

        return results[0]
    
    def process_articles(
            self,
            articles: List[Dict],
            deduplicate: bool = True,
            summarize: bool = True
    ) -> List[ProcessedArticle]:
        """
    Process a batch of articles through the complete pipeline.
    
    Args:
        articles: List of raw article dictionaries
        deduplicate: Whether to deduplicate
        summarize: Whether to generate summaries
        
    Returns:
        List of processed articles
    """
        start_time = datetime.now()
        logger.info(f"Processing {len(articles)} articles")

        #Step 1: Convert to Article objects

        article_objects = [
            Article(
                id = a.get('id', str(hash(a['url']))),
                title = a['title'],
                content = a['content'],
                url = a['url'],
                published_at = a.get('published_at', '')
            )
            for a in articles
        ]

        #Step 2: Dedpulication phase
        duplicate_ids = set()
        if deduplicate:
            logger.info("Starting deduplication phase")
            article_objects, duplicate_ids = self.deduplicator.deduplicate(
                article_objects
            )

            if self.metrics:
                self.metrics.record_deduplication(
                    total = len(articles),
                    unique = len(article_objects),
                    duplicates = len(duplicate_ids)
                )
        
        #Step 3: Summarization phase
        summaries = []
        if summarize and article_objects:
            logger.info(f"Starting summarization for {len(article_objects)} articles")

            contents = [a.content for a in article_objects]
            summaries = self.summarizer.summarize(contents)

            if self.metrics:
                self.metrics.record_summarization(
                    count = len(summaries),
                    avg_input_length = sum(len(c) for c in contents) / len(contents),
                    avg_summary_length = sum(len(s) for s in summaries)/ len(summaries),
                )

        else:
            summaries = [''] * len(article_objects)

        #Step 4: Create processed articles
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        processed = []

        for article, summary in zip(article_objects, summaries):
            processed.append(
                ProcessedArticle(
                    id = article.id,
                    title = article.title,
                    content = article.content,
                    summary = summary,
                    url = article.url,
                    published_at = article.published_at,
                    is_duplicate = False,
                    processing_time_ms = processing_time/len(article_objects)
                )
            )

        #Step 5: Mark duplicates
        for article_dict in articles:
            article_id = article_dict.get('id', str(hash(article_dict['url'])))
            if article_id in duplicate_ids:
                processed.append(
                    ProcessedArticle(
                        id = article_id,
                        title = article_dict['title'],
                        content = article_dict['content'],
                        summary = '',
                        url = article_dict['url'],
                        published_at = article_dict.get('published_at', ''),
                        is_duplicate = True,
                        processing_time_ms = 0
                    )
                )
            
        logger.info(
            f"Processing complete: {len(processed)} articles "
            f"({len(article_objects)} unique) in {processing_time:.2f}ms"
        )

        #Step 6: Record metrics
        if self.metrics:
            self.metrics.record_pipeline_run(
                total_articles = len(articles),
                processing_time_ms = processing_time
            )

        return processed
    
    def get_metrics_summary(self) -> Dict:
        """Get Processing metrics summary"""
        if not self.metrics:
            return {}
        
        return self.metrics.get_summary()




        
        
        



        

