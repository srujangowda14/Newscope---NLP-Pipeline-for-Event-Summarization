"""Semantic deduplication using sentence embeddings and cosine similarity."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Set
from dataclasses import dataclass
import logging
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Article data structure"""
    id: str
    title: str
    content: str
    url: str
    published_at: str
    embedding: np.ndarray = None

class SemanticDeduplicator:
    """Deduplicates articles using semantic similarity"""

    def __init__(self, model_name: str = None, threshold: float = None):
        """
        Initialize deduplicator.
        
        Args:
            model_name: Sentence transformer model name
            threshold: Cosine similarity threshold (0-1)
        """

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.threshold = threshold or settings.SIMILARITY_THRESHOLD

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

        logger.info(f"Deduplicator initialized with threshold = {self.threshold}")

    def compute_embeddings(
            self, articles: List[Article],
            batch_size: int = None,
    ) -> List[Article]:
        """
        Compute embeddings for articles.
        
        Args:
            articles: List of Article objects
            batch_size: Batch size for encoding
            
        Returns:
            Articles with computed embeddings
        """

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        texts = [
            f"{article.title}. {article.content}"
            for article in articles
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar = False,
            convert_to_numpy=True,
            normalize_embeddings = True, #L2 normalization for cosine similarity so that
            #dot product would be just A.B cos(A,B)=∥A∥∥B∥A⋅B​ but here it is just |A||B| because
            #vectors are already L2normalized divided by their magnitude i.e |A| or |B|
        )

        #zip just makes it one dictionary sort of
        for article, embedding in zip(articles, embeddings):
            article.embedding = embedding

        return articles
    
    def deduplicate(
            self, articles: List[Article],
            compute_embeddings: bool = True,
    ) -> Tuple[List[Article], Set[str]]:
        """
        Remove duplicate articles based on semantic similarity.
        
        Args:
            articles: List of articles to deduplicate
            compute_embeddings: Whether to compute embeddings
            
        Returns:
            Tuple of (unique_articles, duplicate_ids)
        """
         
        if not articles:
            return [], set()
        
        if compute_embeddings:
            articles = self.compute_embeddings(articles)

        unique_articles = []
        duplicate_ids = set()

        #Stack embeddings for vectorized operations
        embeddings = np.vstack([a.embedding for a in articles]) #stacks the embeddings on top of one another

        for i, article in enumerate(articles): #enumerate pairs up index and elements
            if article.id in duplicate_ids:
                continue

            #Compute similarity with all remaining articles
            similarities = np.dot(embeddings[i+1:], article.embedding)

            duplicate_mask = similarities >= self.threshold
            duplicate_indices = np.where(duplicate_mask)[0] + i + 1

            for idx in duplicate_indices:
                duplicate_ids.add(articles[idx].id)

            unique_articles.append(article)

        reduction_pct = len(duplicate_ids) / len(articles) * 100

        logger.info(
            f"Deduplication complete: {len(unique_articles)} unique, "
            f"{len(duplicate_ids)} duplicates ({reduction_pct:.1f}% reduction)"

        )

        return unique_articles, duplicate_ids
    
    def find_similar_clusters(
            self,
            articles: List[Article],
            min_cluster_size: int = 2,
    ) -> List[List[Article]]:
        """
        Group similar articles into clusters.
        
        Args:
            articles: List of articles
            min_cluster_size: Minimum articles per cluster
            
        Returns:
            List of article clusters
        """
        if not articles:
            return []
        
        articles = self.compute_embeddings(articles)
        embeddings = np.vstack([a.embedding for a in articles])

        #Compute similarity matrix

        similarity_matrix = np.dot(embeddings, embeddings.T) #dot pro b/w matrix and its transpose
        #M[0][0] similarity woth itself M[0][1] similarity with 0 and 1 embedding

        #find clusters using greedy approach
        clusters = []
        visited = set()

        for i, article in enumerate(articles):
            if i in visited:
                continue

            #find all similar articles
            similar_indices = np.where(similarity_matrix[i] >= self.threshold)
            #compares every element in row i with threshold

            if len(similar_indices) >= min_cluster_size:
                cluster = [articles[idx] for idx in similar_indices]
                clusters.append(cluster)
                visited.update(similar_indices)

            logger.info(f"Found {len(clusters)} article clusters")

            return clusters
    
    def compute_similarity(
            self,
            article1: Article,
            article2: Article,
    ) -> float:
        """
        Compute cosine similarity between two articles.
        
        Args:
            article1: First article
            article2: Second article
            
        Returns:
            Similarity score (0-1)
        """
        if article1.embedding is None:
            self.compute_embeddings([article1])

        if article2.embedding is None:
            self.compute_embeddings([article2])

        similarity = np.dot(article1.embedding, article2.embedding)

        return float(similarity)
        
    
        









        


        


         
        


        

