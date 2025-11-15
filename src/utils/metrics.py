from dataclasses import dataclass, field
from typing import List
from datetime import datetime
import statistics

@dataclass
class MetricsCollector:
    #Collects and aggregates processing metrics

    total_processed: int = 0
    total_duplicates: int = 0
    total_unique: int = 0

    processing_times: List[float] = field(default_factory=list)
    input_lengths: List[float] = field(default_factory=list)
    summary_lengths: List[float] = field(default_factory=list)

    start_time: datetime = field(default_factory=datetime.now)

    def record_deduplication(
            self,
            total: int,
            unique: int,
            duplicates: int,
    ):
        self.total_processed += total
        self.total_unique += unique
        self.total_duplicates += duplicates

    def record_summarization(
            self,
            count: int,
            avg_input_length: float,
            avg_summary_length: float,
    ):
        self.input_lengths.append(avg_input_length)
        self.summary_lengths.append(avg_summary_length)

    def record_pipeline_run(
            self,
            total_articles: int,
            processing_time_ms: float,
    ):
        
        self.processing_times.append(processing_time_ms)

    def get_summary(self) -> dict:
        uptime = (datetime.now() - self.start_time).total_seconds

        return{
            'total_processed': self.total_processed,
            'total_unique': self.total_unique,
            'total_duplicates': self.total_duplicates,
            'duplication_rate': (
                self.total_duplicates/ self.total_processed
                if self.total_processed > 0 else 0
            ),
            'avg_processing_time_ms': (
                statistics.mean(self.processing_times)
                if self.processing_times else 0
            ),
            'median_processing_time_ms': (
                statistics.median(self.processing_times)
                if self.processing_times else 0
            ),
            'p95_processing_time_ms':(
                statistics.quantiles(self.processing_times, n=20)[18]
                if len(self.processing_times) > 20 else 0
            ),
            'avg_input_length':(
                statistics.mean(self.input_lengths)
                if self.input_lengths > 0 else 0
            ),
            'avg_summary_length':(
                statistics.mean(self.summary_lengths)
                if self.summary_lengths > 0 else 0
            ),
            'compression_ratio':(
                statistics.mean(self.input_lengths) / statistics.mean(self.summary_lengths)
                if self.summary_lengths and statistics.mean(self.summary_lengths) > 0 else 0
            ),
            'throughput_per_second':(
                self.total_processed/uptime if uptime > 0 else 0
            ),
            'uptime_seconds': uptime
        }
    
def reset(self):
    self.total_processed = 0
    self.total_duplicates = 0
    self.total_unique = 0
    self.processing_times.clear()
    self.input_lengths.clear()
    self.summary_lengths.clear()
    self.start_time = datetime.now()