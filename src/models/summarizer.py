import torch
from transformers import(
    AuthTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
)
from typing import List, Union
import logging
from ..config import settings

logger = logging.getLogger(__name__)

class Summarizer:

    def __init__(self, model_type: str = None, device: str = None):
        self.model_type =model_type or settings.SUMMARIZER_MODEL
        self.device = device or settings.DEVICE

        if not torch.cuda.is_available() and self.device == "cuda":
            logger.warning("CUDA not avaialble, falling back to CPU")
            self.device = "cpu"
        
        self._load_model()
        logger.info(f"Loaded {self.model_type} model on {self.device}")
    
    def _load_model(self):
        if self.model_type == "bart":
            model_name = settings.BART_MODEL_NAME
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        elif self.model_type == "t5":
            model_name = settings.T5_MODEL_NAME
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.tokenizer = AuthTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, texts, max_length=None, min_length=None, batch_size=None) -> Union[str, List[str]]:
        """
        Generate summaries for input text(s).
    
        Args:
           texts: Single text or list of texts
           max_length: Maximum summary length
           min_length: Minimum summary length
           batch_size: Batch size for processing
        
        Returns:
           Summary or list of summaries
        """
        single_input = isinstance(texts, str)

        if single_input:
            texts = [texts]

        max_length = max_length or settings.MAX_SUMMARY_LENGTH
        min_length = min_length or settings.MIN_SUMMARY_LENGTH
        batch_size = batch_size or settings.BATCH_SIZE

        summaries = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_summaries = self._summarize_batch(
                batch, max_length, min_length
            )
            summaries.extend(batch_summaries)

        return summaries[0] if single_input else summaries

        

    def _summarize_batch(self, texts, max_length, min_length) -> List[str]:
        if self.model_type == "t5":
            texts = [f"summarize: {text}" for text in texts]

        input = self.tokenizer(
            texts,
            max_length = settings.MAX_INPUT_LENGTH,
            truncation = True,
            padding = True,
            return_tensors = "pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        #Generate Summaries
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length = max_length,
                min_length = min_length,
                num_beams=4,
                length_penalty=2.0,
                earlt_stopping=True
            )
        
        summaries = self.tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens = True
        )

        return summaries
    

