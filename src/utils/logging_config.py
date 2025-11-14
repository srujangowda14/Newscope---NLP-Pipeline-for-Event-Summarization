import logging
import sys
from pythonjsonlogger import jsonlogger
from datetime import datetime
from ..config import settings

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

        log_record['timestamp'] = datetime.utcnow().isoformat()

        log_record['service'] = 'newsscope'

        log_record['level'] = record.levelname

        log_record['source'] = f"{record.filename}:{record.lineno}"

        log_record['function'] = record.funcName

def setup_logging():

    #Get root loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    #Remove existing handlers to avoid duplicate
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)

    if settings.LOG_FORMAT == 'json':
        formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(service)s %(name)s %(message)s'
    )

    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    root_logger.info("Logging configured successfully")

class LogContext:

    def __init__(self, **kwargs):
        self.context = kwargs #holds extra fields you want to add
        self.old_factory = None

    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)





