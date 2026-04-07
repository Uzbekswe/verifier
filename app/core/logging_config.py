"""
Structured logging configuration.
Supports JSON (production) and text (development) formats.
"""
import logging
import sys


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        try:
            from pythonjsonlogger import jsonlogger

            class _JsonFormatter(jsonlogger.JsonFormatter):
                def add_fields(self, log_record, record, message_dict):
                    super().add_fields(log_record, record, message_dict)
                    from app.middleware.correlation_id import get_request_id
                    log_record["request_id"] = get_request_id()
                    log_record["logger"] = record.name
                    log_record["level"] = record.levelname

            handler.setFormatter(_JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        except ImportError:
            # Fallback to text if python-json-logger not installed yet
            log_format = "text"

    if log_format == "text":
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level.upper())
