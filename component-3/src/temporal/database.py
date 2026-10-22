"""
SQLite Temporal Persistence Store for Longitudinal Banana Severity Tracking (FR9).

(Compatibility Facade - logic moved to src.inference.database)
"""

from src.inference.database import SeverityDatabase, DEFAULT_DB_PATH

__all__ = ["SeverityDatabase", "DEFAULT_DB_PATH"]
