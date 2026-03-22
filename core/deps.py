"""
BuddyOS Dependency Injection

This module provides the dependency container for PydanticAI agents.
"""

from dataclasses import dataclass
import aiosqlite
import duckdb
from typing import Optional

from core.models import ModelInfo


@dataclass
class BuddyDeps:
    """
    Dependency container for BuddyOS agents.
    
    This dataclass holds all dependencies needed by the PydanticAI agent:
    - Database connections (SQLite and DuckDB)
    - Current model information
    - Additional context as needed
    """
    
    sqlite_conn: aiosqlite.Connection
    duckdb_conn: duckdb.DuckDBPyConnection
    current_model: ModelInfo
    conversation_id: Optional[str] = None