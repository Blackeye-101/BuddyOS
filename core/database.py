"""
BuddyOS Hybrid Persistence Layer

This module manages two distinct storage systems:
1. SQLite (via aiosqlite) - Conversations and messages (transactional)
2. DuckDB - User facts and knowledge (analytical)
"""

import asyncio
import aiosqlite
import duckdb
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager


@dataclass
class Conversation:
    """Conversation metadata."""
    id: str
    created_at: str
    updated_at: str
    title: str
    model_id: str


@dataclass
class Message:
    """Individual message in a conversation."""
    id: str
    conversation_id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    model_id: str
    token_count: int
    created_at: str


@dataclass
class UserFact:
    """User knowledge fact stored in DuckDB."""
    id: str
    category: str
    fact_text: str
    confidence: float  # 0.0 to 1.0
    last_seen: str
    is_active: bool = True


class BuddyDatabase:
    """
    Hybrid database manager for BuddyOS.
    
    Manages:
    - SQLite (async): Conversations and messages
    - DuckDB (sync, wrapped in asyncio.to_thread): User facts
    """
    
    def __init__(self, sqlite_path: str = "data/buddy.db", duckdb_path: str = "data/knowledge.duckdb"):
        """
        Initialize database connections.
        
        Args:
            sqlite_path: Path to SQLite database file
            duckdb_path: Path to DuckDB database file
        """
        self.sqlite_path = sqlite_path
        self.duckdb_path = duckdb_path
        self._sqlite_conn: Optional[aiosqlite.Connection] = None
        self._duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None
    
    async def initialize(self):
        """Initialize both databases and create schemas."""
        await self._init_sqlite()
        await self._init_duckdb()
    
    async def _init_sqlite(self):
        """Initialize SQLite database with schema."""
        self._sqlite_conn = await aiosqlite.connect(self.sqlite_path)
        # Enable foreign keys
        await self._sqlite_conn.execute("PRAGMA foreign_keys = ON")
        
        # Create conversations table
        await self._sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT,
                model_id TEXT NOT NULL
            )
        """)
        
        # Create index for conversations
        await self._sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_created 
            ON conversations(created_at DESC)
        """)
        
        # Create messages table
        await self._sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model_id TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) 
                    REFERENCES conversations(id) 
                    ON DELETE CASCADE
            )
        """)
        
        # Create index for messages
        await self._sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id, created_at)
        """)
        
        await self._sqlite_conn.commit()
    
    async def _init_duckdb(self):
        """Initialize DuckDB database with schema (wrapped in async)."""
        def _create_duckdb_schema():
            self._duckdb_conn = duckdb.connect(self.duckdb_path)
            
            # Create facts table
            self._duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id VARCHAR PRIMARY KEY,
                    category VARCHAR NOT NULL,
                    fact_text VARCHAR NOT NULL,
                    confidence DOUBLE NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Create indexes
            self._duckdb_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_active 
                ON facts(is_active, last_seen DESC)
            """)
            
            self._duckdb_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_category 
                ON facts(category, is_active)
            """)
        
        # Wrap synchronous DuckDB in async
        await asyncio.to_thread(_create_duckdb_schema)
    
    # ============================================
    # Conversation Methods (SQLite - Async)
    # ============================================
    
    async def create_conversation(self, title: str, model_id: str) -> str:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            model_id: Model being used
            
        Returns:
            conversation_id: UUID of created conversation
        """
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        await self._sqlite_conn.execute(
            """
            INSERT INTO conversations (id, created_at, updated_at, title, model_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, now, now, title, model_id)
        )
        await self._sqlite_conn.commit()
        
        return conversation_id
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve conversation by ID.
        
        Args:
            conversation_id: UUID of conversation
            
        Returns:
            Conversation object or None if not found
        """
        async with self._sqlite_conn.execute(
            "SELECT id, created_at, updated_at, title, model_id FROM conversations WHERE id = ?",
            (conversation_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Conversation(
                    id=row[0],
                    created_at=row[1],
                    updated_at=row[2],
                    title=row[3],
                    model_id=row[4]
                )
        return None
    
    async def list_conversations(self, limit: int = 50) -> List[Conversation]:
        """
        List recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of Conversation objects, newest first
        """
        conversations = []
        async with self._sqlite_conn.execute(
            """
            SELECT id, created_at, updated_at, title, model_id 
            FROM conversations 
            ORDER BY updated_at DESC 
            LIMIT ?
            """,
            (limit,)
        ) as cursor:
            async for row in cursor:
                conversations.append(Conversation(
                    id=row[0],
                    created_at=row[1],
                    updated_at=row[2],
                    title=row[3],
                    model_id=row[4]
                ))
        return conversations
    
    async def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title."""
        await self._sqlite_conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.utcnow().isoformat(), conversation_id)
        )
        await self._sqlite_conn.commit()
    
    async def update_conversation_timestamp(self, conversation_id: str):
        """Update conversation updated_at timestamp."""
        await self._sqlite_conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), conversation_id)
        )
        await self._sqlite_conn.commit()
    
    async def delete_conversation(self, conversation_id: str):
        """
        Delete conversation and all associated messages (cascade).
        
        Args:
            conversation_id: UUID of conversation to delete
        """
        await self._sqlite_conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        await self._sqlite_conn.commit()
    
    # ============================================
    # Message Methods (SQLite - Async)
    # ============================================
    
    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_id: str,
        token_count: int
    ) -> str:
        """
        Save a message to a conversation.
        
        Args:
            conversation_id: UUID of conversation
            role: 'user', 'assistant', or 'system'
            content: Message content
            model_id: Model that generated the message
            token_count: Number of tokens in the message
            
        Returns:
            message_id: UUID of created message
        """
        message_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        await self._sqlite_conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, model_id, token_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, model_id, token_count, now)
        )
        await self._sqlite_conn.commit()
        
        # Update conversation timestamp
        await self.update_conversation_timestamp(conversation_id)
        
        return message_id
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get all messages in a conversation.
        
        Args:
            conversation_id: UUID of conversation
            limit: Optional limit on number of messages
            
        Returns:
            List of Message objects in chronological order
        """
        query = """
            SELECT id, conversation_id, role, content, model_id, token_count, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        messages = []
        async with self._sqlite_conn.execute(query, (conversation_id,)) as cursor:
            async for row in cursor:
                messages.append(Message(
                    id=row[0],
                    conversation_id=row[1],
                    role=row[2],
                    content=row[3],
                    model_id=row[4],
                    token_count=row[5],
                    created_at=row[6]
                ))
        return messages
    
    async def get_recent_history(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[Dict[str, str]]:
        """
        Get recent messages formatted for PydanticAI.
        
        This is the key method for Step 4 (Orchestrator).
        
        Args:
            conversation_id: UUID of conversation
            limit: Number of recent messages to retrieve
            
        Returns:
            List of message dicts in format: [{"role": "user", "content": "..."}]
        """
        messages = []
        
        # Get last N messages
        async with self._sqlite_conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit)
        ) as cursor:
            async for row in cursor:
                messages.append({
                    "role": row[0],
                    "content": row[1]
                })
        
        # Reverse to get chronological order
        messages.reverse()
        return messages
    
    async def get_conversation_token_count(self, conversation_id: str) -> int:
        """
        Get total token count for a conversation.
        
        Args:
            conversation_id: UUID of conversation
            
        Returns:
            Total token count across all messages
        """
        async with self._sqlite_conn.execute(
            "SELECT SUM(token_count) FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row[0] else 0
    
    # ============================================
    # User Facts Methods (DuckDB - Async Wrapped)
    # ============================================
    
    async def save_user_fact(
        self,
        category: str,
        fact_text: str,
        confidence: float
    ) -> str:
        """
        Save a user fact to DuckDB.
        
        Args:
            category: Fact category (Personal, Professional, Preferences, Tech Stack, or dynamic)
            fact_text: The actual fact
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            fact_id: UUID of created fact
        """
        def _save_fact():
            fact_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            self._duckdb_conn.execute(
                """
                INSERT INTO facts (id, category, fact_text, confidence, last_seen, is_active)
                VALUES (?, ?, ?, ?, ?, TRUE)
                """,
                (fact_id, category, fact_text, confidence, now)
            )
            return fact_id
        
        return await asyncio.to_thread(_save_fact)
    
    async def get_user_facts(self, active_only: bool = True) -> List[UserFact]:
        """
        Retrieve all user facts.
        
        Args:
            active_only: If True, only return active facts
            
        Returns:
            List of UserFact objects
        """
        def _get_facts():
            query = "SELECT id, category, fact_text, confidence, last_seen, is_active FROM facts"
            if active_only:
                query += " WHERE is_active = TRUE"
            query += " ORDER BY last_seen DESC"
            
            result = self._duckdb_conn.execute(query).fetchall()
            
            facts = []
            for row in result:
                facts.append(UserFact(
                    id=row[0],
                    category=row[1],
                    fact_text=row[2],
                    confidence=row[3],
                    last_seen=row[4],
                    is_active=row[5]
                ))
            return facts
        
        return await asyncio.to_thread(_get_facts)
    
    async def get_facts_by_category(self, category: str) -> List[UserFact]:
        """
        Get facts filtered by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of UserFact objects
        """
        def _get_by_category():
            result = self._duckdb_conn.execute(
                """
                SELECT id, category, fact_text, confidence, last_seen, is_active
                FROM facts
                WHERE category = ? AND is_active = TRUE
                ORDER BY confidence DESC, last_seen DESC
                """,
                (category,)
            ).fetchall()
            
            facts = []
            for row in result:
                facts.append(UserFact(
                    id=row[0],
                    category=row[1],
                    fact_text=row[2],
                    confidence=row[3],
                    last_seen=row[4],
                    is_active=row[5]
                ))
            return facts
        
        return await asyncio.to_thread(_get_by_category)
    
    async def update_fact_confidence(self, fact_id: str, new_confidence: float):
        """
        Update confidence score for a fact.
        
        Args:
            fact_id: UUID of fact
            new_confidence: New confidence score (0.0 to 1.0)
        """
        def _update():
            self._duckdb_conn.execute(
                "UPDATE facts SET confidence = ?, last_seen = ? WHERE id = ?",
                (new_confidence, datetime.utcnow().isoformat(), fact_id)
            )
        
        await asyncio.to_thread(_update)
    
    async def deactivate_fact(self, fact_id: str):
        """
        Mark a fact as inactive (soft delete).
        
        Args:
            fact_id: UUID of fact to deactivate
        """
        def _deactivate():
            self._duckdb_conn.execute(
                "UPDATE facts SET is_active = FALSE WHERE id = ?",
                (fact_id,)
            )
        
        await asyncio.to_thread(_deactivate)
    
    async def search_facts(self, query: str) -> List[UserFact]:
        """
        Search facts by text content.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching UserFact objects
        """
        def _search():
            result = self._duckdb_conn.execute(
                """
                SELECT id, category, fact_text, confidence, last_seen, is_active
                FROM facts
                WHERE fact_text LIKE ? AND is_active = TRUE
                ORDER BY confidence DESC
                """,
                (f"%{query}%",)
            ).fetchall()
            
            facts = []
            for row in result:
                facts.append(UserFact(
                    id=row[0],
                    category=row[1],
                    fact_text=row[2],
                    confidence=row[3],
                    last_seen=row[4],
                    is_active=row[5]
                ))
            return facts
        
        return await asyncio.to_thread(_search)
    
    # ============================================
    # Cleanup and Close
    # ============================================
    
    async def close(self):
        """Close both database connections."""
        if self._sqlite_conn:
            await self._sqlite_conn.close()
        
        if self._duckdb_conn:
            await asyncio.to_thread(self._duckdb_conn.close)
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for SQLite transactions.
        
        Usage:
            async with db.transaction():
                await db.save_message(...)
                await db.save_message(...)
        """
        try:
            yield self._sqlite_conn
            await self._sqlite_conn.commit()
        except Exception:
            await self._sqlite_conn.rollback()
            raise


# ============================================
# Convenience Functions
# ============================================

async def create_database(
    sqlite_path: str = "data/buddy.db",
    duckdb_path: str = "data/knowledge.duckdb"
) -> BuddyDatabase:
    """
    Create and initialize a BuddyDatabase instance.
    
    Args:
        sqlite_path: Path to SQLite database
        duckdb_path: Path to DuckDB database
        
    Returns:
        Initialized BuddyDatabase instance
    """
    db = BuddyDatabase(sqlite_path, duckdb_path)
    await db.initialize()
    return db