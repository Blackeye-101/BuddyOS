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

from core.fact_utils import FactNormalizer


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
    keywords: str = ""
    topics: str = ""


@dataclass
class UserFact:
    """User knowledge fact stored in DuckDB."""
    id: str
    category: str
    fact_text: str
    confidence: float  # 0.0 to 1.0
    last_seen: str
    is_active: bool = True
    fact_key: Optional[str] = None


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
        self._normalizer: Optional[FactNormalizer] = None
    
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
                keywords TEXT NOT NULL DEFAULT '',
                topics TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (conversation_id) 
                    REFERENCES conversations(id) 
                    ON DELETE CASCADE
            )
        """)

        # Migration guards for existing databases
        for col, definition in [
            ("keywords", "TEXT NOT NULL DEFAULT ''"),
            ("topics",   "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                await self._sqlite_conn.execute(
                    f"ALTER TABLE messages ADD COLUMN {col} {definition}"
                )
            except Exception:
                pass  # Column already exists — safe to ignore

        # Create index for messages
        await self._sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id, created_at)
        """)

        await self._sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_keywords
            ON messages(conversation_id, keywords)
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
                    fact_key VARCHAR,
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

            self._duckdb_conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_key
                ON facts(fact_key)
            """)
        
        # Wrap synchronous DuckDB in async
        await asyncio.to_thread(_create_duckdb_schema)
        self._normalizer = FactNormalizer()
    
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
        token_count: int,
        keywords: str = "",
        topics: str = "",
    ) -> str:
        """
        Save a message to a conversation.

        Args:
            conversation_id: UUID of conversation
            role: 'user', 'assistant', or 'system'
            content: Message content
            model_id: Model that generated the message
            token_count: Number of tokens in the message
            keywords: Comma-sentinel string e.g. ",python,django," for exact-word LIKE search
            topics: Comma-separated LLM-generated topic labels (may be filled later)

        Returns:
            message_id: UUID of created message
        """
        message_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        await self._sqlite_conn.execute(
            """
            INSERT INTO messages
                (id, conversation_id, role, content, model_id, token_count, created_at,
                 keywords, topics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, model_id, token_count, now,
             keywords, topics),
        )
        await self._sqlite_conn.commit()

        # Update conversation timestamp
        await self.update_conversation_timestamp(conversation_id)

        return message_id

    async def update_message_topics(self, message_id: str, topics: str) -> None:
        """Back-fill the topics column after background LLM extraction."""
        await self._sqlite_conn.execute(
            "UPDATE messages SET topics = ? WHERE id = ?",
            (topics, message_id),
        )
        await self._sqlite_conn.commit()
    
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
            SELECT id, conversation_id, role, content, model_id, token_count, created_at,
                   keywords, topics
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
                    created_at=row[6],
                    keywords=row[7] or "",
                    topics=row[8] or "",
                ))
        return messages
    
    async def get_recent_history(
        self,
        conversation_id: str,
        query_keywords: Optional[List[str]] = None,
        tier1_limit: int = 5,
        token_budget: int = 1000,
    ) -> List[Dict[str, str]]:
        """
        Two-tiered retrieval of conversation history for LLM context.

        Tier 1 (temporal): the absolute last `tier1_limit` messages.
        Tier 2 (semantic):  up to 3 historical messages whose `keywords`
            column contains any of `query_keywords` (exact-word match via
            comma-sentinel LIKE pattern ',keyword,').

        The merged list is deduplicated, sorted chronologically, and capped
        at `token_budget` tokens. Tier 1 messages are always included first.

        Args:
            conversation_id: UUID of conversation
            query_keywords: Keywords extracted from the current user message
            tier1_limit: Number of most-recent messages always included
            token_budget: Maximum total token_count across all returned messages

        Returns:
            List of {"role": str, "content": str} dicts in chronological order
        """
        # -- Tier 1: last N messages -----------------------------------------
        tier1_rows: list[tuple] = []
        async with self._sqlite_conn.execute(
            """
            SELECT id, role, content, token_count, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, tier1_limit),
        ) as cursor:
            async for row in cursor:
                tier1_rows.append(row)
        tier1_rows.reverse()  # chronological order
        tier1_ids = {row[0] for row in tier1_rows}

        # -- Tier 2: keyword-matched historical messages ----------------------
        tier2_hits: dict[str, list] = {}  # id -> [id, role, content, token_count, created_at, hit_count]
        if query_keywords:
            for kw in query_keywords:
                # Comma-sentinel exact-word match: '%,keyword,%' finds the sentinel
                # anywhere in the stored string while preventing partial-word hits
                # (e.g. ',python,' in '%,python,%' does NOT match ',pythonic,')
                pattern = f"%,{kw},%"
                async with self._sqlite_conn.execute(
                    """
                    SELECT id, role, content, token_count, created_at
                    FROM messages
                    WHERE conversation_id = ? AND keywords LIKE ?
                    """,
                    (conversation_id, pattern),
                ) as cursor:
                    async for row in cursor:
                        msg_id = row[0]
                        if msg_id in tier1_ids:
                            continue
                        if msg_id in tier2_hits:
                            tier2_hits[msg_id][-1] += 1
                        else:
                            tier2_hits[msg_id] = [*row, 1]  # append hit_count

        # Sort by hit count desc, take top 3
        tier2_rows = sorted(tier2_hits.values(), key=lambda r: r[-1], reverse=True)[:3]

        # -- Merge, sort chronologically, apply token budget ------------------
        # Tier 1 always included first; Tier 2 fills remaining budget
        budget_used = sum(r[3] for r in tier1_rows)
        selected_tier2: list[tuple] = []
        for row in sorted(tier2_rows, key=lambda r: r[4]):  # sort by created_at
            token_count = row[3]
            if budget_used + token_count <= token_budget:
                selected_tier2.append(row)
                budget_used += token_count

        # Combine and sort all selected rows by created_at
        all_rows = [
            (r[0], r[1], r[2], r[4]) for r in tier1_rows  # (id, role, content, created_at)
        ] + [
            (r[0], r[1], r[2], r[4]) for r in selected_tier2
        ]
        all_rows.sort(key=lambda r: r[3])  # sort by created_at ASC

        return [{"role": r[1], "content": r[2]} for r in all_rows]
    
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
        Save a user fact to DuckDB using an UPSERT pattern keyed on fact_key.

        - New fact → INSERT.
        - Duplicate fact (same key, same text) → increment confidence (+0.1, max 1.0).
        - Contradiction (same key, different text) → overwrite text, reset confidence to 0.6.

        Args:
            category: Fact category (Personal, Professional, Preferences, Tech Stack, or dynamic)
            fact_text: The actual fact
            confidence: Initial confidence score (0.0 to 1.0)

        Returns:
            fact_id: UUID of the affected row
        """
        fact_key = await self._normalizer.normalize(category, fact_text)
        normalizer = self._normalizer  # capture for thread closure

        def _upsert_fact() -> str:
            now = datetime.utcnow().isoformat()

            existing = self._duckdb_conn.execute(
                "SELECT id, fact_text, confidence FROM facts WHERE fact_key = ? AND is_active = TRUE",
                (fact_key,),
            ).fetchone()

            if existing:
                existing_id, existing_text, existing_conf = existing
                if normalizer.is_contradiction(fact_key, existing_text, fact_text):
                    self._duckdb_conn.execute(
                        "UPDATE facts SET fact_text = ?, confidence = 0.6, last_seen = ? WHERE id = ?",
                        (fact_text, now, existing_id),
                    )
                else:
                    new_conf = min(existing_conf + 0.1, 1.0)
                    self._duckdb_conn.execute(
                        "UPDATE facts SET confidence = ?, last_seen = ? WHERE id = ?",
                        (new_conf, now, existing_id),
                    )
                return existing_id

            fact_id = str(uuid.uuid4())
            self._duckdb_conn.execute(
                """
                INSERT INTO facts (id, fact_key, category, fact_text, confidence, last_seen, is_active)
                VALUES (?, ?, ?, ?, ?, ?, TRUE)
                """,
                (fact_id, fact_key, category, fact_text, confidence, now),
            )
            return fact_id

        return await asyncio.to_thread(_upsert_fact)
    
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