"""
BuddyOS Hybrid Persistence Layer

This module manages two distinct storage systems:
1. SQLite (via aiosqlite) - Conversations and messages (transactional)
2. DuckDB - User facts and knowledge (analytical)
"""

import asyncio
import logging
import aiosqlite
import duckdb
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """Conversation metadata."""
    id: str
    created_at: str
    updated_at: str
    title: str
    model_id: str
    summary: str = ""


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
        self._duckdb_lock = asyncio.Lock()
        self._normalizer = None  # Injected by BuddyOrchestrator after init
        self._vss_available: bool = False  # Set True after successful VSS load
    
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
                model_id TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT ''
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

        # Migration: add summary column to conversations for pre-existing databases
        try:
            await self._sqlite_conn.execute(
                "ALTER TABLE conversations ADD COLUMN summary TEXT NOT NULL DEFAULT ''"
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

            # ── VSS extension (vector similarity search) ────────────────
            try:
                self._duckdb_conn.execute("INSTALL vss;")
                self._duckdb_conn.execute("LOAD vss;")
                self._vss_available = True
                logger.info("DuckDB VSS extension loaded.")
            except Exception as exc:
                self._vss_available = False
                logger.warning(
                    "DuckDB VSS unavailable — falling back to full fact load. Error: %s", exc
                )

            # ── Core facts table ────────────────────────────────────────
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

            # Migration: add embedding column to existing databases
            try:
                self._duckdb_conn.execute(
                    "ALTER TABLE facts ADD COLUMN IF NOT EXISTS embedding FLOAT[384]"
                )
            except Exception as exc:
                logger.warning("Embedding column migration skipped: %s", exc)

            # ── Scalar indexes ──────────────────────────────────────────
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

            # ── HNSW vector index ────────────────────────────────────────
            if self._vss_available:
                try:
                    self._duckdb_conn.execute(
                        "SET hnsw_enable_experimental_persistence = true;"
                    )
                    self._duckdb_conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_facts_embedding
                        ON facts USING HNSW (embedding)
                        WITH (metric = 'cosine')
                    """)
                    logger.info("HNSW index on facts.embedding ready.")
                except Exception as exc:
                    logger.warning("HNSW index creation skipped: %s", exc)

            # ── Personal RAG Documents ────────────────────────────────
            self._duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR PRIMARY KEY,
                    filename VARCHAR NOT NULL,
                    filepath VARCHAR NOT NULL,
                    filehash VARCHAR UNIQUE NOT NULL,
                    extension VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)

            self._duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id VARCHAR PRIMARY KEY,
                    document_id VARCHAR NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text VARCHAR NOT NULL,
                    embedding FLOAT[384],
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)

            if self._vss_available:
                try:
                    self._duckdb_conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                        ON document_chunks USING HNSW (embedding)
                        WITH (metric = 'cosine')
                    """)
                    logger.info("HNSW index on document_chunks.embedding ready.")
                except Exception as exc:
                    logger.warning("HNSW index on chunks creation skipped: %s", exc)

        # Wrap synchronous DuckDB in async
        async with self._duckdb_lock:
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
            INSERT INTO conversations (id, created_at, updated_at, title, model_id, summary)
            VALUES (?, ?, ?, ?, ?, '')
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
            "SELECT id, created_at, updated_at, title, model_id, summary FROM conversations WHERE id = ?",
            (conversation_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Conversation(
                    id=row[0],
                    created_at=row[1],
                    updated_at=row[2],
                    title=row[3],
                    model_id=row[4],
                    summary=row[5] or ""
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
            SELECT id, created_at, updated_at, title, model_id, summary
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
                    model_id=row[4],
                    summary=row[5] or ""
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

    async def update_conversation_summary(self, conversation_id: str, summary: str):
        """Persist an auto-generated one-line summary for a conversation."""
        await self._sqlite_conn.execute(
            "UPDATE conversations SET summary = ? WHERE id = ?",
            (summary, conversation_id),
        )
        await self._sqlite_conn.commit()

    async def purge_all_conversations(self):
        """Delete every conversation and all associated messages (via CASCADE)."""
        await self._sqlite_conn.execute("DELETE FROM conversations")
        await self._sqlite_conn.commit()

    async def purge_all_facts(self):
        """Hard-delete every user fact from DuckDB (leaves documents intact)."""
        def _purge():
            self._duckdb_conn.execute("DELETE FROM facts")

        async with self._duckdb_lock:
            await asyncio.to_thread(_purge)

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

    async def get_oldest_messages(self, conversation_id: str, limit: int) -> List[Message]:
        """
        Get the oldest messages in a conversation.
        Useful for context window summarization.
        """
        async with self._sqlite_conn.execute(
            """
            SELECT id, conversation_id, role, content, model_id, 
                   token_count, created_at, keywords, topics
            FROM messages 
            WHERE conversation_id = ? 
            ORDER BY created_at ASC 
            LIMIT ?
            """,
            (conversation_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [Message(
                id=row[0],
                conversation_id=row[1],
                role=row[2],
                content=row[3],
                model_id=row[4],
                token_count=row[5],
                created_at=row[6],
                keywords=row[7],
                topics=row[8]
            ) for row in rows]

    async def delete_messages(self, message_ids: List[str]) -> None:
        """
        Delete a list of messages by ID.
        """
        if not message_ids:
            return
            
        placeholders = ",".join("?" for _ in message_ids)
        await self._sqlite_conn.execute(
            f"DELETE FROM messages WHERE id IN ({placeholders})",
            message_ids
        )
        await self._sqlite_conn.commit()
    
    # ============================================
    # User Facts Methods (DuckDB - Async Wrapped)
    # ============================================
    
    async def save_user_fact(
        self,
        category: str,
        fact_text: str,
        confidence: float,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Save a user fact to DuckDB.
        Generates a unique fact_key via UUID to ensure we avoid DuckDB collision constraints,
        allowing the system to store multiple facts of the same category.
        """
        import uuid
        from datetime import datetime
        
        # Generate a unique programmatic ID instead of relying on unreliable LLM categorizations
        safe_cat = "".join(c if c.isalnum() else "_" for c in category.lower())
        fact_key = f"{safe_cat}_{uuid.uuid4().hex[:8]}"
        
        normalizer = self._normalizer  # capture for thread closure

        # Generate embedding before entering the sync thread
        from core.embeddings import generate_embedding_safe
        embedding = await generate_embedding_safe(fact_text)

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
                        "UPDATE facts SET fact_text = ?, confidence = 0.6, last_seen = ?, embedding = ? WHERE id = ?",
                        (fact_text, now, embedding, existing_id),
                    )
                else:
                    new_conf = min(existing_conf + 0.1, 1.0)
                    self._duckdb_conn.execute(
                        "UPDATE facts SET confidence = ?, last_seen = ?, embedding = ? WHERE id = ?",
                        (new_conf, now, embedding, existing_id),
                    )
                return existing_id

            fact_id = str(uuid.uuid4())
            self._duckdb_conn.execute(
                """
                INSERT INTO facts (id, fact_key, category, fact_text, confidence, last_seen, is_active, embedding)
                VALUES (?, ?, ?, ?, ?, ?, TRUE, ?)
                """,
                (fact_id, fact_key, category, fact_text, confidence, now, embedding),
            )
            return fact_id

        async with self._duckdb_lock:

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
        
        async with self._duckdb_lock:
        
            return await asyncio.to_thread(_get_facts)

    async def get_relevant_facts(
        self,
        query_embedding: list,
        limit: int = 5,
    ) -> List[UserFact]:
        """
        Retrieve the most semantically relevant active facts using cosine similarity.

        Falls back to recency-ordered full load when:
        - VSS extension is not available, or
        - No facts have embeddings yet.

        Args:
            query_embedding: 384-float list from core.embeddings.generate_embedding.
            limit:           Maximum number of facts to return.

        Returns:
            List of UserFact objects sorted by relevance (most relevant first).
        """
        if not self._vss_available:
            logger.debug("VSS unavailable — falling back to full fact load.")
            all_facts = await self.get_user_facts(active_only=True)
            return all_facts[:limit]

        def _semantic_search() -> List[UserFact]:
            try:
                rows = self._duckdb_conn.execute(
                    """
                    SELECT id, category, fact_text, confidence, last_seen, is_active,
                           array_cosine_similarity(embedding, ?::FLOAT[384]) AS similarity
                    FROM facts
                    WHERE is_active = TRUE AND embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT ?
                    """,
                    [query_embedding, limit],
                ).fetchall()

                return [
                    UserFact(
                        id=row[0],
                        category=row[1],
                        fact_text=row[2],
                        confidence=row[3],
                        last_seen=row[4],
                        is_active=row[5],
                    )
                    for row in rows
                ]
            except Exception as exc:
                logger.warning(
                    "Semantic search failed (%s) — falling back to recency order.", exc
                )
                rows = self._duckdb_conn.execute(
                    """
                    SELECT id, category, fact_text, confidence, last_seen, is_active
                    FROM facts
                    WHERE is_active = TRUE
                    ORDER BY last_seen DESC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchall()
                return [
                    UserFact(
                        id=row[0], category=row[1], fact_text=row[2],
                        confidence=row[3], last_seen=row[4], is_active=row[5],
                    )
                    for row in rows
                ]

        async with self._duckdb_lock:

            return await asyncio.to_thread(_semantic_search)

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
        
        async with self._duckdb_lock:
        
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
        
        async with self._duckdb_lock:
        
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
        
        async with self._duckdb_lock:
        
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
        
        async with self._duckdb_lock:
        
            return await asyncio.to_thread(_search)
    
    # ============================================
    # Cleanup and Close
    # ============================================
    
    # ============================================
    # Personal RAG Methods (DuckDB)
    # ============================================

    async def document_hash_exists(self, file_hash: str) -> bool:
        """Check if a file hash already exists in the database."""
        def _check():
            result = self._duckdb_conn.execute(
                "SELECT id FROM documents WHERE filehash = ?", [file_hash]
            ).fetchone()
            return result is not None
            
        async with self._duckdb_lock:
            return await asyncio.to_thread(_check)

    async def save_document(self, metadata: Dict[str, Any]) -> str:
        """
        Saves document metadata and its embedded chunks to DuckDB.
        `metadata` should match the output of DocumentParser.process_file,
        but chunks must be augmented with 'embedding' vectors.

        Returns:
            document_id: UUID of the inserted document
        """
        def _save():
            doc_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            # 1. Insert Document metadata
            self._duckdb_conn.execute(
                """
                INSERT INTO documents (id, filename, filepath, filehash, extension, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [doc_id, metadata['filename'], metadata['filepath'], metadata['filehash'], metadata['extension'], now]
            )
            
            # 2. Insert Chunks
            for i, chunk in enumerate(metadata['chunks']):
                chunk_id = str(uuid.uuid4())
                self._duckdb_conn.execute(
                    """
                    INSERT INTO document_chunks (id, document_id, chunk_index, chunk_text, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [chunk_id, doc_id, i, chunk['text'], chunk['embedding']]
                )
            return doc_id

        async with self._duckdb_lock:
            return await asyncio.to_thread(_save)

    async def search_document_chunks(self, embedding: List[float], limit: int = 5, distance_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Searches for relevant document chunks using vector similarity.
        """
        if not self._vss_available:
            return []  # Graceful fail without VSS
            
        def _search():
            # SQL logic matches what we use for user facts
            query = f"""
                SELECT c.chunk_text, d.filename, d.filepath, list_cosine_distance(c.embedding, ?::FLOAT[384]) as distance
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE list_cosine_distance(c.embedding, ?::FLOAT[384]) <= ?
                ORDER BY distance ASC
                LIMIT ?
            """
            
            # Using embedding parameter twice for SELECT and WHERE clauses
            results = self._duckdb_conn.execute(query, [embedding, embedding, distance_threshold, limit]).fetchall()
            
            return [
                {
                    "text": row[0],
                    "filename": row[1],
                    "filepath": row[2],
                    "distance": row[3]
                }
                for row in results
            ]
            
        async with self._duckdb_lock:
            return await asyncio.to_thread(_search)

    async def get_messages(self, conversation_id: str) -> List[Message]:
        """Alias for get_conversation_history — used by the Streamlit UI layer."""
        return await self.get_conversation_history(conversation_id)

    async def list_documents(self) -> List[Dict[str, Any]]:
        """Return metadata for all ingested documents, newest first."""
        def _list():
            rows = self._duckdb_conn.execute(
                "SELECT id, filename, extension, created_at FROM documents ORDER BY created_at DESC"
            ).fetchall()
            return [
                {"id": r[0], "filename": r[1], "extension": r[2], "created_at": r[3]}
                for r in rows
            ]

        async with self._duckdb_lock:
            return await asyncio.to_thread(_list)

    async def purge_all_documents(self):
        """Hard-delete all document chunks and documents from DuckDB (leaves facts intact)."""
        def _purge():
            self._duckdb_conn.execute("DELETE FROM document_chunks")
            self._duckdb_conn.execute("DELETE FROM documents")

        async with self._duckdb_lock:
            await asyncio.to_thread(_purge)

    async def close(self):
        """Close both database connections."""
        if self._sqlite_conn:
            await self._sqlite_conn.close()
        
        if self._duckdb_conn:
            async with self._duckdb_lock:
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
