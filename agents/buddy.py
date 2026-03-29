"""
agents/buddy.py -- re-export shim (Wave 2 Phase B)

BuddyOrchestrator was migrated to core/orchestrator.py.
This file exists solely so that existing imports such as
    from agents.buddy import create_orchestrator
continue to work without changes.
"""

from core.orchestrator import (  # noqa: F401
    BuddyOrchestrator,
    OrchestratorRequest,
    OrchestratorResponse,
    create_orchestrator,
)

__all__ = [
    "BuddyOrchestrator",
    "OrchestratorRequest",
    "OrchestratorResponse",
    "create_orchestrator",
]
