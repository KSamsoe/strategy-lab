"""Run registry and journals: the lab's memory of what it did and why.

``registry.sqlite`` answers "what was run, from which commit, on which data,
and how many attempts came before this one"; ``journal.sqlite`` answers "what
did the strategy see at 09:35 and what did the gate do about it".
"""

from __future__ import annotations

from lab.registry.db import SCHEMA_SQL, close_all, connect, init_db, journal_path, registry_path, transaction
from lab.registry.journal import DecisionJournal, EventJournal
from lab.registry.runs import (
    RUN_KINDS,
    RunRecord,
    RunRegistry,
    config_hash,
    git_commit,
    run_artifact_dir,
    run_from_row,
)

__all__ = [
    "SCHEMA_SQL",
    "DecisionJournal",
    "EventJournal",
    "RUN_KINDS",
    "RunRecord",
    "RunRegistry",
    "close_all",
    "config_hash",
    "connect",
    "git_commit",
    "init_db",
    "journal_path",
    "registry_path",
    "run_artifact_dir",
    "run_from_row",
    "transaction",
]
