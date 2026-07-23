"""
Agent module for extracting checklist items from case documents.

Submodules are imported lazily (PEP 562) so that lightweight consumers —
e.g. data_processing.py, which only needs TokenizerWrapper — don't pull in
the heavy inference dependencies (vllm/torch) required by the agent runtime.
"""

import importlib

_LAZY_IMPORTS = {
    "Driver": (".driver", "Driver"),
    "BatchDriver": (".driver", "BatchDriver"),
    "Orchestrator": (".orchestrator", "Orchestrator"),
    "SnapshotBuilder": (".snapshot_builder", "SnapshotBuilder"),
    "VLLMClient": (".llm_client", "VLLMClient"),
    "DocumentManager": (".document_manager", "DocumentManager"),
    "TokenizerWrapper": (".tokenizer", "TokenizerWrapper"),
}

__all__ = list(_LAZY_IMPORTS)

__version__ = "0.1.0"


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
