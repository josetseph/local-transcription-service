"""Best-effort release of Python + GPU/unified memory after model use."""

from __future__ import annotations

import gc


def release_accelerator_memory() -> None:
    """Run GC and empty CUDA/MPS caches when safe (no-op if torch absent)."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        mps = getattr(torch, "mps", None)
        backends = getattr(torch, "backends", None)
        backends_mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is None or backends_mps is None or not backends_mps.is_available():
            return
        try:
            allocated = int(mps.driver_allocated_memory())
        except Exception:
            return
        if allocated > 0:
            mps.empty_cache()
    except Exception:
        pass
