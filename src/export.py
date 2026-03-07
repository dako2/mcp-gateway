"""Export results to data/servers.json."""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone


def export(entries: list[dict], output_path: str = "data/servers.json") -> str:
    """Write entries to JSON, adding last_checked timestamp."""
    now = datetime.now(timezone.utc).isoformat()

    for entry in entries:
        entry["last_checked"] = now

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"[export] wrote {len(entries)} entries to {output_path}")
    return output_path
