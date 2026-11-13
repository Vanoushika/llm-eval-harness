"""Dataset loading — suites defined in YAML or JSON files.

A suite file is the plain JSON shape of an EvalSuite, so a file, an API
payload and a stored row are all the same document.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import yaml

from .models import EvalSuite


def stable_suite_id(name: str) -> str:
    """Deterministic ID for a file-loaded suite, derived from its name.

    Two loads of the same suite file must compare as the *same* suite —
    otherwise ``compare_runs`` (which requires matching suite IDs) can
    never compare runs produced by separate CLI invocations, which is
    exactly how a CI gate uses it. An explicit ``id`` in the file still
    wins; this only fills the default.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"llm-eval-harness:suite:{name}").hex[:12]


def load_suite(path: str | Path) -> EvalSuite:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"suite file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported suite file type: {path.suffix} (use .yaml/.yml/.json)")
    if not isinstance(data, dict):
        raise ValueError(f"suite file {path} must contain a single mapping")
    suite = EvalSuite(**data)
    if "id" not in data:
        suite.id = stable_suite_id(suite.name)
    return suite
