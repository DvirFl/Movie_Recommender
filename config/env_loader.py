"""Load a .env file into os.environ before anything else reads environment variables.

Supports:
  - KEY=value
  - KEY="quoted value"
  - KEY='single quoted'
  - # comments
  - blank lines
  - ${OTHER_VAR} variable interpolation

Existing environment variables are NOT overridden (shell always wins).

Usage:
    from config.env_loader import load_env
    load_env()                      # looks for .env in project root
    load_env("/path/to/.env")       # explicit path
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Default locations to search for .env, in priority order
_DEFAULT_SEARCH = [
    Path.cwd() / ".env",
    Path(__file__).parent.parent / ".env",   # project root
]

_INTERPOLATION_RE = re.compile(r"\$\{([^}]+)\}")


def _interpolate(value: str) -> str:
    """Replace ${VAR} with the current value of VAR from os.environ."""
    def replacer(match: re.Match) -> str:
        return os.environ.get(match.group(1), match.group(0))
    return _INTERPOLATION_RE.sub(replacer, value)


def _parse_line(line: str) -> tuple[str, str] | None:
    """Parse a single .env line.  Returns (key, value) or None to skip."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None

    key, _, raw = line.partition("=")
    key = key.strip()

    # Strip optional inline comments: KEY=value  # comment
    raw = raw.split("#")[0].strip()

    # Strip surrounding quotes
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]

    value = _interpolate(raw)
    return key, value


def load_env(path: str | Path | None = None, override: bool = False) -> Path | None:
    """Load .env file into os.environ.

    Args:
        path:     explicit path to a .env file.  If None, searches default locations.
        override: if True, existing env vars ARE overridden by the .env file.
                  Default False (shell always wins).

    Returns:
        The Path of the .env file that was loaded, or None if none was found.
    """
    candidates = [Path(path)] if path else _DEFAULT_SEARCH
    env_path: Path | None = None

    for candidate in candidates:
        if candidate.exists():
            env_path = candidate
            break

    if env_path is None:
        logger.debug("No .env file found (searched: %s).", [str(c) for c in candidates])
        return None

    loaded, skipped = 0, 0
    with env_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            parsed = _parse_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if key in os.environ and not override:
                logger.debug(".env line %d: '%s' already set — skipping.", lineno, key)
                skipped += 1
            else:
                os.environ[key] = value
                loaded += 1

    logger.info(
        "Loaded .env from %s — %d variable(s) set, %d skipped (already in env).",
        env_path, loaded, skipped,
    )
    return env_path
