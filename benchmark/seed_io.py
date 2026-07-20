from __future__ import annotations

from pathlib import Path
import re


_SEED_PATTERN = re.compile(r"[0-9]+\Z")


class SeedFileError(ValueError):
    """Raised when a benchmark seed file is empty or malformed."""


def parse_seed_text(text: str, *, source: str = "<seeds>") -> list[int]:
    """Parse an ordered, unique list of non-negative decimal integer seeds.

    Blank lines and comments introduced by ``#`` are ignored.  Inline comments
    are allowed.  Every other non-empty line must contain exactly one decimal
    integer.  Duplicate seeds are rejected instead of silently de-duplicated so
    that the requested episode order remains auditable.
    """

    seeds: list[int] = []
    first_line_by_seed: dict[int, int] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        value_text = raw_line.split("#", 1)[0].strip()
        if not value_text:
            continue
        if _SEED_PATTERN.fullmatch(value_text) is None:
            raise SeedFileError(
                f"{source}:{line_number}: expected one non-negative decimal "
                f"integer seed, got {value_text!r}"
            )
        seed = int(value_text, 10)
        previous_line = first_line_by_seed.get(seed)
        if previous_line is not None:
            raise SeedFileError(
                f"{source}:{line_number}: duplicate seed {seed}; first declared "
                f"on line {previous_line}"
            )
        first_line_by_seed[seed] = line_number
        seeds.append(seed)

    if not seeds:
        raise SeedFileError(f"{source}: seed file contains no seeds")
    return seeds


def load_seeds(path: str | Path) -> list[int]:
    """Load and strictly validate a UTF-8 benchmark seed file."""

    seed_path = Path(path).expanduser()
    if not seed_path.is_file():
        raise FileNotFoundError(f"Seed file does not exist: {seed_path}")
    try:
        text = seed_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SeedFileError(f"Seed file is not valid UTF-8: {seed_path}") from exc
    return parse_seed_text(text, source=str(seed_path))


__all__ = ["SeedFileError", "load_seeds", "parse_seed_text"]
