"""Render a tournament dashboard from a complete or in-progress JSON payload."""

from __future__ import annotations

import argparse
from pathlib import Path

from plump.tournament_dashboard import render_tournament_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    output = args.output or args.payload.with_suffix(".png")
    print(
        f"wrote {render_tournament_dashboard(args.payload, output, dpi=args.dpi)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
