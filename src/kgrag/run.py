"""`kgrag <stage>` — the pipeline CLI.

Stages are independently runnable and each one reads the previous stage's jsonl, so a
failure never costs more than the stage it happened in. `all` runs them in order.
"""

from __future__ import annotations

import argparse

from . import fireworks

STAGES = ("fetch", "chunk", "extract", "resolve", "load", "embed")


def main() -> None:
    parser = argparse.ArgumentParser(prog="kgrag", description=__doc__)
    parser.add_argument(
        "stage",
        choices=(*STAGES, "all", "verify", "models", "candidates", "mine-pairs", "sweep", "bakeoff"),
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=5.0,
        help="hard cap on Fireworks spend in USD for this run (default: 5.00)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="process only the first N chunks (dry runs)"
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the cost-projection confirmation prompt"
    )
    parser.add_argument(
        "--only",
        default=None,
        help="bakeoff: run just the models whose label contains this substring",
    )
    args = parser.parse_args()

    fireworks.METER.limit_usd = args.budget

    if args.stage == "models":
        fireworks.list_models()
        return
    if args.stage == "verify":
        from . import verify

        verify.run()
        return
    if args.stage == "candidates":
        from . import resolve

        resolve.write_candidates()
        return
    if args.stage == "mine-pairs":
        from . import resolve

        resolve.mine_pairs()
        return
    if args.stage == "sweep":
        from . import resolve

        resolve.sweep()
        return
    if args.stage == "bakeoff":
        from . import bakeoff

        bakeoff.run(only=args.only)
        return

    stages = STAGES if args.stage == "all" else (args.stage,)
    try:
        for name in stages:
            print(f"\n=== {name}")
            match name:
                case "fetch":
                    from . import fetch

                    fetch.run()
                case "chunk":
                    from . import chunk

                    chunk.run()
                case "extract":
                    from . import extract

                    extract.run(limit=args.limit, assume_yes=args.yes)
                case "resolve":
                    from . import resolve

                    resolve.run()
                case "load":
                    from . import load

                    load.run()
                case "embed":
                    from . import embed

                    embed.run(assume_yes=args.yes, limit=args.limit)
    except fireworks.BudgetExceeded as exc:
        raise SystemExit(f"\nBUDGET STOP: {exc}") from exc
    finally:
        if fireworks.METER.calls or fireworks.METER.cached:
            print(f"\n{fireworks.METER.report()}")
