from __future__ import annotations

__all__ = ["skeleton", "driver", "extract"]

# Optional CLI entrypoint
def main(argv=None) -> int:
    """
    Entry point so you can run:
        python -m planner --help
        python -m planner "rev (rev xs) = xs"
    """
    from . import cli
    return cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
