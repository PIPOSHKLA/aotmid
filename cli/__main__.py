"""``python -m cli`` — unified entry point."""

from __future__ import annotations

from cli.fetch import register_fetch_parser
from cli.train import register_train_parser

try:
    from argparse import ArgumentParser
except ImportError:
    from argparse import ArgumentParser  # noqa: F811


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="python -m cli",
        description="AOT Stock Network — CLI toolset",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register_fetch_parser(sub)
    register_train_parser(sub)
    return parser


def main() -> None:
    """Dispatch CLI subcommand."""
    from aot_stock_network.logging_setup import setup_logging
    from aot_stock_network.seed import set_seed

    setup_logging()
    set_seed()

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        from cli.fetch import handle_fetch

        handle_fetch(args)
    elif args.command == "train":
        from cli.train import handle_train

        handle_train(args)
    else:
        # fallback: try dynamic import
        import importlib

        try:
            mod = importlib.import_module(f"cli.{args.command}")
            if hasattr(mod, f"handle_{args.command}"):
                getattr(mod, f"handle_{args.command}")(args)
        except (ImportError, AttributeError):
            parser.print_help()


if __name__ == "__main__":
    main()
