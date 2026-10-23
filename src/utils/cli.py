"""Shared command-line flags so every script behaves the same way.

    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
"""

from __future__ import annotations

import argparse

from src.utils.config import load_config


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --config, --profile, --device, --set and --log-level to a parser."""
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--profile", default=None, choices=["local", "colab"],
                        help="config profile (default: DFG_PROFILE env, else auto-detect)")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                        help="compute device (default from config: auto = cuda if available else cpu)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.batch_size=8 (repeatable)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    """Build the resolved config from parsed common args."""
    return load_config(args.config, profile=args.profile, overrides=args.overrides, device=args.device)
