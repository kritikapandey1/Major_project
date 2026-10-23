"""Show the resolved configuration for this machine (profile, device, paths, models).

Usage:
    python scripts/show_config.py                     # auto profile
    python scripts/show_config.py --profile colab --set train.batch_size=8
    python scripts/show_config.py --dump              # full YAML
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import yaml  # noqa: E402

from src.utils.cli import add_common_args, config_from_args  # noqa: E402
from src.utils.config import ConfigError, get_path, resolve_device  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("show_config")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--dump", action="store_true", help="print the full resolved config as YAML")
    args = parser.parse_args()
    setup_logging(args.log_level)

    try:
        cfg = config_from_args(args)
    except ConfigError as e:
        logger.error("Config error: %s", e)
        return 2

    try:
        device = resolve_device(cfg["device"])
    except ImportError:
        device = "unknown (torch not installed)"
    except ConfigError as e:
        logger.error("%s", e)
        return 2

    print(f"Profile : {cfg['profile']}")
    print(f"Device  : requested={cfg['device']} -> {device}")
    print(f"Seed    : {cfg['seed']}")
    print("Paths   :")
    for name in ("work_root", "persist_root", "samples", "raw", "processed", "features",
                 "checkpoints", "logs", "model_cache"):
        p = get_path(cfg, name)
        print(f"  {name:13} {p}  [{'exists' if p.exists() else 'not created yet'}]")
    print("Models  :")
    for key in ("image", "audio", "text", "detectgpt_scoring", "detectgpt_reference"):
        print(f"  {key:20} {cfg['models'][key]}")
    print(f"  {'transcript':20} {cfg['preprocess']['transcript']['engine']} "
          f"{cfg['preprocess']['transcript']['model']} ({cfg['preprocess']['transcript']['compute_type']})")
    print(f"Dims    : {cfg['models']['dims']}  -> fusion d_model={cfg['fusion']['d_model']}")
    print(f"Train   : batch_size={cfg['train']['batch_size']} epochs={cfg['train']['epochs']} "
          f"lr={cfg['train']['lr']}")
    if args.dump:
        print("-" * 60)
        print(yaml.safe_dump(cfg, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
