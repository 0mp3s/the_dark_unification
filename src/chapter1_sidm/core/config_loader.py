"""
Configuration loader for Secluded-Majorana-SIDM scripts.

Usage in any script:
    from config_loader import load_config
    cfg = load_config(__file__)

Looks for config.json in the same directory as the calling script.
Alternatively: cfg = load_config(__file__, "path/to/custom_config.json")
Or via command-line: python script.py --config my_config.json
"""
import json
import os
import sys


def load_config(caller_file, config_path=None):
    """
    Load configuration JSON.

    Priority:
      1. --config CLI argument (if present)
      2. config_path argument (if given)
      3. config.json in the same directory as caller_file

    Returns dict (empty if no config found).
    """
    # Check CLI for --config
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    # Default: config.json next to the calling script
    if config_path is None:
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        config_path = os.path.join(caller_dir, "config.json")

    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg
