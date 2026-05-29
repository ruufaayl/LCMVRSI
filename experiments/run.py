#!/usr/bin/env python
"""Entry point for running a single LCMVRSI experiment from a YAML config.

Usage::

    uv run python experiments/run.py -c configs/base.yaml
    uv run python experiments/run.py -c configs/base.yaml --steps 2000 --device cuda -o results

The real logic lives in ``lcmvrsi.train.cli`` so it can be imported and unit-tested; this
file is a thin, discoverable launcher.
"""

from lcmvrsi.train.cli import main

if __name__ == "__main__":
    main()
