from __future__ import annotations

import importlib.metadata

import screwmpc_experiments as m


def test_version():
    assert importlib.metadata.version("screwmpc_experiments") == m.__version__
