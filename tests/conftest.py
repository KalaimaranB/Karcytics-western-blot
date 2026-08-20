"""Shared pytest fixtures for the western_blot plugin test suite.

Unlike flow-cytometry's conftest.py, this does not replace karcytics_sdk
with a hand-maintained MagicMock shim. The real karcytics_sdk (resolved via
the `[tool.uv.sources]` sibling-repo editable install — see pyproject.toml)
imports and instantiates cleanly under `QT_QPA_PLATFORM=offscreen` with no
Hub process required, so tests exercise the actual SDK contract instead of
a mock that can silently drift out of sync with it.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo_root, "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def blank_image() -> np.ndarray:
    """A flat, featureless grayscale image — useful for shape/plumbing tests."""
    return np.full((200, 400), 0.9, dtype=np.float64)


@pytest.fixture
def synthetic_blot_image() -> np.ndarray:
    """A grayscale image with 4 vertical lanes, each containing one dark band.

    Background is bright (~0.9); lanes are darker columns; each lane has one
    darker horizontal band roughly in its middle third — mimics a real western
    blot closely enough to exercise lane/band detection without needing a
    real fixture image on disk.
    """
    height, width = 300, 400
    image = np.full((height, width), 0.9, dtype=np.float64)

    lane_width = width // 4
    for lane_idx in range(4):
        x0 = lane_idx * lane_width + lane_width // 4
        x1 = (lane_idx + 1) * lane_width - lane_width // 4
        band_y = 100 + lane_idx * 20
        image[band_y - 8 : band_y + 8, x0:x1] = 0.2

    return image
