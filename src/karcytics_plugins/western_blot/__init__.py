"""Western Blot Densitometry Plugin for Karcytics."""

from typing import Any

__version__ = "1.2.2"
__plugin_id__ = "western_blot"


def get_panel_class() -> type:
    """Returns the main QWidget class that should be injected into the UI.

    Standard Karcytics entry point.
    """
    from .ui.western_blot_panel import WesternBlotPanel

    return WesternBlotPanel


def cleanup() -> None:
    """Module-level cleanup (no instance state)."""


def shutdown() -> None:
    """Module-level shutdown."""


# Late import: PluginContext lives in karcytics_sdk which depends on this package
# at runtime — importing it at module load would create a circular import.
from karcytics_sdk.plugin.context import PluginContext  # noqa: E402


def initialize(context: PluginContext) -> Any:
    """V3 Plugin Entry Point."""
    logger = context.get("logger")
    logger.info("Initializing Western Blot Densitometry with PluginContext")

    # Return the module itself so the core can call .get_panel_class(), .cleanup(), etc.
    import sys

    return sys.modules[__name__]
