"""Karcytics plugin contract tests — see karcytics_sdk.testing.ContractTestBase."""

from pathlib import Path

from karcytics_sdk.testing import ContractTestBase


class TestWesternBlotContract(ContractTestBase):
    PLUGIN_DIR = Path(__file__).parent.parent
