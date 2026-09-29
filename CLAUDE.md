# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`western_blot` — a Karcytics plugin for automated Western Blot / Ponceau S densitometry. Treats gel images as 1D intensity signal series (vertical projection + rolling-ball morphological top-hat baseline, AUC integration, SNR-based peak finding), replicating the ImageJ protocol without manual thresholding. Runs as a **separate process** (`process_model = "isolated"`) — never import `karcytics.*` from the Hub, only `karcytics_sdk` (sibling `Karcytics-SDK` repo, editable local path via `[tool.uv.sources]`). Entry point: `karcytics_plugins.western_blot:initialize`.

Implements the Hub's `WizardPanel` interface (a step-driven UI flow) and manages state via an `AnalysisState` dataclass.

See `../Karcytics/ECOSYSTEM.md` for the full map of all 7 Karcytics repos and their relationships.

Read `docs/developer/01_ANALYSIS_AND_MATH.md` before touching the analysis engine (rolling-ball baselines, AUC integration, SNR logic) and `docs/developer/00_UI_ARCHITECTURE.md` before touching the wizard/`ImageCanvas` — both are current and detailed; prefer them over re-deriving the math from source.

## Commands

```bash
uv run pytest tests/unit/ --tb=short -q   # what pre-commit runs
uv run pytest tests/ -q                   # full suite incl. tests/ui
uv run pytest tests/unit/analysis/test_x.py -q
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run mypy --explicit-package-bases src/
```

Pre-commit runs ruff, mypy, pip-audit, license compliance, `tests/unit/` (not `tests/ui`), plugin re-signing, and security-ledger verification. Don't pre-run the full suite before committing.

## Structure

`src/karcytics_plugins/western_blot/`: `analysis/` (rolling-ball baseline, peak/lane detection, fold-change normalization — no Qt) and `ui/` (wizard panel, `ImageCanvas`). Same `PluginBase`/`AnalysisBase` split as the other Karcytics plugins — see [Karcytics-SDK](../Karcytics-SDK)'s `CLAUDE.md`.

`tests/unit/analysis` and `tests/unit/ui` mirror that split; `tests/ui` (top-level) holds Qt/display-dependent tests.
