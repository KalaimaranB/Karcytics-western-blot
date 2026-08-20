# 🎞️ Western Blot Module: Automated Densitometry

This module provides a high-precision pipeline for quantifying Western Blot and Ponceau S stains. It replicates the gold-standard ImageJ protocol while removing human bias through automated peak-finding and dynamic lane mapping.

## Overview
The Karcytics Western Blot plugin treats gel images as a series of 1D intensity signals. To achieve this, it uses a multi-stage mathematical engine that handles everything from background subtraction to fold-change normalization.

## 📚 Deep Dive: Support Documentation

| Document | Description |
| :--- | :--- |
| **[Analysis & Mathematics](./docs/developer/01_ANALYSIS_AND_MATH.md)** | Nitty-gritty details on Rolling Ball baselines, AUC integration, and SNR logic. |
| **[UI Architecture](./docs/developer/00_UI_ARCHITECTURE.md)** | How the Wizard-based pipeline and ImageCanvas work under the hood. |
| **[User Guide](./docs/user/00_USER_GUIDE.md)** | Step-by-step instructions and a quick-reference for the new split/merge interactions. |

---

## Technical Core
- **Engine**: Python / NumPy / SciPy / Scikit-Image
- **UI**: PyQt6 / Matplotlib
- **Methodology**: 1D Vertical Projection + Rolling Ball Morphological Top-Hat

## Development
This module is built using the Karcytics Plugin Architecture. It implements the `WizardPanel` interface and manages its own internal state via the `AnalysisState` dataclass.

---

### Why use this over ImageJ?
ImageJ requires "subjective thresholding"—you decide where the baseline starts and ends by eye. Karcytics is an **objective judge**. It treats every pixel and every lane with the exact same geometric rules, making your data 100% reproducible for publication.
