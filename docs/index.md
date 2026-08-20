# Western Blot Densitometry

A high-precision pipeline for quantifying Western Blot and Ponceau S stains, built as a
[Karcytics](https://github.com/KalaimaranB) plugin. It replicates the gold-standard ImageJ
densitometry protocol while removing human bias through automated lane mapping, rolling-ball
baseline subtraction, and objective peak-finding.

- **[User Guide](user/00_USER_GUIDE.md)** — step-by-step instructions for running an analysis.
- **[UI Architecture](developer/00_UI_ARCHITECTURE.md)** — how the Wizard-based pipeline and
  ImageCanvas work under the hood.
- **[Analysis & Mathematics](developer/01_ANALYSIS_AND_MATH.md)** — Rolling Ball baselines, AUC
  integration, and SNR logic.
