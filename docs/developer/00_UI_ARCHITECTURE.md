# 🏛️ Western Blot: UI Architecture

The Western Blot plugin is designed as a state-driven wizard. This architecture ensures that any change in the pipeline (e.g., re-running lane detection) automatically propagates to all downstream steps.

## 1. Unified State: `AnalysisState`

Everything related to the current analysis session is stored in a simple, serializable dataclass called `AnalysisState`. 

- **State Independence**: Each analyzer (Western Blot or Ponceau) owns its own `AnalysisState`.
- **State Persistence**: The state can be saved (via the "Save Workflow") and restored precisely, without recalculating.

## 2. Wizard & Pipeline Steps

The UI consists of a `WesternBlotPanel` which contains a `QStackedWidget`. Each step in the analysis is a `WizardStep`:

1.  **LoadStep**: Image loading and preprocessing (invert, contrast, rotate, crop).
2.  **LanesStep**: Detects vertical lane boundaries in the preprocessed image.
3.  **BandsStep**: Extracts 1D profiles, estimates the baseline, and detects peaks.
4.  **ResultsStep**: Normalization math and final densitometry.

## 3. Communication System (Signals)

Steps communicate via the parent `WesternBlotPanel` through dedicated `pyqtSignal` events:

- `image_changed`: Fired when invert, contrast, or rotation is updated.
- `lanes_detected`: Fired after a lane-detection pass or manual adjustment.
- `bands_detected`: Fired after the band-picking stage.
- `state_changed`: A global signal that triggers a refresh of the history/undo stack.

## 4. The Interactive Canvas: `ImageCanvas`

The `ImageCanvas` is the central component for user interaction. It uses a custom `QGraphicsScene` with dedicated items:

- **`LaneBorderItem`**: Draggable vertical lines for boundary refinement.
- **`BandOverlayItem`**: Selectable peaks that can be removed or added.
- **`ResizableCropItem`**: Handles spatial cropping on the "base image."

## 5. Visual Results: `ResultsWidget`

The results widget is a dynamic aggregator. It receives the final `pandas` DataFrame from the `ResultsStep` and renders:

- **Matplotlib Charts**: Visualizes the density comparison between samples.
- **Data Tables**: Allows for quick audit and CSV/Excel export.
- **Fold-Change Engine**: Automatically calculates relative density versus a control lane.
