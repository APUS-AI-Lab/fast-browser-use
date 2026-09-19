# Performance measurements

## Benchmark Configuration

All measurements are conducted with **100% local inference** on consumer-grade hardware:
- **Hardware**: Apple Silicon Mac (Apple M2 Pro, 32 GB unified memory, macOS)
- **Model & Weights**: `Qwen3.5-9B MLX 4-bit` (~5.95 GB memory footprint)
- **Inference Runtime**: MLX 0.32.2 / MLX-LM 0.31.3
- **Browser Environment**: Playwright Chromium (headed 1120×780, en-US)
- **Execution Mode**: Full-goal execution (`FBU_PLAN=0`, `FBU_REASONING=0`, `FBU_DECISION_MODE=auto`)

Weights are pinned to identical snapshot SHA-256 hashes across Hugging Face and ModelScope distributions; see [model-sources.json](model-sources.json). Model loading and initial page navigation precede task timing. Every subsequent decision, field generation, browser action, preparation wait, and guard check is included in the recorded task duration. No downloads or network inference occur during execution.

## Wikipedia Live Task Performance

Target: *"Find and open the Wikipedia article about Python, the programming language, starting from Main_Page."*  
Verification: Independent exact destination URL (`https://en.wikipedia.org/wiki/Python_(programming_language)`) and page title check (`Python (programming language) - Wikipedia`).

| Trial | Actual Task Time | Scoring Passes | Decisions | Executed Actions | Verification Check |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 30.079 s | 4 | 4 | 3 | Passed |
| 2 | 30.440 s | 4 | 4 | 3 | Passed |
| 3 | 30.091 s | 4 | 4 | 3 | Passed |
| **Median** | **30.091 s** | **4** | **4** | **3** | **3/3 Passed** |

- **Reflex Speed**: First browser action dispatched in **8.52 seconds**.
- **Compact Decision Path**: The entire live task is executed in **4 single-token forward passes** (Search focus → Enter query → Click main entry → Confirm completion), completely bypassing autoregressive code generation loops.
- **Bounded Settling**: Average settling wait of ~2.2 s across the task ensures the dynamic DOM is stable before scoring, eliminating stale or invalid action dispatches.

## Multi-Scenario Suite Performance

The same local Qwen3.5 runtime was evaluated across multiple distinct web scenarios. Each task runs in a fresh, isolated browser context and is verified by independent post-run assertions:

| Scenario & Task | Actual Task Time | Executed Actions | Independent Verification Check |
| :--- | :---: | :---: | :--- |
| **Workspace Settings Form** (Name, timezone dropdown, weekly digest toggle) | **12.002 s** | Fill, Select, Toggle, Save | Exact match on all three saved values in confirmation notice |
| **Reading-Room Article Navigation** | **5.430 s** | Search, Link Click | Exact match on target article URL and title |
| **Python.org Navigation** (Navigate to About page) | **8.095 s** | Nav menu hover & click | Exact match on destination `/about/` URL |
| **Example.com → IANA Information** | **7.309 s** | Anchor detection & click | Exact match on destination domain URL |

Full scenario specifications and outcome expectations are defined in [benchmarks/tasks.json](../benchmarks/tasks.json) and [benchmarks/public-tasks.json](../benchmarks/public-tasks.json). Telemetry records are preserved in [generality.json](generality.json).

## Demo Video & Telemetry

- **[8.32 s MP4 Preview (4× Playback)](demo.mp4)** · **[Telemetry Measurement (JSON)](measurement.json)**
- **[7.24 s Settings Form MP4 (2× Playback)](workspace-demo.mp4)** · **[Form Telemetry (JSON)](workspace-demo-measurement.json)**

The video preview uses the median Wikipedia run (**30.091 s actual task time**). The preview runs at **4× playback with no cuts**, visibly labeled on-screen. Setup time, verification, and an explicit one-second final hold account for total recording duration.

Playback acceleration is purely a preview convenience; reported task execution times always reflect actual unaccelerated wall-clock time.
