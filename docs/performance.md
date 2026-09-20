# Performance measurements

## Benchmark Configuration

All measurements use **100% local inference** on consumer-grade hardware. The original 9B baseline configuration:
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

## Qwen3.5-35B-A3B comparison (2026-09-20)

The same Wikipedia goal was recorded with local Qwen3.5-35B-A3B and Qwen3.5-9B MLX 4-bit weights.
Prompts, candidates, chat templates, waits, guards and the timing boundary are unchanged.

| Model | Trial 1 | Trial 2 | Trial 3 | Median task time |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 30.079 s | 30.440 s | 30.091 s | 30.091 s |
| Qwen3.5-35B-A3B | 19.152 s | 18.902 s | 18.834 s | **18.902 s** |

Qwen3.5-35B-A3B task time is **37.2% lower (1.59× faster)** than the 9B median (**18.902 s** vs. **30.091 s**).
All trials use a fresh process, model and headless Chromium context (1120×780, en-US), the same M2 Pro / 32 GB Mac
and runtime versions above. There is no task-specific warmup before timing.

A separate series reused one model process across three fresh browser contexts: **19.037 / 16.783 /
17.009 s**, all verified. After the first task, model residency and the invariant 186-token policy/goal
prefix are reused; the subsequent two tasks have a median of **16.896 s**. Both published 35B previews
select their three-run series' median.

Model loading and initial page observation precede the task; the first inference, all decisions,
field generation, actions and waits are included. Independent verification and the final one-second
hold follow the timed task. Model-load medians: **6.858 s (35B)** and **1.400 s (9B)**.
Median action-decision latency across trial medians: **3,012 ms (35B)** and **5,663 ms (9B)**.
Action counts: 3 / 3 / 3 for 35B, 4 / 4 / 4 for 9B.

All runs passed independent article URL (query string excluded) and exact title checks;
saved HTML title and canonical URL were also audited. Local model hashes were rechecked. Tiny random
dense/MoE tests compare candidate scores with full forward passes, without downloads.

- Fresh process: [7.32 s preview (3×, 18.902 s task)](qwen35b-demo.mp4) · [21.76 s original](qwen35b-demo-original.webm) · [Telemetry](qwen35b-demo-measurement.json)
- Reused model: [6.68 s preview (3×, 17.009 s task)](qwen35b-warm-demo.mp4) · [19.92 s original](qwen35b-warm-demo-original.webm) · [Telemetry](qwen35b-warm-demo-measurement.json)
- [All trials, source hashes and verified model checksums](qwen35b-comparison.json)

The complete original videos, traces and saved pages remain under ignored local directories.
Only reviewed public-site evidence is published; models and personal traces remain excluded from git.

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

- **[Recording Preview (3× Playback)](qwen35b-demo.mp4)** · **[Telemetry Measurement (JSON)](qwen35b-demo-measurement.json)**
- **[Settings Form MP4 (2× Playback)](workspace-demo.mp4)** · **[Form Telemetry (JSON)](workspace-demo-measurement.json)**

The video preview uses the median Wikipedia run with Qwen3.5-35B-A3B (**18.902 s actual task time**). The preview runs at **3× playback with no cuts**, visibly labeled on-screen. Setup time, verification, and an explicit one-second final hold account for total recording duration.

Playback acceleration is purely a preview convenience; reported task execution times always reflect actual unaccelerated wall-clock time.
