# Design

The reference is [Jev Ultrafast](https://github.com/browser-use/jev-ultrafast).
The guarded DOM reader, action executor, fixture and inspector are adapted under MIT.
This project uses an isolated Playwright Chromium context and local MLX or PyTorch inference.
It does not wrap the `browser-use.Agent` class or call Jev/TypeSafe.

## Full-goal execution

Qwen3.5-9B is the supported model: MLX 4-bit on Apple Silicon, or original weights through PyTorch on CUDA/CPU. The default `FBU_PLAN=0` works directly on
the complete user goal with no generated checklist, task-specific action plan or prepared field text.
An optional diagnostic `FBU_PLAN=1` asks the same model for a bounded subgoal checklist. Its planner
uses the request and observed form labels on compact pages, or the request alone on large pages.

In checklist mode, a separate one-token binary check asks whether the current subgoal is complete. It keeps
the established compact context: page title/URL, current or touched control values, button labels and
actions for that subgoal. The optional full-goal binary check also includes the document language, visible text and all
control values. The optional checklist mode uses only its current subgoal for completion judgments.
Filled fields and landing-page descriptions are not proof of a submitted search. This is still a model judgment, with known
limitations on tasks whose completion requires understanding arbitrary page text.

When the check says continue, the action scorer chooses among currently legal actions. A tie continues;
there is no automatic advancement merely because an action was executed. Each completed subgoal must
receive the model's DONE judgment. Final demonstrated outcomes are checked by separate browser code.

With `FBU_PLAN=0`, the default `FBU_DECISION_MODE=auto` instead includes DONE in the same
one-token candidate selection as the complete observed browser actions. It reads the page once per
decision, without a preceding completion inference. Page prose, current values, recent actions,
candidate labels and operation/target indices remain intact. DONE still passes the full freshness
guard, and independently verified demonstrations still require the external outcome checks.
`FBU_DECISION_MODE=binary` restores the separate check; `joint` explicitly selects the joint mode.
Native thinking retains the binary path in all modes. Scores remain relative preferences, not
correctness probabilities or a basis for skipping guards.

Full-goal runs also compare successive read-only semantic snapshots before inference. The default
quiet interval is 150 ms, with a minimum 500 ms observation window on a new document and 300 ms
after filling. Loading stylesheets prevent early readiness; hidden recording clocks do not reset
the quiet interval. The default deadline is 1.5–2 seconds, so continuously changing pages cannot
block indefinitely. A deadline returns the latest observation with `settled=false`; it does not waive
execution guards. If navigation prevents every read, inference is deferred until a page can be observed.
All of this time, including initial settling, is inside the task timer and the original-speed video.
`FBU_SETTLE_MS=0` disables this preparation; an explicit nonzero value also enables it in checklist mode.
The ordinary checklist path keeps its existing post-input observation waits by default.

Traces record preparation latency and whether it settled, scoring-pass counts, and stale rejection
reasons with prediction indices. A post-input observation failure is marked `execution_recorded=true`
and is not counted as a rejected, unexecuted decision. The action remains in history and is not replayed.

`FBU_REASONING=1` enables optional native Qwen thinking for the completion check.
The tokenizer's chat template determines the native thought boundary. After the channel ends, a fixed `Action code:`
answer slot is constrained to the binary continue/complete codes. If the subgoal is incomplete,
the ordinary one-token scorer chooses a complete observed action. Thought text is never executed
or interpreted as selectors. If the native thought
channel does not finish within 2,048 tokens, the decision stops without an action. All generation time
is included in recordings.
This mode requires a supported single-token thought boundary and does not imply greater
reliability on every website.
The answer slot matters: scoring letters at an unconstrained prose boundary can mistake the initial
“I” of a sentence for candidate I, even with a very high candidate-normalized score.

## One action selection is a complete action

1. Atomically read visible DOM text and controls. Assign code-owned node identities.
2. Enumerate compatible `(operation, element, observed option)` tuples.
3. Assign unique one-token codes, verified against the actual model tokenizer.
4. Format the goal, recent history, current values and candidates using the model tokenizer's chat template.
5. Read next-token logits and normalize over the candidate codes. Select one complete tuple.
6. Revalidate its document, values, semantic context, geometry and occlusion before input.
7. Record execution before observing the resulting page. Never automatically retry a mutation.

`TYPE_TEXT` makes a separate call to the same local weights, with both the full user request and current
subgoal so a missing field can be recovered during a later step. The result must be
exactly a JSON object with one nonempty string `text`. The executor never prepares task-specific values.
`SELECT` uses the observed native option value, with no free-form generation.

## Caching and model compatibility

Both loaders validate Qwen3.5-9B architecture dimensions before loading weights. MLX additionally
requires the 4-bit configuration and uses the upstream MLX-LM text backend. PyTorch requires original,
unquantized weights and uses Transformers' Qwen3_5ForCausalLM text loader with local-only loading and
remote code disabled. It rejects missing/mismatched language tensors. Vision weights are excluded.
MLX and PyTorch have separate pinned Hugging Face revisions; the ModelScope mirror is MLX-only.

`FBU_BACKEND=auto` selects MLX on Apple Silicon and PyTorch elsewhere. The `torch` extra installs the
optional PyTorch dependencies. `FBU_DEVICE=auto` selects available CUDA or CPU; `cuda:N` selects one
visible GPU. Automatic precision is BF16 on supported CUDA devices, FP16 on other CUDA devices, and
FP32 on CPU. Explicit unsupported/unavailable devices fail instead of silently moving inference.
The PyTorch backend scores only candidate token IDs in float32 after a single prompt forward pass,
projecting only the last hidden state into vocabulary logits. It currently uses no prefix cache across
decisions (`cached_tokens=0`, `cache_hit=false`). Text, planning and optional thinking use greedy local
decoding with a fresh hybrid attention cache per call. Both backends share the tokenizer chat template,
JSON parsing, candidate-code validation and browser guards. Existing MLX timings do not describe PyTorch.

In MLX, the policy and current subgoal form an invariant prefix. Completion checks and action scoring use separate cache slots. A cached prefix is reused only when its token IDs match
exactly. Each decision gets a deep copy of the complete model cache, including rotating and shared-KV
structures supported by MLX. Dynamic page tokens are always evaluated again. Planning and text generation have their own fresh caches. New observations do not reuse old DOM action mappings.

This avoids the original RLCD prototype's independent-field mismatch, shared-first-token ambiguity,
and fabricated confidence floor. Candidate softmax is a relative model preference, not calibrated
probability of success. The inspector labels these values as scores.

## Boundaries

- The MVP handles visible HTML/ARIA clicks, text input, native selects, page and container scrolling, and waits.
  Visible labels can operate styled native checkboxes. Each scroll action names an observed region;
  the executor checks its hit target before sending a wheel event. Covered text and controls are excluded.
- No screenshots enter the model. Screenshots and video are for the human inspector and evidence.
- No iframe traversal, shadow-root traversal, canvas grounding, file upload, or new-tab orchestration.
- A `DONE` choice is a model claim. A task-specific independent check is required to claim success.
- No site-specific plans are encoded in the policy. The shipped demo uses an explicit natural-language goal naming visible controls. Fixtures and verifiers live outside it.
- Prefix caching does not eliminate prefill of new page content. A short output alone does not imply
  a sub-100 ms step, and navigation time remains part of task latency.
- Model load/download time is separate from measured warm task time.

The local inspector binds to loopback, checks Host and Origin, and requires a per-process token for
mutations. One worker owns Playwright and serializes all commands. Browser sessions are isolated
from the user's everyday profile.

## Headless recording

`fbu record` explicitly launches headless Chromium, overriding the interactive `FBU_HEADLESS=0`
setting. Playwright records browser frames directly into WebM, with no desktop, DISPLAY or Xvfb.
Blocking local inference and waits remain in the original timing. Context close finalizes the video
before it is saved as `browser.webm`. Tests use a local page, simulated blocking inference, independent
success/failure checks and ffprobe to verify the original duration. No test downloads model weights.
Runtime metadata includes only installed distributions, plus the selected backend, device and dtype.
Preview labels use this metadata rather than assuming MLX or Apple hardware. Rendering a labeled
accelerated preview requires system ffmpeg/ffprobe and preserves the original recording.
