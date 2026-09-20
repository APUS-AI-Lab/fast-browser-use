<div align="center">

<img src="docs/banner.svg" alt="Fast Browser Use — 真实浏览器操作，基于本地模型" width="100%" />

# Fast Browser Use

**面向 Claude Code、Codex、OpenCode 等 Agent 的端侧极速“系统 1”浏览器自动化引擎与 Agent Skill。**  
*基于 Qwen3.5-9B，通过 MLX 或 PyTorch（CUDA / CPU）本地运行。零云端推理、秒级反射决策、从结构上彻底杜绝选择器幻觉。*

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Cross-platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-black.svg)](#linuxwindowsgpu-服务器pytorch)
[![Model: Qwen3.5-9B-4bit](https://img.shields.io/badge/Model-Qwen3.5--9B--4bit-purple.svg)](https://huggingface.co/Qwen/Qwen3.5-9B)
[![Cloud Inference: None](https://img.shields.io/badge/云端推理-零调用%20(%20100%25%20离线本地%20)-orange.svg)](#-本地端侧架构)
[![Agent Skill: Claude Code & Codex](https://img.shields.io/badge/Agent%20Skill-Claude%20Code%20%7C%20Codex-brightgreen.svg)](skills/fast-browser-use/SKILL.md)

[English](README.md) · [简体中文](README.zh-CN.md) · [快速上手](#-快速上手) · [核心架构与技术拆解](#-核心架构与技术拆解) · [实测基准数据](#-实测基准数据) · [Agent Skill 安装](#-agent-skill-安装配置) · [Python API](#-python-api)

</div>

---

<div align="center">
<a href="docs/demo.mp4"><img src="docs/demo.gif" alt="本地 Qwen3.5 浏览器真实操作，明确标注为 4x 倍速播放" width="100%" /></a>

**[8.32 秒 MP4 预览（4× 倍速播放）](docs/demo.mp4)** · **[实测遥测数据（JSON）](docs/measurement.json)** · **[性能基准与测试协议](docs/performance.md)**
</div>

> **Jev 离散决策范式的开源逆向实现 · 纯本地 Browser-Use Agent Skill**  
> 本项目基于开源模型（Qwen3.5-9B）复刻 Jev 的“系统 1”离散打分思想，以浏览器自动化（Browser-use）为落地场景。通过将可见交互原子映射为 Token 进行单步 Logits 打分，从机制上彻底杜绝选择器幻觉，并封装为开箱即用的 Agent Skill，支持纯本地模型离线执行。

---

## 💡 背景：Jev 的开源逆向工程与 Browser-Use 落地

2026 年 9 月中旬，TypeSafe AI 推出了 **Jev** 模型（由前 OpenAI 团队成员 Diogo Almeida 联合创立），在社区引发了关于智能体决策范式的广泛讨论：

> *“大家习惯了把所有事情都交给千亿、万亿参数的云端大模型一个 Token 一个 Token 自回归生成，不仅响应慢、成本高，而且频繁出现幻觉——面对大量确定性、有边界的自动化操作，难道没有更好的解法吗？”*

TypeSafe AI 由此提出了“系统 1”决策模型的理念：将 GPT、Claude 这类擅长复杂推演与长程规划的模型归为慢思考的“系统 2”，而将局部、高频且边界明确的即时判定交给轻量的“系统 1”模型。Jev 通过对类型化有界空间直接投影打分，替代自由文本自回归生成，宣称决策速度最高可达传统大模型的 20~200 倍。

目前 Jev 主要作为云端 API 服务提供，未公开模型权重与具体实现细节。开源社区此前的探索（如 [`jev-ultrafast`](https://github.com/browser-use/jev-ultrafast)）主要通过调用其云端接口来体验这一决策范式。而 Fast Browser Use 则进一步探索将离散决策机制完全迁移到本地设备上，借助开源权重与本地端侧算力，实现完全离线运行、零云端成本与数据不出本地的浏览器自动化体验。

### 基于开源模型的 100% 本地端侧落地

**Fast Browser Use 是对 Jev 底层决策范式的一次开源逆向工程实现，并将其完整落地于最具代表性的浏览器自动化（Browser-use）场景。**

我们基于 Jev 公开文档展示的输入范式与评估逻辑，以及借鉴了一些开源项目的思路，逆向拆解了其“跳过自回归解码、隐状态直接打分”的核心逻辑。为了检验这套思路在没有闭源权重的前提下是否可行，我们以本地开源模型 **Qwen3.5-9B** 与消费级 Apple Silicon（MLX）为底座，构建了端到端的浏览器自动化 Agent Skill：

- **从生成代码到受限单选**：传统 Browser-use 方案让大模型编写 Playwright 脚本或猜测选择器，极易在动态页面上产生“选择器幻觉”。Fast Browser Use 在宿主端原子化扫描当前渲染树中**真实可见、可交互**的元素，组装为离散候选元组 `(CLICK, btn_7)`、`(SELECT, opt_2)`。模型只在客观存在的候选中做单选，从结构上彻底杜绝选择器幻觉；
- **单 Token Logits 秒级反射（$O(1)$ 决策）**：每个合法候选动作映射到模型词表中的独立单个 Token，单步前向传播提取 Next-token Logits 并做 Softmax 归一化（**单步秒级即时响应**），彻底省去多秒的自回归文本解码；
- **动作判定与文本生成解耦**：结构化点击/选择走离散 Logits 打分；仅在真正需要输入文字（`TYPE_TEXT`）时，才调用同一模型生成字段文本，算力分配高度聚焦；
- **100% 本地端侧运行**：零云端 API 调用、零外部依赖与订阅费，在消费级 Mac 上离线运行，敏感网页数据永不出机。

---

## 📊 架构对比：传统云端 Agent vs. Fast Browser Use

| 维度 | 传统云端多模态 Agent | Fast Browser Use（本地系统 1） |
| :--- | :--- | :--- |
| **推理位置** | 云端商业 API（OpenAI / Anthropic 等） | **100% 本地运行**（MLX 或 PyTorch CUDA / CPU） |
| **单步交互延迟** | 3,000 – 8,000 ms（网络往返 + 逐字解码） | **秒级响应**（纯 Logits 投影前向） |
| **决策模式** | 自由文本 / 自回归 JSON 代码生成 | **受限离散候选中做单选**（单 Token） |
| **选择器可靠性** | 频繁出现选择器幻觉（找不到元素） | **从结构上零幻觉**（仅扫描可见真实 DOM） |
| **语法解析成功率** | 存在漏括号、少引号等 JSON 语法崩塌 | **100% 语法合法**（确定性宿主代码组装） |
| **数据隐私** | 完整网页正文与操作截图上传云端 | **100% 离线端侧**，敏感数据不出本地设备 |
| **推理费用** | 随 Token 与截图步数不断计费 | **$0.00**（利用本地算力，完全免费） |
| **宿主协同** | 孤立单体执行脚本 | **标准 Agent Skill**，无缝接入 Claude Code / Codex |

---

## 🧠 智能体分工：系统 1（快反射）与 系统 2（慢大脑）

Fast Browser Use 并不是要取代通用大模型，而是让 Agent 体系实现清晰的**快思考与慢思考层次分工**：

<img src="docs/02-action-selection.svg" alt="从生成动作文本到选择完整候选" width="100%" />

- **系统 2（上层主脑：Claude Code、Codex、Cursor、Antigravity 等）**：  
  负责理解用户长程业务目标、多文件逻辑推演、拆解宏观任务，并在执行完成后进行最终客观审计；
- **系统 1（端侧执行引擎：Fast Browser Use）**：  
  作为本地 Skill，负责局部页面的高频感知、动作单选打分、表单填写与原子化页面交互；
- **独立审计闭环**：  
  执行结束后，Fast Browser Use 向宿主返回执行追踪快照（Trace）。主脑 Agent 对页面最终状态进行严格断言核验，再决定下一步流程。

通过这种分工，上层主脑无需为几十步琐碎的页面点击支付昂贵的云端 Token 费用和漫长的往返等待，用户的私密业务数据也始终留在本地内存中。

---

## 🔬 核心架构与技术拆解

在没有闭源模型权重的情况下，Fast Browser Use 如何在本地实现极速与零幻觉？核心是将原本发散的文本生成任务重构为**封闭有界的分类决策问题**：

<img src="docs/01-jev-parallel-decision.svg" alt="Jev 机制拆解：KV-Cache 广播与并行单步决策" width="100%" />

### 1. 从结构上彻底消除“选择器幻觉”
宿主通过轻量脚本原子化扫描当前页面的渲染树，仅提取**当前真实可见、可点击、可交互**的元素，格式化为严格合法的动作元组：
```python
("CLICK", "btn_search")
("SELECT", "opt_timezone_sg")
("TYPE_TEXT", "input_query")
("DONE", "task_completed")
```
模型只能在这些已存在的元组索引中做单选。**由于模型根本不编写选择器或代码，从数学与物理上杜绝了选择器幻觉与执行脚本错误。**

### 2. 单 Token Logits 快速决策，跳过自回归解码
系统将每个合法候选动作映射为模型词表中的单个特定 Token（如 `A`、`B`、`C`...，均通过实际 Tokenizer 校验）。  
模型仅需进行**一次单步前向传播（Forward Pass）**，提取下一个 Token 的 Logits，并在候选 Token 集合上做一次 Softmax 归一化：
$$P(c_i \mid \text{Context}) = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$
计算复杂度从 $O(\text{Tokens} \times \text{Layers})$ 骤降为 $O(1)$ 的单步前向，单步决策耗时大幅缩减至秒级。

### 3. 动作选择与内容生成的合理解耦
网页中绝大多数操作（点击、展开下拉、滚动）无需文字创作。系统对交互进行了明确解耦：
- 结构化动作直接走 **单 Token 离散评分**；
- 只有当命中的动作确实是文本输入（`TYPE_TEXT`）时，才调用同一本地模型生成对应的字段输入文本，最大化节省端侧算力。

### 4. KV-Cache 广播与并发批量评估
对于多字段决策场景，单次计算 Context 的基础 KV-Cache，通过沿 Batch 维度广播并拼接预编译字段后缀，单次并发前向即可完成多字段评分，彻底切断了自回归误差放大链条。

---

## 🛡️ Harness 守卫与闭环执行工程

端侧小模型必须配合严谨的 Harness 守卫工程。Fast Browser Use 构建了多层防御体系：

<img src="docs/03-execution-loop.svg" alt="受守卫的闭环执行循环" width="100%" />

1. **有界语义稳定观察窗口（Bounded Settling）**：现代单页应用（SPA）具有动态渲染延迟。系统引入 150ms 静默窗口（新文档 ≥500ms，输入后 ≥300ms），在控件完全挂载后再触发推理，避免无效决策；
2. **候选动作与 `DONE` 合并打分**：读取一次页面即可同时评估下一步动作或结束任务，单任务打分 Pass 从 14 次大幅精简至 4 次；
3. **多重前置物理守卫**：
   - **可见性与遮挡校验**：确认目标控件未被浮层弹窗、Mask 遮挡；
   - **DOM 过期性校验**：确认快照与实时渲染树一致，节点未发生脱落；
   - **只读约束防护**：拦截不可编辑或已禁用的控件；
4. **单向记录与严禁盲目重试**：所有动作在派发前先记录到 Trace。**已派发的变动操作绝不自动盲目重试**，以防止外部表单出现重复提交或数据破坏；
5. **独立业务结果审计**：模型输出的 `DONE` 仅代表其主观判定，必须由外部调用方通过 `--expect-url`、`--expect-title`、`--expect-text` 进行客观严谨的最终断言验证。

---

## ⚡ 实测基准数据

以下数据均在本地消费级设备上**100% 离线完成**（无任何云端推理请求）：
- **运行设备**：Apple Silicon Mac（Apple M2 Pro，32 GB 统一内存，macOS）
- **推理后端**：MLX 0.32.2 / MLX-LM 0.31.3
- **本地权重**：`Qwen3.5-9B MLX 4-bit`（显存占用约 5.95 GB）

### 1. Wikipedia 维基百科真实任务实测

任务目标：*“从英文首页出发，检索并打开介绍 Python 编程语言的词条，严格校验最终 URL 与页面标题。”*

| 测试轮次 | 实际任务耗时 |
| :---: | :---: |
| 第 1 次 | 30.079 秒 |
| 第 2 次 | 30.440 秒 |
| 第 3 次 | 30.091 秒 |
| **中位数** | **30.091 秒** |

*首个动作发出仅需约 8.5 秒，全流程仅需 4 次单 Token 快速打分即可完成全目标操作。*

### 2. 多场景实测性能汇总

| 任务用例 | 实际任务耗时 | 包含动作 | 独立业务断言 |
| :--- | :---: | :---: | :--- |
| **工作区偏好设置表单**（名称、时区、开启周报） | **12.002 秒** | 输入、下拉选择、复选框勾选、保存 | 严格核验最终提示文本中的三项保存值 |
| **本地阅读室文章导航** | **5.430 秒** | 列表检索、链接点击 | 精确匹配目标 URL 与文章标题 |
| **Python.org 官网导航**（跳转 About 页面） | **8.095 秒** | 导航栏识别、跨页跳转 | 精确匹配目标页面 URL (`.../about/`) |
| **Example.com → IANA 信息页导航** | **7.309 秒** | 锚点识别、域名跳转 | 精确匹配目标页面 URL |

---

## 🚀 快速上手

### 环境要求
- **硬件**：MLX 使用 Apple Silicon；PyTorch 使用 NVIDIA GPU 或 CPU，内存要求见下文。
- **系统与运行环境**：Linux、Windows 或 macOS，Python 3.12+， [uv 包管理器](https://docs.astral.sh/uv/getting-started/installation/)
- **依赖工具**：Node.js / npm（用于 `npx skills` 安装），Git

---

### 📦 Agent Skill 安装配置（Claude Code / Codex）

本项目符合社区通用 [Agent Skills 规范](https://github.com/vercel-labs/skills)。

#### 第一步：注册 Agent Skill
```bash
# 全局注册给 Claude Code 与 Codex（-g 代表跨项目可用，-y 跳过交互）
npx skills add APUS-AI-Lab/fast-browser-use --skill fast-browser-use -a claude-code -a codex -g -y
```

#### 第二步：全局安装运行环境与模型权重

以下为 Apple Silicon / MLX 安装方式；其它平台请使用下方 PyTorch 配置。
```bash
# 1. 在独立隔离环境中全局安装 fbu 命令
uv tool install --python 3.12 "git+https://github.com/APUS-AI-Lab/fast-browser-use.git"

# 2. 安装匹配的 Playwright Chromium 浏览器内核
fbu install-browser

# 3. 下载并缓存固定的 Qwen3.5-9B 4-bit 本地权重（约 5.95 GB）
fbu download
```
*(若本地已有同规格 Qwen3.5-9B 4-bit 权重，可设置 `export FBU_MODEL=/path/to/weights` 直接跳过下载。)*

#### 第三步：在宿主 Agent 中直接调用
在任意项目的终端会话中启动宿主 Agent：

**在 Claude Code 中：**
```text
/fast-browser-use 打开 https://en.wikipedia.org/wiki/Main_Page，找到 Python 编程语言的词条，并验证最终 URL 和标题。
```

**在 Codex 中：**
```text
$fast-browser-use 打开 https://en.wikipedia.org/wiki/Main_Page，找到 Python 编程语言的词条，并验证最终 URL 和标题。
```

---

### Linux、Windows、GPU 服务器（PyTorch）

`torch` 可选依赖包含 PyTorch、Transformers 和 Accelerate。默认 `FBU_BACKEND=auto` 在
Apple Silicon 上选择 MLX，其它平台选择 PyTorch；可通过 `--backend torch` 显式指定。

```bash
# 在远程 Linux GPU 服务器的项目目录内执行
uv sync --locked --extra torch --python 3.12
uv run fbu install-browser --with-deps
uv run fbu download --backend torch
uv run fbu record --backend torch --device cuda --scenario wikipedia
```

`--with-deps` 安装 Chromium 的 Linux 系统依赖，可能需要 root/sudo 权限。操作和录屏无需桌面、
`DISPLAY`、Xvfb、VNC 或 inspector。原速 `browser.webm`、截图、trace 和独立验证结果保存在服务器上。

需要全局 CLI 时：

```bash
uv tool install --python 3.12 'fast-browser-use[torch] @ git+https://github.com/APUS-AI-Lab/fast-browser-use.git'
fbu install-browser --with-deps  # Windows/macOS 去掉 --with-deps
fbu download --backend torch
```

PyTorch 使用固定版本的原版 [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)，通过
[Transformers 纯文本加载器](https://huggingface.co/docs/transformers/model_doc/qwen3_5) 运行。
**MLX 4-bit 权重不能用于 PyTorch**；切换后端时清除旧 `FBU_MODEL`，或将其指向匹配的本地权重目录。
PyTorch 从 Hugging Face 下载；现有 ModelScope 镜像仅用于 MLX。下载后设置 `HF_HUB_OFFLINE=1`
可禁止后续 Hub 访问，访问在线网页仍需网络。

| 参数 | 默认值与可选值 |
| :--- | :--- |
| `FBU_BACKEND` / `--backend` | `auto`、`mlx`、`torch` |
| `FBU_DEVICE` / `--device` | `auto` 优先使用可用 CUDA，否则 CPU；支持 `cpu`、`cuda`、`cuda:N` |
| `FBU_DTYPE` / `--dtype` | `auto` 在 CUDA 上优先 BF16，否则 FP16；CPU 默认 FP32。可指定 `bfloat16`、`float16`、`float32` |
| `FBU_MODEL` | 当前后端的固定版本仓库，或匹配的本地权重目录 |

CUDA 使用指定的单张 GPU，编号遵循 PyTorch 可见设备（包括 `CUDA_VISIBLE_DEVICES`）。如果无法识别 GPU，
请安装[与驱动匹配的 PyTorch 构建](https://pytorch.org/get-started/locally/)。Windows PowerShell 可使用相同 CLI，
设置环境变量的语法为 `$env:FBU_BACKEND='torch'`。

9B 的 BF16/FP16 权重本身约需 18 GB，另需缓存和计算空间；24 GB GPU 可作为起点，但不保证所有页面均可容纳。
CPU 默认 FP32，权重本身约需 36 GB，速度也会慢很多。当前 PyTorch 后端不加载 4-bit 权重，也不跨 GPU 分片。
无需安装可选 DeltaNet 加速内核。现有性能数据均来自 MLX；PyTorch 每次候选打分重新计算提示，不跨决策复用前缀缓存。

### 💻 独立 CLI 运行与 CI 业务断言

你可以直接在终端或自动化流水线中使用 `fbu run`，并通过 `--expect-*` 传入独立业务断言：

```bash
# 导航任务与严格结果断言
fbu run https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Python, the programming language.' \
  --expect-url 'https://en.wikipedia.org/wiki/Python_(programming_language)' \
  --expect-title 'Python (programming language) - Wikipedia' \
  --trace artifacts/wikipedia_trace.json
```

```bash
# 复杂表单填写与多重文本断言
fbu run 'https://target.example/settings' \
  --goal 'Save workspace preferences with timezone Asia/Singapore and weekly digest enabled.' \
  --expect-title 'Preferences saved' \
  --expect-text 'Timezone: Asia/Singapore.' \
  --expect-text 'Weekly digest: enabled.' \
  --trace artifacts/preferences.json
```

- `--expect-url` 与 `--expect-title`：对最终页面状态进行完全精确匹配；
- `--expect-text`：匹配页面正文关键文本（可多次重复传入）；
- 断言均基于操作结束后重新抓取的页面快照，**绝不提前泄露给模型**；未满足断言将直接以非零状态码退出。

<details>
<summary>从本地源码安装 / 不使用 Node.js</summary>

```bash
git clone https://github.com/APUS-AI-Lab/fast-browser-use.git
cd fast-browser-use
uv tool install --python 3.12 .
fbu install-browser
fbu download
```

若不使用 Node.js，可直接创建软链接：
```bash
mkdir -p "$HOME/.claude/skills" "$HOME/.agents/skills"
ln -s "$PWD/skills/fast-browser-use" "$HOME/.claude/skills/fast-browser-use"
ln -s "$PWD/skills/fast-browser-use" "$HOME/.agents/skills/fast-browser-use"
```

</details>

<details>
<summary>多源模型下载：ModelScope 支持与离线模式</summary>

国内开发者可直接从 ModelScope 下载预转格式权重：
```bash
uv sync --extra modelscope
uv run fbu download --source modelscope --output models/Qwen3.5-9B-4bit
FBU_MODEL=models/Qwen3.5-9B-4bit fbu run https://www.python.org/ \
  --goal 'Open the About Python page.' --expect-url 'https://www.python.org/about/'
```

模型准备完毕后，可设置离线环境变量禁止任何网络下载：
```bash
export HF_HUB_OFFLINE=1
```

</details>

---

## 🐍 Python SDK

在 Python 代码中以编程方式调用 Fast Browser Use：

```python
from fast_browser_use import Agent
from fast_browser_use.model import get_model

# 预热并加载本地权重
get_model()

# 执行任务并获取流式执行状态
with Agent("https://example.com", "Open the More information link.") as agent:
    for state in agent.run():
        print(f"[{state['elapsed_ms']}ms] 当前状态: {state['status']}")
        if "action" in state:
            print(f"  执行动作: {state['action']}")
```

---

## 🎥 录屏演示与复现

录屏固定使用 headless Chromium，即使设置了调试用的 `FBU_HEADLESS=0` 也不会打开窗口。
Playwright 直接采集浏览器画面，无需桌面录屏。原速视频保留真实等待与推理耗时；
生成预览需额外安装 `ffmpeg`/`ffprobe`，原始录屏无需该系统命令：

```bash
# 执行录屏场景（保留 1x 原始视频与全量 telemetry）
uv run fbu record --scenario wikipedia

# 渲染带倍速角标的短预览（目标 <= 10 秒，保留原始录屏）
uv run python scripts/render_demo.py artifacts/recordings/<timestamp> --max-seconds 10
```

---

## 🛠️ 工程开发与自动化校验

在提交代码前，执行完整的本地测试与安全守卫核验：

```bash
uv run ruff check .
uv run pytest
# 使用随机初始化的小模型验证 PyTorch，不下载预训练权重：
uv run --extra torch pytest tests/test_torch_backend.py
node --check fast_browser_use/static/app.js
node --check fast_browser_use/snapshot.js
uv run python scripts/check_guards.py
uv build
```

---

## 📄 开源协议与致谢

本项目采用 [MIT 开源协议](LICENSE)。  
本项目灵感源于 [Jev Ultrafast](https://github.com/browser-use/jev-ultrafast)，浏览器执行Harness在其基础上改进。同时借鉴了 [openjev](https://github.com/TheoLeeCJ/openjev) 与 [Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) 的思想与设计。详细归属说明请见 [NOTICE](NOTICE)。

---

<div align="center">
<b>Fast Browser Use</b> · 面向下一代自主端侧 Agent 的极速“系统 1”基建
</div>
