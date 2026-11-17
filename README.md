# sensenova-loop-rs

> **SenseNova-Loop-9B 的独立 Rust 推理引擎**：CPU（AVX2 / AVX-512-VNNI）+ AMD OpenCL GPU 后端，OpenAI 兼容服务，支持 prefix caching。
>
> 现有运行时（llama.cpp / LM Studio / Ollama）都还加载不了这个架构，所以这个项目从零实现了它。

---

## 为什么有这个项目

商汤（SenseTime）2026 年 11 月开源的 **SenseNova-Loop-9B** 不是常规 Transformer：

- **Looped Decoder**：只存 32 组解码层权重，但第 4–28 层以共享权重连跑两遍，一次前向完成 **56 次层计算**（9B 参数、约 14.6B 等效算力）
- **mHC（Manifold-Constrained Hyper-Connections）**：循环段内携带两条残差流，用 Sinkhorn 归一化的双随机矩阵做流间路由
- **混合注意力**：每 4 层里 3 层 GatedDeltaNet 线性注意力 + 1 层全注意力，partial RoPE（25%）

llama.cpp 对该架构的支持仍在开放中（[ggml-org/llama.cpp#31571](https://github.com/ggml-org/llama.cpp/issues/31571)），因此 GGUF 量化件没有任何现成运行时能加载。这个项目补上了这块：**直接加载社区 GGUF（Q4_0），在 CPU 或 AMD 核显上推理**。

## 特性

- **完整架构实现**：56 槽执行计划、mHC 双流残差 + Sinkhorn 路由、GatedDeltaNet、partial RoPE
- **量化推理**：Q4_0 权重常驻内存，全程不反量化成 f16/f32（9B 模型仅 ~5.1 GB）
- **CPU 内核**：decode 走 AVX2 f32；prefill 走 AVX-512 VNNI int8（激活按 32 元素量化）
- **AMD GPU 后端**：OpenCL 2.0，自写 FFI（只用 `libloading` 取符号，**不需要 AMD SDK/导入库**），权重零拷贝常驻显存
- **OpenAI 兼容服务**：`/v1/models`、`/v1/chat/completions`（含 SSE 流式），可直接接 DSH / Cursor / Claude Desktop 等
- **Prefix caching**：多轮对话只 prefill 增量部分

## 实测性能

机器：Ryzen AI 9 HX 370（12C/24T）+ Radeon 890M 核显，LPDDR5-5600 双通道（实测整机顺序读带宽 **50.9 GB/s**），SenseNova-Loop-9B Q4_0。

| 场景 | CPU | GPU prefill | 说明 |
|---|---|---|---|
| prefill 401 token | 43.5 s (9.2 tok/s) | **17.4 s (23.1 tok/s)** | 2.5× |
| prefill 1901 token | 202 s (9.4 tok/s) | **67.0 s (28.4 tok/s)** | 3.0× |
| decode（短上下文） | **4.0 tok/s** | 3.5 tok/s | decode 留 CPU 更快 |
| 单次 matmul 12288×4096 | 0.88 ms | 0.83 ms | m=1 时受带宽限制 |
| prefix cache：2440 token 前缀，第 2 轮 | 62.6 s | — | **4.4 s（14×）** |

**为什么 decode 不用 GPU**：模型每生成 1 token 要流约 9.3 GB 权重（循环层跑两遍），而核显与 CPU 共享同一块内存 —— 换 GPU 不增加带宽。实测 GPU GEMV 只有 35–40 GB/s，CPU 已跑到约 46 GB/s。所以默认策略是 **prefill 用 GPU（算力瓶颈）、decode 用 CPU（带宽瓶颈）**。

## 快速开始

### 1. 取模型权重

本项目**不包含**模型权重。需要：

- GGUF（Q4_0）：社区转换件，例如 Hugging Face 上的 `guideboardlabs/SenseNova-Loop-9B-GGUF`
- tokenizer：取官方仓库的 `tokenizer.json`（`sensenova/SenseNova-Loop-9B-Preview` 未 gated，可直接下）

把 `.gguf` 和 `tokenizer.json` 放在任意目录，运行时用参数指定。模型权重遵循 SenseTime 的 Apache-2.0 许可。

### 2. 构建

```bash
# CPU 版
cargo build --release

# 含 GPU 后端（Windows + AMD 驱动自带 OpenCL 即可）
cargo build --release --target-dir target-gpu
```

要求：Rust 1.80+；GPU 需要 OpenCL 2.0 运行时（AMD Adrenalin 驱动自带）。

### 3. 命令行推理

```powershell
# 纯 CPU
.\target\release\sensenova-rs.exe `
  --model  path\to\SenseNova-Loop-9B-Q4_0.gguf `
  --tokenizer path\to\tokenizer.json `
  --prompt "The capital of France is" --max-new 16

# GPU prefill + CPU decode（推荐）
$env:SENSENOVA_GPU_PREFILL_ONLY = '1'
.\target-gpu\release\sensenova-rs.exe --model ... --tokenizer ... --prompt "..." --gpu
```

常用开关：

| 开关 | 作用 |
|---|---|
| `--chat` | 套用官方 chat template（默认关闭思考） |
| `--thinking` | 开启 `<think>` 思考段 |
| `--serve --port 8090` | 起 OpenAI 兼容服务 |
| `--gpu` | 启用 OpenCL 后端 |
| `--prefix-cache` | 服务模式下复用上轮前缀的 KV/GDN 状态 |
| `--bench` / `--bench-gpu` | 内置基准 |
| `--selfcheck` | prefill 与逐步 decode 的 logits 一致性自检 |
| `--topk N` | 打印首个位置的 top-N 候选 |

### 4. OpenAI 兼容服务

```powershell
$env:SENSENOVA_GPU_PREFILL_ONLY = '1'   # prefill 上 GPU，decode 留 CPU
.\target-gpu\release\sensenova-rs.exe `
  --model  path\to\SenseNova-Loop-9B-Q4_0.gguf `
  --tokenizer path\to\tokenizer.json `
  --serve --port 8090 --gpu --prefix-cache
```

端点：`http://127.0.0.1:8090/v1`，模型 id `sensenova-loop-9b`。也可用 `scripts/run-server.ps1`（参数化路径）。

## 架构实现要点

如果需要复刻，这四处是最容易踩错的地方（都是和 HF 参考实现/GGUF 约定相关的坑）：

1. **attention 的 softmax 轴**：必须对 key 维（最后一维）归一化。写错成 query 维时，decode 阶段 `t=1` 会让 softmax 退化成全 1，输出变成所有 value 直接相加。
2. **`ssm_a` 已经是 `-exp(A_log)`**：GGUF 按 llama.cpp 约定预存了负衰减率，不能再取一次 `exp`。
3. **GDN 的 q/k 头是 tile 布局**：GGUF 里按 `[k0..k15, k0..k15]` 排布，不是 HF 源码里的 `repeat_interleave`。（用错时模型仍能生成通顺英文，但答不对事实题——"巴黎"会变成"法国大革命"。）
4. **norm 权重已被转换脚本加过 1.0**：按普通 RMSNorm 直接用，不要再 `w + 1`。

另外，执行计划由 56 个「槽」组成，权重只存 32 组（`blk.0..27` + `blk.52..55`），每个槽需要独立的 KV / 卷积 / 递推状态 —— 这也是 prefix caching 必须按槽快照的原因。

## 限制

- **纯文本**：视觉塔（`mmproj`）未实现，图片输入不支持
- **无 YaRN 缩放**：官方标称 512K 上下文依赖 YaRN；本实现按 HF 默认 RoPE 走，建议控制在 32K 以内
- **单流**：服务端串行处理请求，无批处理
- **速度受内存带宽限制**：9B / 56 层计算的架构在共享内存的轻薄本上，decode 很难超过 ~6 tok/s

## 许可

代码：Apache-2.0（见 `LICENSE`）。模型权重版权归 SenseTime，遵循其 Apache-2.0 许可；本项目不分发权重。

## 致谢

- [SenseTime SenseNova](https://github.com/sensenova) —— 模型与 `modeling_sensenova_loop.py` 参考实现
- [candle](https://github.com/huggingface/candle) —— 张量运行时
- HuggingFace `transformers` 的 `qwen3_5` 实现 —— 骨干参考
- ggml 的 Q4_0 量化格式与 int8 点积思路

---

## English summary

A standalone Rust inference engine for **SenseNova-Loop-9B** (Looped Transformer with mHC hyper-connections and GatedDeltaNet hybrid attention). No existing runtime can load this architecture yet — llama.cpp support is still pending ([#31571](https://github.com/ggml-org/llama.cpp/issues/31571)) — so the full forward pass, GGUF Q4_0 loader, AVX2/AVX-512-VNNI CPU kernels, an OpenCL GPU prefill backend, an OpenAI-compatible server and prefix caching are implemented here.

Measured on Ryzen AI 9 HX 370 + Radeon 890M: prefill 1901 tokens in **67 s** with GPU (3.0× over CPU), decode **4 tok/s** on CPU, and prefix caching cuts a 2440-token second turn from 62.6 s to **4.4 s**. GPU is used for prefill only — the iGPU shares system memory, so decode is bandwidth-bound and faster on the CPU.

License: Apache-2.0. Model weights are not redistributed.
