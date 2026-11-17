# SenseNova-Loop-9B：AMD Radeon 890M（OpenCL）GPU 后端 + Prefix Caching

日期：2026-11-18
机器：AMD Radeon 890M（gfx1150，RDNA3.5 核显，与 CPU 共享 LPDDR5，实测整机读带宽 ~50–64 GB/s）
驱动：OpenCL 2.0 AMD-APP (3683.0)；Vulkan 1.4.359 也在，但未采用（见下）

---

## 1. 后端选型：OpenCL 2.0（自写 FFI + OpenCL C 内核）

**选定 OpenCL，不选 Vulkan/wgpu，也不用 candle 的 GPU。**

理由：

1. **AMD 驱动自带完整 OpenCL 2.0 运行时**。实测枚举到
   `platform: OpenCL 2.1 AMD-APP (3683.0)` / `device: gfx1150`，
   不需要装 SDK，也不需要 `OpenCL.lib`：`OpenCL.dll` 是 Khronos ICD loader，
   用 `libloading` 运行时取 20 来个符号即可（`src/gpu.rs` 里手写 FFI，
   完全没有引入 `opencl3` / `cl3` 这类需要链接导入库的 crate）。
2. **到“能跑 Q4_0 GEMV/GEMM”的距离最短**。OpenCL C 直接给 `__local`、
   work-group 尺寸、`cl_mem` 常驻缓冲；WGSL/Vulkan 要自己搭 pipeline layout、
   descriptor set、command buffer，而这里权重有 ~200 个矩阵、需要长期常驻，
   OpenCL 就是“一个 cl_mem 一个矩阵”，代码量和出错面都小一个量级。
3. **candle 0.11 没有 ROCm/HIP 后端**（只有 CUDA/Metal），走 candle 的 GPU 在
   Windows + AMD 上根本不可行，所以必须自写；自写时 OpenCL 明显比 Vulkan 划算。
4. 代价：FFI 和 kernel 都是自己的代码（约 750 行），需要自己做参数/错误检查。

结论：`--gpu` / `SENSENOVA_GPU=1` 走 OpenCL，**默认仍然是 CPU，CPU 路径一行没改**。

---

## 2. 改动的文件

| 文件 | 改动 |
|---|---|
| `src/gpu.rs` | **新增**。OpenCL 运行时加载、设备初始化、权重常驻缓存（按 host 指针索引）、激活/输出 scratch 复用、`q4_gemv`（decode）、`q4_gemm`（prefill）、两个设备能力微基准、`--gpu-probe` 用的一致性检查 |
| `src/q4.rs` | `Q4::from_bytes` 里登记 GPU 缓冲（`CL_MEM_USE_HOST_PTR` 零拷贝）；`Q4::mat` 在 GPU 开启时先走 GPU，失败自动回退 CPU；新增 `mat_cpu_ref`（纯 f32 AVX2 参考，用于验证 GPU kernel） |
| `src/main.rs` | `mod gpu`；`--gpu`、`--gpu-probe`、`--bench-gpu`、`--prefix-cache`；`bench_gpu()`（CPU/GPU 三组对照 + 真实权重 f32 误差检查）；`State` 加 `Clone`（prefix cache 快照用） |
| `src/server.rs` | Prefix cache：Engine 持有 `Mutex<Option<(Vec<u32>, Vec<State>)>>`，在消息边界（最后一个 `<|im_end|>`）打快照，下一轮只 prefill 新增部分 |
| 其他 | 测试脚本 `prefix_test.ps1`；实测输出 `bench-gpu-final.txt`、`gpu-probe-final.txt` |

（`Cargo.toml` 只加了一行 `libloading = "0.8"`。）

---

## 3. 设计要点

### 3.1 权重常驻

- 每个 Q4 矩阵在 `Q4::from_bytes` 时上传一次；用 `CL_MEM_USE_HOST_PTR`
  直接别名到 host 页（核显统一内存，**不复制 5.08GB 权重**，进程内存不翻倍，
  16GB 机器能正常跑）。驱动拒绝该指针时才退化成 `COPY_HOST_PTR`（`SENSENOVA_CL_COPY=1` 可强制）。
- `Q4::mat` 每次只上传激活 `x`（m×k×4B）并读回结果（m×n×4B），scratch 缓冲复用、按需扩容。

### 3.2 两个 kernel

- **`q4_gemv`（decode，m ≤ 8）**：每个输出行一个 128 lane 的 work-group，
  每个 lane 处理一个 18 字节 Q4 block（128 lane × 18B = 2304B 连续读），
  `uchar16` 向量载入 + 树形归约。一次 launch 处理一个 token（m 个 token 发 m 次）。
- **`q4_gemm`（prefill，m > 8）**：256 lane = 16 个“4-token 组” × 16 个“8 行组”，
  每个 lane 算 8 行 × 4 token = 32 个输出；每个 k-chunk 把
  **权重 tile（128 行 × 16 k，转置成 `[k][row]`，一条 LDS.128 取 4 个权重）**
  和 **激活 tile（16 k × 64 token，转置成 `[k][token]`，float4 读无 bank conflict）**
  搬进 local memory，内层就是 3 条 LDS + 32 条 FMA。

### 3.3 为什么第一版只有 234 GFLOP/s

第一版是 8 行 × 32 token 的 tile。**8 行的分组意味着每个行分组都要把整份激活重读一遍**：
`n/8 × m×k×4B ≈ 10GB/次 matmul`，直接把 kernel 压到 234 GFLOP/s（比 CPU VNNI 还慢）。
改成 128 行 × 64 token 后激活流量降到 ~624MB、权重流量 ~198MB，kernel 跑到 **1053–1136 GFLOP/s**
（约 fp32 峰值的 25–35%，实测 fp32 峰值 4187 GFLOP/s、读带宽 63.7 GB/s）。

---

## 4. 实测数字（CPU vs GPU）

### 4.1 端到端（`--bench-gpu`，本机 24 线程 CPU，Q4_0 权重）

| 场景 | CPU | GPU | 加速比 |
|---|---|---|---|
| 单次 matmul 12288×4096，m=1（ffn_up） | **0.95 ms**（106 GFLOP/s） | **0.83 ms**（121 GFLOP/s） | **1.14×** |
| prefill 401 token（端到端） | **44.93 s**（8.92 tok/s） | **17.27 s**（23.22 tok/s） | **2.60×** |
| decode 32 token（端到端） | **8.64 s**（3.70 tok/s） | **9.16 s**（3.49 tok/s） | **0.94×** |

decode 的启动状态是同一个 16-token 前缀，逐 token forward 计时。

### 4.2 kernel 级（`--gpu-probe`，合成 12288×4096 Q4_0，排除其它算子）

| m | CPU | GPU | 加速比 | GPU 算力 |
|---|---|---|---|---|
| 1 | 1.23 ms | 0.88 ms | 1.39× | 114 GFLOP/s |
| 8 | 3.98 ms | 5.17 ms | 0.77× | 156 GFLOP/s |
| 64 | 22.97 ms | 7.78 ms | **2.95×** | 828 GFLOP/s |
| 401 | 148.4 ms | 38.3 ms | **3.87×** | 1053 GFLOP/s |

设备微基准：fp32 FMA **4187 GFLOP/s**，读带宽 **63.7 GB/s**。

### 4.3 prefill 为什么只有 2.6×（而 matmul 是 3.87×）

一次 forward 约 390 次 matmul，GPU 只接管这部分：44.93 s / 3.87 ≈ 11.6 s，
剩下的 ~5.7 s 是仍然跑在 CPU/candle 上的非矩阵算子（attention、GDN 卷积/递推、
RMSNorm、SiLU、softmax、mHC 的 Sinkhorn）。要进一步提速需要把这些也搬上 GPU，
超出“只替换 Q4::mat”的范围。

---

## 5. 正确性验证

| 检查 | 结果 |
|---|---|
| `Q4::mat`（GPU GEMM）vs 纯 f32 AVX2 参考，真实 `blk.0.ffn_up`，m=64，变化激活 | **max\|diff\| = 4.9e-6**（max\|y\|=1.83，相对 2.7e-6） |
| 合成 Q4_0 vs CPU f32 | max\|diff\| = 6.9e-5（\|y\|≈31） |
| 401-token prefill logits，GPU vs CPU **f32**（`SENSENOVA_NOVNNI=1`） | max\|diff\| = **0.0293**，top-1 相同 |
| 同一设置下 decode 32 token 的 token 序列 | **32/32 完全一致** |
| 401-token prefill logits，GPU vs 默认 CPU（prefill 走 int8 VNNI） | max\|diff\| = 0.90，top-1 相同（差值主要来自 CPU int8 量化误差，见上一行） |
| `--prompt "The capital of France is" --max-new 8 --gpu` | 续出 **` Paris. The capital of Germany is Berlin`** ✅ |
| `--selfcheck`（CPU 默认，`SENSENOVA_NOVNNI=1`） | **0.0000** ✅ |
| `--selfcheck --gpu` | **0.0000**（5-token prompt 两侧都走同一个 GEMV kernel） |

注：不带 `SENSENOVA_NOVNNI` 时 CPU 自己的 `--selfcheck` 是 **0.3402**，
这是改动前就有的（用改动前的 `target\release\sensenova-rs.exe` 复现得到同样的 0.3402）：
prefill 走 int8 VNNI、逐步 decode 走 AVX2 f32，两个 kernel 不可能逐位相同。
0.0000 这条基线对应的是 f32 路径（`SENSENOVA_NOVNNI=1`），GPU 路径同样满足。

---

## 6. 如何启用

```powershell
# 编译（不覆盖可能被占用的 target\release\sensenova-rs.exe）
cargo build --release --target-dir target-gpu

# CPU（默认，行为不变）
.\target-gpu\release\sensenova-rs.exe --model <gguf> --tokenizer <json> --prompt "..." --max-new 32

# GPU
.\target-gpu\release\sensenova-rs.exe --model <gguf> --tokenizer <json> --prompt "..." --max-new 32 --gpu
# 等价：$env:SENSENOVA_GPU='1'

# 只要 GPU 加速 prefill，decode 留在 CPU（decode 上 CPU 更快）
$env:SENSENOVA_GPU='1'; $env:SENSENOVA_GPU_PREFILL_ONLY='1'

# 三组 CPU/GPU 对照
.\target-gpu\release\sensenova-rs.exe --model <gguf> --tokenizer <json> --bench-gpu

# kernel 级 probe（不需要模型）
.\target-gpu\release\sensenova-rs.exe --gpu-probe

# 服务（prefix cache）
.\target-gpu\release\sensenova-rs.exe --model <gguf> --tokenizer <json> --serve --port 8091 --prefix-cache --gpu
```

环境变量：

| 变量 | 作用 |
|---|---|
| `SENSENOVA_GPU=1` | 等价 `--gpu` |
| `SENSENOVA_GPU_PREFILL_ONLY=1` | m≤8 的 matmul 留在 CPU（decode 用 CPU，prefill 用 GPU） |
| `SENSENOVA_CL_COPY=1` | 权重走 `COPY_HOST_PTR` 而不是零拷贝（调试/驱动不兼容时用） |
| `SENSENOVA_GPU_VERBOSE=1` | 打印设备信息、惰性上传 |
| `SENSENOVA_PREFIX_CACHE=1` | 等价 `--prefix-cache` |

失败语义：GPU 初始化/kernel 失败只打印一次 `[gpu] disabling GPU path: ...` 并**自动回退 CPU**，
不会让进程挂掉，也不会影响默认路径。

---

## 7. GPU 反而更慢的地方：瓶颈在哪

**decode 明确是带宽瓶颈，而且 CPU 已经吃满了。**

- 这个架构每 token 要流 ~9.3GB 权重（循环层跑两遍）。CPU decode 233 ms/token
  ≈ 40–46 GB/s，已经接近实测整机 50.87 GB/s；
- GPU 和 CPU **共享同一块 LPDDR5**，换到 GPU 不会增加总带宽；
- GPU GEMV 实测 0.83–0.88 ms / 28MB ≈ **35–40 GB/s**，比 CPU 的顺序流还低一点：
  每 32 个权重（18B）要配 128B 激活、约 50 条指令，且每 token 有 ~390 次
  launch + `clFinish` + 16–48KB 往返，摊薄到每次 matmul 约 0.1ms；
- 所以 decode 端到端 0.94×（略慢）。这就是加 `SENSENOVA_GPU_PREFILL_ONLY`
  的原因：prefill 用 GPU、decode 用 CPU。

**prefill 是算力瓶颈，GPU 有真实收益**：matmul 3.87×、端到端 2.60×；
剩下的差距来自仍留在 CPU 上的非矩阵算子，以及 GPU kernel 只跑到 fp32 峰值的 ~30%
（还可以继续做 register blocking / 更大 tile / `cl-fast-relaxed-math`，但收益会递减）。

---

## 8. Prefix Caching（已实现并测过）

### 实现

`src/server.rs`：`Engine` 增加 `cache: Mutex<Option<(Vec<u32>, Vec<State>)>>`，
`--prefix-cache`（或 `SENSENOVA_PREFIX_CACHE=1`）打开。

关键点：**tokenizer 不是前缀稳定的**——上一轮 prompt 以 assistant header 结尾，
这一轮在它后面接上回复后重新编码，边界处的 token 会变（实测 lcp 只到 12/18）。
所以缓存的锚点放在**最后一个 `<|im_end|>`（消息边界，special token，前缀稳定）**：

1. prefill 时先 forward `[reused..boundary)`，在 boundary 处 `states.clone()` 存快照，
   再 forward 尾部 header 拿 logits；
2. 下一轮若 `ids.starts_with(cached_tokens)` 就把快照取出来，`forward(suffix, pos=reused)`，
   只 prefill 新增的消息 + header；
3. GDN 是递推状态、**不能回退**，所以只支持“精确前缀命中”，不做部分回退；
   不做命中就整段重算（正确但慢）。

### 实测（同一段 3 轮对话，每轮生成 6 个 token，GPU 开）

| 轮次 | 有 prefix cache | 无 prefix cache | 加速 |
|---|---|---|---|
| Turn 1（118 prompt token） | 9.55 s | 10.39 s | — |
| Turn 2（复用 95 token，prefill 23） | **5.28 s** | 9.93 s | **1.88×** |
| Turn 3（复用 112 token，prefill 23） | **5.32 s** | 11.01 s | **2.07×** |

两边的回复文本**完全一致**（`Elias saved two children from` / `The children stayed with Elias on` /
`The keeper's name was Elias`）。对话越长复用比例越高（每轮只重算最后一条消息 + header）。

限制：单槽缓存（服务是单线程的）；只支持“整段缓存前缀命中”；快照是 KV 的深拷贝
（401 token 上下文约 42MB，克隆开销 ~10ms 量级）。

---

## 9. 复现命令

```powershell
# 正确性
.\target-gpu\release\sensenova-rs.exe --model $M --tokenizer $T --prompt "The capital of France is" --max-new 8 --gpu
$env:SENSENOVA_NOVNNI='1'; .\target-gpu\release\sensenova-rs.exe --model $M --tokenizer $T --selfcheck
.\target-gpu\release\sensenova-rs.exe --model $M --tokenizer $T --selfcheck --gpu

# 性能
.\target-gpu\release\sensenova-rs.exe --model $M --tokenizer $T --bench-gpu
.\target-gpu\release\sensenova-rs.exe --gpu-probe
```

（`$M` = 你的 GGUF 路径，
`$T` = 你的 tokenizer.json 路径）

