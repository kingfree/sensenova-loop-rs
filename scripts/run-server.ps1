<#
.SYNOPSIS
    启动 SenseNova-Loop-9B 的 OpenAI 兼容服务。

.DESCRIPTION
    prefill 默认走 GPU（若存在 GPU 构建），decode 留在 CPU —— 核显与 CPU 共享内存，
    decode 是带宽瓶颈，换 GPU 不会更快。prefix cache 默认开启。

.EXAMPLE
    .\scripts\run-server.ps1 -Model C:\models\SenseNova-Loop-9B-Q4_0.gguf

.EXAMPLE
    .\scripts\run-server.ps1 -Model D:\m\model.gguf -Tokenizer D:\m\tokenizer.json -Port 9000 -NoGpu
#>
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Tokenizer = "",
    [int]$Port = 8090,
    [switch]$NoGpu,
    [switch]$NoCache
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$candidates = @(
    (Join-Path $root "target-gpu\release\sensenova-rs.exe"),
    (Join-Path $root "target\release\sensenova-rs.exe")
)
$exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) {
    throw "找不到可执行文件。先编译：cargo build --release --target-dir target-gpu（GPU）或 cargo build --release（CPU）"
}
if (-not (Test-Path $Model)) { throw "模型文件不存在：$Model" }
if (-not $Tokenizer) { $Tokenizer = Join-Path (Split-Path -Parent (Resolve-Path $Model)) "tokenizer.json" }
if (-not (Test-Path $Tokenizer)) { throw "tokenizer.json 不存在：$Tokenizer（用 -Tokenizer 指定）" }

$cliArgs = @("--model", (Resolve-Path $Model), "--tokenizer", (Resolve-Path $Tokenizer), "--serve", "--port", "$Port")
if (-not $NoGpu) {
    $env:SENSENOVA_GPU_PREFILL_ONLY = '1'
    $cliArgs += "--gpu"
}
if (-not $NoCache) { $cliArgs += "--prefix-cache" }

Write-Host "引擎  : $exe" -ForegroundColor Cyan
Write-Host "模型  : $Model" -ForegroundColor Cyan
Write-Host "端点  : http://127.0.0.1:$Port/v1   (model id: sensenova-loop-9b)" -ForegroundColor Cyan
Write-Host "GPU   : $(if ($NoGpu) { '关闭（纯 CPU）' } else { 'prefill 启用，decode 走 CPU' })" -ForegroundColor Cyan

& $exe @cliArgs
