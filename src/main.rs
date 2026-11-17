// SenseNova-Loop-9B inference in Rust.
//
// Architecture: Qwen3.5-class hybrid backbone (GatedDeltaNet linear attention +
// full attention every 4th layer) with a shared-weight looped decoder block
// (physical layers 4..27 executed twice) and manifold-constrained
// hyper-connections (mHC: two residual streams + Sinkhorn routing).
//
// GGUF layout of this checkpoint: physical blocks are named blk.0..blk.27 and
// blk.52..blk.55 (the tail is numbered in the 56-slot execution space); the loop
// is expressed by metadata (`mhc.layer_kinds` / `recurrent_layers`).
//
// Q4_0 weights are streamed by our own rayon-parallel kernels (see q4.rs);
// candle is used for tensor ops and for the single Q6_K lm_head.

mod gpu;
mod q4;
mod server;

use anyhow::{anyhow, bail, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QMatMul};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::ops::{log_softmax, sigmoid, silu, softmax};
use q4::Q4;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use tokenizers::Tokenizer;

// ---------------------------------------------------------------- config

#[derive(Clone, Debug)]
struct Cfg {
    hidden: usize,
    n_phys: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    eps: f64,
    inter: usize,
    n_k_heads: usize,
    k_head_dim: usize,
    n_v_heads: usize,
    v_head_dim: usize,
    conv_k: usize,
    loop_start: usize,
    loop_end: usize,
    loop_count: usize,
    mhc_streams: usize,
    mhc_iters: usize,
    mhc_pre_scale: f64,
    swapqg: bool,
    swapab: bool,
    convrev: bool,
    headtile: bool,
}

impl Cfg {
    fn key_dim(&self) -> usize {
        self.n_k_heads * self.k_head_dim
    }
    fn value_dim(&self) -> usize {
        self.n_v_heads * self.v_head_dim
    }
    fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
    fn is_full(&self, block: usize) -> bool {
        block % 4 == 3
    }
}

fn meta_usize(md: &HashMap<String, gguf_file::Value>, keys: &[&str], default: usize) -> usize {
    for k in keys {
        if let Some(v) = md.get(*k) {
            if let Ok(x) = v.to_u32() {
                return x as usize;
            }
            if let Ok(x) = v.to_u64() {
                return x as usize;
            }
        }
    }
    default
}

fn meta_f64(md: &HashMap<String, gguf_file::Value>, keys: &[&str], default: f64) -> f64 {
    for k in keys {
        if let Some(v) = md.get(*k) {
            if let Ok(x) = v.to_f32() {
                return x as f64;
            }
            if let Ok(x) = v.to_f64() {
                return x;
            }
        }
    }
    default
}

// ---------------------------------------------------------------- readers

fn read_q4(f: &mut std::fs::File, content: &gguf_file::Content, name: &str) -> Result<Q4> {
    let info = content
        .tensor_infos
        .get(name)
        .ok_or_else(|| anyhow!("tensor {name} not found"))?;
    if info.ggml_dtype != GgmlDType::Q4_0 {
        bail!("{name} is {:?}, expected Q4_0", info.ggml_dtype);
    }
    let dims = info.shape.dims();
    if dims.len() != 2 {
        bail!("{name} has rank {} (expected 2)", dims.len());
    }
    let (n, k) = (dims[0], dims[1]);
    let nbytes = n * (k / 32) * 18;
    f.seek(SeekFrom::Start(content.tensor_data_offset + info.offset))?;
    let mut buf = vec![0u8; nbytes];
    f.read_exact(&mut buf)?;
    Q4::from_bytes(buf, n, k)
}

fn read_f32(f: &mut std::fs::File, content: &gguf_file::Content, dev: &Device, name: &str) -> Result<Tensor> {
    let qt = content.tensor(f, name, dev)?;
    Ok(qt.dequantize(dev)?.to_dtype(DType::F32)?)
}

// ---------------------------------------------------------------- weights

struct Router {
    w: Tensor, // [2*s + s*s, s*hidden]
    b: Tensor,
}

struct FullAttn {
    q: Q4,
    k: Q4,
    v: Q4,
    o: Q4,
    qn: Tensor,
    kn: Tensor,
}

struct Gdn {
    qkv: Q4,
    z: Q4,
    b: Q4,
    a: Q4,
    conv: Tensor, // [conv_dim, conv_k]
    dt: Tensor,   // [n_v]
    alog: Tensor, // [n_v] == -exp(A_log)
    norm: Tensor, // [v_head_dim]
    out: Q4,
}

enum Mixer {
    Full(FullAttn),
    Linear(Gdn),
}

struct Mlp {
    gate: Q4,
    up: Q4,
    down: Q4,
}

struct Layer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    mixer: Mixer,
    mlp: Mlp,
    mhc_a: Option<Router>,
    mhc_f: Option<Router>,
}

#[derive(Clone)]
pub enum State {
    Full {
        k: Option<Tensor>,
        v: Option<Tensor>,
    },
    Linear {
        conv: Tensor,
        rec: Vec<f32>,
    },
}

// kind: 0 = plain layer, 1 = mHC enter, 2 = mHC exit, 4 = mHC middle
pub struct Model {
    cfg: Cfg,
    dev: Device,
    embed: Q4,
    layers: HashMap<usize, Layer>,
    final_norm: Tensor,
    lm_head: QMatMul,
    plan: Vec<(usize, u8)>,
    inv_freq: Vec<f64>,
}

impl Model {
    fn load(path: &str, dev: &Device, nomhc: bool, normoff: bool, swapqg: bool, swapab: bool, convrev: bool, headtile: bool) -> Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut f)?;
        let md = &content.metadata;

        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_default();
        println!("architecture: {arch}");

        let hidden = meta_usize(md, &["sensenova_loop.embedding_length"], 4096);
        let n_heads = meta_usize(md, &["sensenova_loop.attention.head_count"], 16);
        let n_kv_heads = meta_usize(md, &["sensenova_loop.attention.head_count_kv"], 4);
        let head_dim = meta_usize(
            md,
            &[
                "sensenova_loop.attention.key_length",
                "sensenova_loop.attention.value_length",
            ],
            256,
        );
        let inter = meta_usize(md, &["sensenova_loop.feed_forward_length"], 12288);
        let eps = meta_f64(md, &["sensenova_loop.attention.layer_norm_rms_epsilon"], 1e-6);
        let rope_theta = meta_f64(md, &["sensenova_loop.rope.freq_base"], 1e7);
        let rope_dim = meta_usize(md, &["sensenova_loop.rope.dimension_count"], 64);
        let conv_k = meta_usize(md, &["sensenova_loop.ssm.conv_kernel"], 4);

        let mut blocks: Vec<usize> = Vec::new();
        for name in content.tensor_infos.keys() {
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some((idx, _)) = rest.split_once('.') {
                    if let Ok(i) = idx.parse::<usize>() {
                        if !blocks.contains(&i) {
                            blocks.push(i);
                        }
                    }
                }
            }
        }
        blocks.sort_unstable();
        println!("physical blocks: {blocks:?}");

        let cfg = Cfg {
            hidden,
            n_phys: blocks.len(),
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            rope_theta,
            eps,
            inter,
            n_k_heads: 16,
            k_head_dim: 128,
            n_v_heads: 32,
            v_head_dim: 128,
            conv_k,
            loop_start: 4,
            loop_end: 28,
            loop_count: 2,
            mhc_streams: meta_usize(md, &["sensenova_loop.mhc.streams"], 2),
            mhc_iters: 20,
            mhc_pre_scale: 1.0,
            swapqg,
            swapab,
            convrev,
            headtile,
        };
        println!("cfg: {cfg:?}");

        let norm_off = |t: Tensor| -> Result<Tensor> {
            if normoff {
                Ok((t + 1.0)?)
            } else {
                Ok(t)
            }
        };

        let embed = read_q4(&mut f, &content, "token_embd.weight")?;
        let lm_head = QMatMul::from_qtensor(content.tensor(&mut f, "output.weight", dev)?)?;
        let final_norm = norm_off(read_f32(&mut f, &content, dev, "output_norm.weight")?)?;

        let mut layers: HashMap<usize, Layer> = HashMap::new();
        for &b in &blocks {
            let p = |s: &str| format!("blk.{b}.{s}");
            let attn_norm = norm_off(read_f32(&mut f, &content, dev, &p("attn_norm.weight"))?)?;
            let ffn_norm = norm_off(read_f32(&mut f, &content, dev, &p("post_attention_norm.weight"))?)?;

            let mixer = if cfg.is_full(b) {
                let qn = norm_off(read_f32(&mut f, &content, dev, &p("attn_q_norm.weight"))?)?;
                let kn = norm_off(read_f32(&mut f, &content, dev, &p("attn_k_norm.weight"))?)?;
                Mixer::Full(FullAttn {
                    q: read_q4(&mut f, &content, &p("attn_q.weight"))?,
                    k: read_q4(&mut f, &content, &p("attn_k.weight"))?,
                    v: read_q4(&mut f, &content, &p("attn_v.weight"))?,
                    o: read_q4(&mut f, &content, &p("attn_output.weight"))?,
                    qn,
                    kn,
                })
            } else {
                let conv_q = content
                    .tensor(&mut f, &p("ssm_conv1d.weight"), dev)?
                    .dequantize(dev)?
                    .to_dtype(DType::F32)?;
                let cs = conv_q.dims().to_vec();
                let conv = match cs.len() {
                    3 => conv_q.reshape((cs[0], cs[2]))?,
                    2 if cs[0] == cfg.conv_dim() => conv_q,
                    2 if cs[1] == cfg.conv_dim() => conv_q.t()?.contiguous()?,
                    _ => bail!("unexpected conv1d shape {cs:?}"),
                };
                Mixer::Linear(Gdn {
                    qkv: read_q4(&mut f, &content, &p("attn_qkv.weight"))?,
                    z: read_q4(&mut f, &content, &p("attn_gate.weight"))?,
                    b: read_q4(&mut f, &content, &p("ssm_beta.weight"))?,
                    a: read_q4(&mut f, &content, &p("ssm_alpha.weight"))?,
                    conv,
                    dt: read_f32(&mut f, &content, dev, &p("ssm_dt.bias"))?,
                    alog: read_f32(&mut f, &content, dev, &p("ssm_a"))?,
                    norm: read_f32(&mut f, &content, dev, &p("ssm_norm.weight"))?,
                    out: read_q4(&mut f, &content, &p("ssm_out.weight"))?,
                })
            };

            let mlp = Mlp {
                gate: read_q4(&mut f, &content, &p("ffn_gate.weight"))?,
                up: read_q4(&mut f, &content, &p("ffn_up.weight"))?,
                down: read_q4(&mut f, &content, &p("ffn_down.weight"))?,
            };

            let (mhc_a, mhc_f) = if content.tensor_infos.contains_key(&p("mhc_attn.weight")) {
                let wa = read_f32(&mut f, &content, dev, &p("mhc_attn.weight"))?;
                let ba = read_f32(&mut f, &content, dev, &p("mhc_attn.bias"))?;
                let wf = read_f32(&mut f, &content, dev, &p("mhc_ffn.weight"))?;
                let bf = read_f32(&mut f, &content, dev, &p("mhc_ffn.bias"))?;
                (Some(Router { w: wa, b: ba }), Some(Router { w: wf, b: bf }))
            } else {
                (None, None)
            };

            layers.insert(
                b,
                Layer {
                    attn_norm,
                    ffn_norm,
                    mixer,
                    mlp,
                    mhc_a,
                    mhc_f,
                },
            );
        }

        let mut plan: Vec<(usize, u8)> = Vec::new();
        for i in 0..cfg.loop_start {
            plan.push((i, 0));
        }
        for _ in 0..cfg.loop_count {
            for i in cfg.loop_start..cfg.loop_end {
                let kind = if i == cfg.loop_start {
                    1
                } else if i == cfg.loop_end - 1 {
                    2
                } else {
                    4
                };
                plan.push((i, kind));
            }
        }
        let tail_off = (cfg.loop_end - cfg.loop_start) * (cfg.loop_count - 1);
        for i in cfg.loop_end..cfg.n_phys {
            plan.push((i + tail_off, 0));
        }
        if nomhc {
            plan = (0..cfg.loop_end).map(|b| (b, 0u8)).collect();
            for i in cfg.loop_end..cfg.n_phys {
                plan.push((i + tail_off, 0));
            }
            println!("--nomhc: single pass, plain residuals");
        }
        println!("execution plan ({} slots): {:?}", plan.len(), plan);
        for (b, _) in &plan {
            if !layers.contains_key(b) {
                bail!("plan references missing block blk.{b}");
            }
        }

        let inv_freq: Vec<f64> = (0..cfg.rope_dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f64 / cfg.rope_dim as f64))
            .collect();

        Ok(Self {
            cfg,
            dev: dev.clone(),
            embed,
            layers,
            final_norm,
            lm_head,
            plan,
            inv_freq,
        })
    }

    fn new_states(&self) -> Vec<State> {
        let s = &self.cfg;
        self.plan
            .iter()
            .map(|(b, _)| {
                if s.is_full(*b) {
                    State::Full { k: None, v: None }
                } else {
                    State::Linear {
                        conv: Tensor::zeros((s.conv_k - 1, s.conv_dim()), DType::F32, &self.dev)
                            .unwrap(),
                        rec: vec![0f32; s.n_v_heads * s.k_head_dim * s.v_head_dim],
                    }
                }
            })
            .collect()
    }

    fn rope_cos_sin(&self, pos0: usize, t: usize) -> Result<(Tensor, Tensor)> {
        let half = self.cfg.rope_dim / 2;
        let dim = self.cfg.rope_dim;
        let mut cv = vec![0f32; t * dim];
        let mut sv = vec![0f32; t * dim];
        for i in 0..t {
            let p = (pos0 + i) as f64;
            for j in 0..half {
                let f = p * self.inv_freq[j];
                let (c, s) = (f.cos() as f32, f.sin() as f32);
                cv[i * dim + j] = c;
                cv[i * dim + half + j] = c;
                sv[i * dim + j] = s;
                sv[i * dim + half + j] = s;
            }
        }
        Ok((
            Tensor::from_vec(cv, (t, 1, dim), &self.dev)?,
            Tensor::from_vec(sv, (t, 1, dim), &self.dev)?,
        ))
    }

    fn router_forward(&self, r: &Router, residual: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (t, s, d) = residual.dims3()?;
        let flat = residual.reshape((t, s * d))?;
        let ms = flat.sqr()?.mean_keepdim(1)?;
        let xn = flat.broadcast_div(&(ms + 1e-6)?.sqrt()?)?;
        let logits = xn.matmul(&r.w.t()?)?.broadcast_add(&r.b)?;
        let pre = sigmoid(&logits.narrow(1, 0, s)?)?;
        let post = sigmoid(&logits.narrow(1, s, s)?)?.affine(2.0, 0.0)?;
        let mixl = logits.narrow(1, 2 * s, s * s)?.reshape((t, s, s))?;
        let mix = sinkhorn(&mixl, self.cfg.mhc_iters)?;
        let branch = residual.broadcast_mul(&pre.unsqueeze(2)?)?.sum(1)?;
        let branch = branch.affine(self.cfg.mhc_pre_scale, 0.0)?;
        Ok((branch, post, mix))
    }

    fn forward(&self, tokens: &[u32], pos0: usize, states: &mut [State]) -> Result<Tensor> {
        let t = tokens.len();
        let prof = std::env::var("SENSENOVA_PROF").is_ok();
        let (mut t_mix, mut t_mlp, mut t_mhc) = (0f64, 0f64, 0f64);
        let mut hidden = self.embed.embed(tokens)?;
        let (cos, sin) = self.rope_cos_sin(pos0, t)?;
        let s = self.cfg.mhc_streams;
        let h = self.cfg.hidden;

        for (visit, &(block, kind)) in self.plan.iter().enumerate() {
            let layer = self
                .layers
                .get(&block)
                .ok_or_else(|| anyhow!("missing block {block}"))?;
            if kind == 0 {
                let x = rms_plain(&hidden, &layer.attn_norm, self.cfg.eps)?;
                let _p1 = std::time::Instant::now();
                let out = self.mixer_forward(layer, &x, &cos, &sin, pos0, visit, states)?;
                t_mix += _p1.elapsed().as_secs_f64();
                hidden = (&hidden + &out)?;
                let x = rms_plain(&hidden, &layer.ffn_norm, self.cfg.eps)?;
                let out = layer.mlp.forward(&x)?;
                hidden = (&hidden + &out)?;
            } else {
                let residual = if kind == 1 {
                    hidden.unsqueeze(1)?.expand((t, s, h))?.contiguous()?
                } else {
                    hidden.reshape((t, s, h))?
                };
                let r_a = layer
                    .mhc_a
                    .as_ref()
                    .ok_or_else(|| anyhow!("block {block} has no mhc_attn"))?;
                let _p0 = std::time::Instant::now();
                let (branch, post, mix) = self.router_forward(r_a, &residual)?;
                let x = rms_plain(&branch, &layer.attn_norm, self.cfg.eps)?;
                let _p1 = std::time::Instant::now();
                let out = self.mixer_forward(layer, &x, &cos, &sin, pos0, visit, states)?;
                t_mix += _p1.elapsed().as_secs_f64();
                t_mhc += _p0.elapsed().as_secs_f64() - _p1.elapsed().as_secs_f64();
                let residual = merge(&residual, &out, &post, &mix)?;

                let r_f = layer
                    .mhc_f
                    .as_ref()
                    .ok_or_else(|| anyhow!("block {block} has no mhc_ffn"))?;
                let _p0 = std::time::Instant::now();
                let (branch, post, mix) = self.router_forward(r_f, &residual)?;
                let x = rms_plain(&branch, &layer.ffn_norm, self.cfg.eps)?;
                let _p1 = std::time::Instant::now();
                let out = layer.mlp.forward(&x)?;
                t_mlp += _p1.elapsed().as_secs_f64();
                t_mhc += _p0.elapsed().as_secs_f64() - _p1.elapsed().as_secs_f64();
                let residual = merge(&residual, &out, &post, &mix)?;

                hidden = if kind == 2 {
                    residual.mean(1)?
                } else {
                    residual.reshape((t, s * h))?
                };
            }
        }

        if prof {
            eprintln!(
                "[prof t={t}] mixer={t_mix:.2}s mlp={t_mlp:.2}s mhc+router={t_mhc:.2}s"
            );
        }
        let hh = rms_plain(&hidden, &self.final_norm, self.cfg.eps)?;
        Ok(self.lm_head.forward(&hh)?)
    }

    fn mixer_forward(
        &self,
        layer: &Layer,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pos0: usize,
        visit: usize,
        states: &mut [State],
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        match (&layer.mixer, &mut states[visit]) {
            (Mixer::Full(a), st @ State::Full { .. }) => a.forward(x, cos, sin, pos0, st, cfg),
            (Mixer::Linear(g), st @ State::Linear { .. }) => g.forward(x, st, cfg),
            _ => bail!("state/mixer mismatch at visit {visit}"),
        }
    }
}

// ---------------------------------------------------------------- ops

fn rms_plain(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xf.broadcast_div(&(var + eps)?.sqrt()?)?;
    Ok(xn.broadcast_mul(w)?)
}

fn rms_gated(x: &Tensor, gate: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xf.broadcast_div(&(var + eps)?.sqrt()?)?.broadcast_mul(w)?;
    Ok(xn.mul(&silu(&gate.to_dtype(DType::F32)?)?)?)
}

fn l2norm(x: &Tensor) -> Result<Tensor> {
    let n = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(x.broadcast_div(&(n + 1e-6)?)?)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let xc = x.clamp(-30f32, 30f32)?;
    Ok((xc.exp()? + 1.0)?.log()?)
}

fn lse(x: &Tensor, dim: usize) -> Result<Tensor> {
    let ls = log_softmax(x, dim)?;
    Ok(x.broadcast_sub(&ls)?)
}

fn sinkhorn(x: &Tensor, iters: usize) -> Result<Tensor> {
    let mut l = x.to_dtype(DType::F32)?;
    for _ in 0..iters {
        let r = lse(&l, 2)?;
        l = l.broadcast_sub(&r)?;
        let c = lse(&l, 1)?;
        l = l.broadcast_sub(&c)?;
    }
    Ok(l.exp()?)
}

fn merge(residual: &Tensor, branch: &Tensor, post: &Tensor, mix: &Tensor) -> Result<Tensor> {
    let mixed = mix.matmul(residual)?;
    let upd = post.unsqueeze(2)?.broadcast_mul(&branch.unsqueeze(1)?)?;
    Ok(mixed.add(&upd)?)
}

fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor, rope_dim: usize) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    if rope_dim == 0 || rope_dim > d {
        return Ok(x.clone());
    }
    let xr = x.narrow(D::Minus1, 0, rope_dim)?.contiguous()?;
    let half = rope_dim / 2;
    let x1 = xr.narrow(D::Minus1, 0, half)?;
    let x2 = xr.narrow(D::Minus1, half, half)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    let out = xr.broadcast_mul(cos)?.add(&rot.broadcast_mul(sin)?)?;
    if rope_dim == d {
        Ok(out)
    } else {
        let xp = x.narrow(D::Minus1, rope_dim, d - rope_dim)?.contiguous()?;
        Ok(Tensor::cat(&[&out, &xp], D::Minus1)?)
    }
}

fn repeat_heads(x: &Tensor, rep: usize) -> Result<Tensor> {
    let (t, n, d) = x.dims3()?;
    Ok(x.unsqueeze(2)?
        .expand((t, n, rep, d))?
        .contiguous()?
        .reshape((t, n * rep, d))?)
}

impl FullAttn {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pos0: usize,
        st: &mut State,
        cfg: &Cfg,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let nkv = cfg.n_kv_heads;

        let qg = self.q.mat(x)?;
        let qg = qg.reshape((t, nh, 2 * hd))?;
        let (q, gate) = if cfg.swapqg {
            (
                qg.narrow(2, hd, hd)?.contiguous()?,
                qg.narrow(2, 0, hd)?.contiguous()?,
            )
        } else {
            (qg.narrow(2, 0, hd)?, qg.narrow(2, hd, hd)?.contiguous()?)
        };
        let q = rms_plain(&q, &self.qn, cfg.eps)?;

        let k = self.k.mat(x)?.reshape((t, nkv, hd))?;
        let k = rms_plain(&k, &self.kn, cfg.eps)?;
        let v = self.v.mat(x)?.reshape((t, nkv, hd))?;

        let q = rope_apply(&q, cos, sin, cfg.rope_dim)?;
        let k = rope_apply(&k, cos, sin, cfg.rope_dim)?;

        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        let (k_all, v_all) = match st {
            State::Full { k: kc, v: vc } => {
                let ka = match kc.as_ref() {
                    Some(prev) => Tensor::cat(&[prev, &k], 1)?,
                    None => k.clone(),
                };
                let va = match vc.as_ref() {
                    Some(prev) => Tensor::cat(&[prev, &v], 1)?,
                    None => v.clone(),
                };
                *kc = Some(ka.clone());
                *vc = Some(va.clone());
                (ka, va)
            }
            _ => bail!("wrong state type"),
        };

        let tk = k_all.dim(1)?;
        let q = q.transpose(0, 1)?.contiguous()?;
        let kk = k_all;
        let vv = v_all;
        let rep = nh / nkv;
        let kk = kk
            .unsqueeze(1)?
            .expand((nkv, rep, tk, hd))?
            .contiguous()?
            .reshape((nh, tk, hd))?;
        let vv = vv
            .unsqueeze(1)?
            .expand((nkv, rep, tk, hd))?
            .contiguous()?
            .reshape((nh, tk, hd))?;

        let scale = 1.0 / (hd as f64).sqrt();
        let scores = q.matmul(&kk.transpose(1, 2)?.contiguous()?)?;
        let scores = scores.affine(scale, 0.0)?;

        let mut mv = vec![f32::NEG_INFINITY; t * tk];
        for i in 0..t {
            let qi = pos0 + i;
            for j in 0..tk {
                if j <= qi {
                    mv[i * tk + j] = 0.0;
                }
            }
        }
        let mask = Tensor::from_vec(mv, (t, tk), &q.device())?;
        let scores = scores.broadcast_add(&mask)?;
        let p = softmax(&scores, D::Minus1)?;
        let out = p.matmul(&vv)?;
        let out = out.transpose(0, 1)?.contiguous()?.reshape((t, nh * hd))?;
        let out = out.mul(&sigmoid(&gate.reshape((t, nh * hd))?)?)?;
        Ok(self.o.mat(&out)?)
    }
}

impl Gdn {
    fn forward(&self, x: &Tensor, st: &mut State, cfg: &Cfg) -> Result<Tensor> {
        let t = x.dim(0)?;
        let mixed = self.qkv.mat(x)?;

        let (conv_st, rec) = match st {
            State::Linear { conv, rec } => (conv, rec),
            _ => bail!("wrong state type"),
        };

        let padded = Tensor::cat(&[conv_st, &mixed], 0)?;
        let mut acc: Option<Tensor> = None;
        for j in 0..cfg.conv_k {
            let tap = if cfg.convrev { cfg.conv_k - 1 - j } else { j };
            let wj = self.conv.narrow(1, tap, 1)?.t()?.contiguous()?;
            let xj = padded.narrow(0, j, t)?;
            let term = xj.broadcast_mul(&wj)?;
            acc = Some(match acc {
                Some(a) => a.add(&term)?,
                None => term,
            });
        }
        let mixed = silu(&acc.unwrap())?;
        *conv_st = padded.narrow(0, t, cfg.conv_k - 1)?.contiguous()?;

        let q = mixed.narrow(1, 0, cfg.key_dim())?;
        let k = mixed.narrow(1, cfg.key_dim(), cfg.key_dim())?;
        let v = mixed.narrow(1, 2 * cfg.key_dim(), cfg.value_dim())?;

        let q = q.reshape((t, cfg.n_k_heads, cfg.k_head_dim))?;
        let k = k.reshape((t, cfg.n_k_heads, cfg.k_head_dim))?;
        let v = v.reshape((t, cfg.n_v_heads, cfg.v_head_dim))?;

        let q = l2norm(&q)?.affine(1.0 / (cfg.k_head_dim as f64).sqrt(), 0.0)?;
        let k = l2norm(&k)?;

        let (b_raw, a_raw) = if cfg.swapab {
            (self.a.mat(x)?, self.b.mat(x)?)
        } else {
            (self.b.mat(x)?, self.a.mat(x)?)
        };
        let b = sigmoid(&b_raw)?;
        let a = a_raw;
        let dt = a.broadcast_add(&self.dt)?;
        // GGUF stores `ssm_a` already as -exp(A_log) (llama.cpp convention)
        let g = softplus(&dt)?.broadcast_mul(&self.alog)?;

        let rep = cfg.n_v_heads / cfg.n_k_heads;
        let (q, k) = if cfg.headtile {
            let (t1, n1, d1) = q.dims3()?;
            let qq = q
                .unsqueeze(1)?
                .expand((t1, rep, n1, d1))?
                .contiguous()?
                .reshape((t1, n1 * rep, d1))?;
            let kk = k
                .unsqueeze(1)?
                .expand((t1, rep, n1, d1))?
                .contiguous()?
                .reshape((t1, n1 * rep, d1))?;
            (qq, kk)
        } else {
            (repeat_heads(&q, rep)?, repeat_heads(&k, rep)?)
        };

        // Sequential gated-delta-rule recurrence in plain f32 buffers: doing this
        // with per-token tensor ops costs ~0.7 ms per op on 2 MB tensors, which made
        // prefill 14x slower than the arithmetic requires.
        let nv = cfg.n_v_heads;
        let kd = cfg.k_head_dim;
        let vd = cfg.v_head_dim;
        let qv = q.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let kv = k.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let vv = v.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let bv = b.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let gv = g.contiguous()?.flatten_all()?.to_vec1::<f32>()?;

        // Heads are independent, so the recurrence is parallel over v-heads.
        // Output is accumulated per head ([nv, t, vd]) and transposed afterwards.
        let mut out = vec![0f32; nv * t * vd];
        rec.as_mut_slice()
            .par_chunks_mut(kd * vd)
            .zip(out.par_chunks_mut(t * vd))
            .enumerate()
            .for_each(|(h, (st, oh))| {
            let mut kv_mem = vec![0f32; vd];
            let mut delta = vec![0f32; vd];
            for i in 0..t {
                let decay = gv[i * nv + h].exp();
                let beta = bv[i * nv + h];
                let qh = &qv[(i * nv + h) * kd..(i * nv + h + 1) * kd];
                let kh = &kv[(i * nv + h) * kd..(i * nv + h + 1) * kd];
                let vh = &vv[(i * nv + h) * vd..(i * nv + h + 1) * vd];

                for b in kv_mem.iter_mut() {
                    *b = 0.0;
                }
                for a in 0..kd {
                    let ka = kh[a];
                    let row = &mut st[a * vd..(a + 1) * vd];
                    for (b, rb) in row.iter_mut().enumerate() {
                        *rb *= decay;
                        kv_mem[b] += *rb * ka;
                    }
                }
                for b in 0..vd {
                    delta[b] = (vh[b] - kv_mem[b]) * beta;
                }
                let o = &mut oh[i * vd..(i + 1) * vd];
                for a in 0..kd {
                    let ka = kh[a];
                    let qa = qh[a];
                    let row = &mut st[a * vd..(a + 1) * vd];
                    for b in 0..vd {
                        row[b] += ka * delta[b];
                        o[b] += row[b] * qa;
                    }
                }
            }
            });

        let out = Tensor::from_vec(out, (nv, t, vd), &x.device())?
            .transpose(0, 1)?
            .contiguous()?;
        let z = self.z.mat(x)?.reshape((t, cfg.n_v_heads, cfg.v_head_dim))?;
        let out = rms_gated(&out, &z, &self.norm, cfg.eps)?;
        let out = out.reshape((t, cfg.value_dim()))?;
        Ok(self.out.mat(&out)?)
    }
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = silu(&self.gate.mat(x)?)?;
        let u = self.up.mat(x)?;
        Ok(self.down.mat(&g.mul(&u)?)?)
    }
}

// ---------------------------------------------------------------- main

pub fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

fn gpu_probe() -> Result<()> {
    use std::time::Instant;
    let dev = Device::Cpu;
    println!("[gpu-probe] synthetic Q4_0 12288x4096, comparing CPU vs OpenCL");

    let (n, k) = (12288usize, 4096usize);
    let nb = k / 32;
    let mut data = vec![0u8; n * nb * 18];
    let mut seed = 0x12345678u64;
    for r in 0..n {
        for b in 0..nb {
            let off = (r * nb + b) * 18;
            // deterministic small fp16 scales (0.0625 .. 0.125)
            let bits: u16 = 0x2C00 | ((r as u16 ^ b as u16) & 0x3F);
            data[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            for j in 0..16 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                data[off + 2 + j] = (seed >> 33) as u8;
            }
        }
    }
    let q = Q4::from_bytes(data, n, k)?;
    let xv: Vec<f32> = (0..k)
        .map(|i| ((i as f32 * 0.017).sin() * 0.5) + ((i as f32 * 0.0031).cos() * 0.3))
        .collect();

    let run = |m: usize, gpu_on: bool, iters: usize| -> Result<(f64, Vec<f32>)> {
        crate::gpu::set_enabled(gpu_on);
        let x = Tensor::from_vec(xv.repeat(m), (m, k), &dev)?;
        let _ = q.mat(&x)?; // warm-up
        let st = Instant::now();
        let mut y = Vec::new();
        for _ in 0..iters {
            y = q.mat(&x)?.flatten_all()?.to_vec1::<f32>()?;
        }
        Ok((st.elapsed().as_secs_f64() * 1000.0 / iters as f64, y))
    };
    let report = |m: usize, iters: usize| -> Result<()> {
        let (cpu_ms, yc) = run(m, false, iters)?;
        let (gpu_ms, yg) = run(m, true, iters)?;
        let dif = yc
            .iter()
            .zip(yg.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mag = yc.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let gflops = |ms: f64| 2.0 * (m * n * k) as f64 / (ms * 1e6);
        println!(
            "[gpu-probe] m={m:3}  CPU {cpu_ms:8.3} ms ({:7.1} GFLOP/s)   GPU {gpu_ms:8.3} ms ({:7.1} GFLOP/s)   speedup {:.2}x   max|diff|={dif:.3e} (mag {mag:.2})",
            gflops(cpu_ms),
            gflops(gpu_ms),
            cpu_ms / gpu_ms
        );
        Ok(())
    };
    report(1, 20)?;
    report(8, 10)?;
    report(64, 5)?;
    report(401, 2)?;
    crate::gpu::peak_bench()?;
    crate::gpu::set_enabled(false);
    Ok(())
}

fn bench_gpu(model: &Model, tok: &Tokenizer, prompt: &str) -> Result<()> {
    use std::time::Instant;
    let dev = Device::Cpu;

    // ---------- 0) make a 401-token prompt
    let mut text = String::new();
    while text.len() < 4000 {
        text.push_str("The quick brown fox jumps over the lazy dog. ");
    }
    let enc = tok.encode(text.as_str(), true).map_err(|e| anyhow!("encode: {e}"))?;
    let mut ids401: Vec<u32> = enc.get_ids().to_vec();
    if ids401.is_empty() {
        bail!("tokenizer produced no ids");
    }
    while ids401.len() < 401 {
        let extra = ids401.clone();
        ids401.extend_from_slice(&extra);
    }
    ids401.truncate(401);

    // ---------- 1) single matmul (ffn_up, m = 1)
    {
        let w = &model.layers[&0usize].mlp.up;
        let x = Tensor::ones((1, 4096), DType::F32, &dev)?;
        let iters = 30;
        crate::gpu::set_enabled(false);
        let _ = w.mat(&x)?;
        let st = Instant::now();
        for _ in 0..iters {
            let _ = w.mat(&x)?;
        }
        let cpu_ms = st.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        crate::gpu::set_enabled(true);
        let _ = w.mat(&x)?;
        let st = Instant::now();
        for _ in 0..iters {
            let _ = w.mat(&x)?;
        }
        let gpu_ms = st.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let macs = 12288.0 * 4096.0;
        println!(
            "[bench-gpu] matmul 12288x4096 m=1: CPU {cpu_ms:.2} ms ({:.1} GFLOP/s)  GPU {gpu_ms:.2} ms ({:.1} GFLOP/s)  speedup {:.2}x",
            macs * 2.0 / (cpu_ms * 1e6),
            macs * 2.0 / (gpu_ms * 1e6),
            cpu_ms / gpu_ms
        );

        // kernel error against a pure-f32 CPU reference on the real weight
        let xv64: Vec<f32> = (0..64 * 4096)
            .map(|i| ((i as f32) * 0.0013).sin() * 0.7 + ((i as f32) * 0.021).cos() * 0.2)
            .collect();
        let x64 = Tensor::from_vec(xv64, (64, 4096), &dev)?;
        crate::gpu::set_enabled(false);
        let yref = w.mat_cpu_ref(&x64)?.flatten_all()?.to_vec1::<f32>()?;
        crate::gpu::set_enabled(true);
        let yg = w.mat(&x64)?.flatten_all()?.to_vec1::<f32>()?;
        let mut dif = 0f32;
        for (a, b) in yref.iter().zip(yg.iter()) {
            dif = dif.max((a - b).abs());
        }
        let mag = yref.iter().map(|v| v.abs()).fold(0f32, f32::max);
        println!(
            "[bench-gpu] GEMM vs CPU f32 (ffn_up, m=64): max|diff|={dif:.3e} (max|y|={mag:.2})"
        );
    }

    // ---------- 2) prefill 401 tokens
    let mut cpu_logits = Vec::new();
    let mut gpu_logits = Vec::new();
    let mut cpu_s = 0.0f64;
    let mut gpu_s = 0.0f64;
    {
        crate::gpu::set_enabled(false);
        let mut st = model.new_states();
        let t = Instant::now();
        let lg = model.forward(&ids401, 0, &mut st)?;
        cpu_s = t.elapsed().as_secs_f64();
        cpu_logits = lg.narrow(0, ids401.len() - 1, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        println!(
            "[bench-gpu] prefill 401: CPU {cpu_s:.2} s ({:.2} tok/s)",
            401.0 / cpu_s
        );
    }
    {
        crate::gpu::set_enabled(true);
        // warm up kernels / scratch buffers with a short prefill
        let mut warm = model.new_states();
        let _ = model.forward(&ids401[..8], 0, &mut warm)?;
        let mut st = model.new_states();
        let t = Instant::now();
        let lg = model.forward(&ids401, 0, &mut st)?;
        gpu_s = t.elapsed().as_secs_f64();
        gpu_logits = lg.narrow(0, ids401.len() - 1, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        println!(
            "[bench-gpu] prefill 401: GPU {gpu_s:.2} s ({:.2} tok/s)  speedup {:.2}x",
            401.0 / gpu_s,
            cpu_s / gpu_s
        );
    }
    let mut dif = 0f32;
    let mut top_cpu = 0usize;
    let mut top_gpu = 0usize;
    for i in 0..cpu_logits.len() {
        dif = dif.max((cpu_logits[i] - gpu_logits[i]).abs());
        if cpu_logits[i] > cpu_logits[top_cpu] {
            top_cpu = i;
        }
        if gpu_logits[i] > gpu_logits[top_gpu] {
            top_gpu = i;
        }
    }
    println!(
        "[bench-gpu] prefill logits: max|diff|={dif:.4}  top1 CPU={top_cpu} ({:?})  GPU={top_gpu} ({:?})  {}",
        tok.decode(&[top_cpu as u32], false).unwrap_or_default(),
        tok.decode(&[top_gpu as u32], false).unwrap_or_default(),
        if top_cpu == top_gpu { "MATCH" } else { "MISMATCH" }
    );

    // ---------- 3) decode 32 tokens (prefill 16 first)
    for (label, gpu_on) in [("CPU", false), ("GPU", true)] {
        crate::gpu::set_enabled(gpu_on);
        let mut st = model.new_states();
        let mut ids: Vec<u32> = ids401[..16].to_vec();
        let lg = model.forward(&ids, 0, &mut st)?;
        let last = lg.narrow(0, ids.len() - 1, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let mut next = argmax(&last);
        let mut toks = Vec::new();
        let t = Instant::now();
        for _ in 0..32 {
            let pos = ids.len();
            let lg = model.forward(&[next as u32], pos, &mut st)?;
            let v = lg.squeeze(0)?.to_vec1::<f32>()?;
            next = argmax(&v);
            toks.push(next);
            ids.push(next as u32);
        }
        let s = t.elapsed().as_secs_f64();
        println!(
            "[bench-gpu] decode 32 tokens: {label} {s:.2} s ({:.2} tok/s)  first8={:?}",
            32.0 / s,
            toks.iter().take(8).collect::<Vec<_>>()
        );
    }
    crate::gpu::set_enabled(false);
    let _ = prompt;
    Ok(())
}

fn main() -> Result<()> {
    let mut model_path = String::new();
    let mut tok_path = String::new();
    let mut prompt = String::from("The capital of France is");
    let mut max_new = 32usize;
    let mut temp = 0.0f32;
    let mut chat = false;
    let mut thinking = false;
    let mut topk = 0usize;
    let mut nomhc = false;
    let mut selfcheck = false;
    let mut normoff = false;
    let mut q4check = false;
    let mut bench = false;
    let mut gpu_on = false;
    let mut gpu_probe_mode = false;
    let mut bench_gpu_mode = false;
    let mut prefix_cache = false;
    let mut sv = false;
    let mut port = 8090u16;
    let mut model_name = String::from("sensenova-loop-9b");
    let mut swapqg = false;
    let mut swapab = false;
    let mut convrev = false;
    let mut headtile = true;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = args[i + 1].clone();
                i += 2;
            }
            "--tokenizer" => {
                tok_path = args[i + 1].clone();
                i += 2;
            }
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            "--max-new" => {
                max_new = args[i + 1].parse()?;
                i += 2;
            }
            "--temp" => {
                temp = args[i + 1].parse()?;
                i += 2;
            }
            "--topk" => {
                topk = args[i + 1].parse()?;
                i += 2;
            }
            "--chat" => {
                chat = true;
                i += 1;
            }
            "--thinking" => {
                thinking = true;
                i += 1;
            }
            "--nomhc" => {
                nomhc = true;
                i += 1;
            }
            "--selfcheck" => {
                selfcheck = true;
                i += 1;
            }
            "--normoffset" => {
                normoff = true;
                i += 1;
            }
            "--q4check" => {
                q4check = true;
                i += 1;
            }
            "--bench" => {
                bench = true;
                i += 1;
            }
            "--gpu" => {
                gpu_on = true;
                i += 1;
            }
            "--gpu-probe" => {
                gpu_probe_mode = true;
                i += 1;
            }
            "--bench-gpu" => {
                bench_gpu_mode = true;
                gpu_on = true;
                i += 1;
            }
            "--serve" => {
                sv = true;
                i += 1;
            }
            "--prefix-cache" => {
                prefix_cache = true;
                i += 1;
            }
            "--port" => {
                port = args[i + 1].parse()?;
                i += 2;
            }
            "--model-name" => {
                model_name = args[i + 1].clone();
                i += 2;
            }
            "--swapqg" => {
                swapqg = true;
                i += 1;
            }
            "--swapab" => {
                swapab = true;
                i += 1;
            }
            "--convrev" => {
                convrev = true;
                i += 1;
            }
            "--headtile" => {
                headtile = true;
                i += 1;
            }
            "--noheadtile" => {
                headtile = false;
                i += 1;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if model_path.is_empty() && !gpu_probe_mode {
        bail!("--model <gguf> is required");
    }

    if gpu_probe_mode {
        return gpu_probe();
    }
    if !gpu_on && std::env::var("SENSENOVA_GPU").ok().as_deref() == Some("1") {
        gpu_on = true;
    }
    if gpu_on {
        crate::gpu::set_enabled(true);
    }

    let dev = Device::Cpu;

    if q4check {
        let mut f = std::fs::File::open(&model_path)?;
        let content = gguf_file::Content::read(&mut f)?;
        for name in [
            "blk.0.ffn_up.weight",
            "blk.4.attn_qkv.weight",
            "blk.3.attn_q.weight",
        ] {
            let q4 = read_q4(&mut f, &content, name)?;
            let qm = QMatMul::from_qtensor(content.tensor(&mut f, name, &dev)?)?;
            let x = Tensor::ones((1, q4.k), DType::F32, &dev)?;
            let y1 = q4.mat(&x)?;
            let y2 = qm.forward(&x)?;
            let d = (&y1 - &y2)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let s1 = y1.sum_all()?.to_scalar::<f32>()?;
            let s2 = y2.sum_all()?.to_scalar::<f32>()?;
            println!("[q4check] {name}: max|ours-candle|={d:.6e} sum ours={s1:.4} candle={s2:.4}");
        }
        return Ok(());
    }

    let t0 = std::time::Instant::now();
    let model = Model::load(&model_path, &dev, nomhc, normoff, swapqg, swapab, convrev, headtile)?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    if bench_gpu_mode {
        let tok = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("tokenizer: {e}"))?;
        return bench_gpu(&model, &tok, &prompt);
    }

    if bench {
        let b0 = &model.layers[&0usize];
        let x = Tensor::zeros((1, 4096), DType::F32, &dev)?;
        let iters = 30;
        let st = std::time::Instant::now();
        for _ in 0..iters {
            let _y = b0.mlp.up.mat(&x)?;
        }
        let ms = st.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let macs = 12288.0 * 4096.0;
        println!("[bench] ffn_up matmul: {ms:.2} ms/call  ({:.1} GFLOP/s)", macs * 2.0 / (ms * 1e6));

        let mut states = model.new_states();
        let _ = model.forward(&[100], 0, &mut states)?;
        let mut states = model.new_states();
        let st = std::time::Instant::now();
        for _ in 0..10 {
            let _ = model.forward(&[100], 0, &mut states)?;
        }
        println!("[bench] full 1-token forward (plans only): {:.1} ms", st.elapsed().as_secs_f64() * 100.0);

        // raw streaming-read bandwidth of this machine
        {
            use rayon::prelude::*;
            let n_bytes = 2usize << 30; // 2 GiB
            let buf = vec![1u8; n_bytes];
            let st = std::time::Instant::now();
            let sum: u64 = buf
                .par_chunks(1 << 20)
                .map(|c| c.iter().map(|&b| b as u64).sum::<u64>())
                .sum();
            let el = st.elapsed().as_secs_f64();
            println!(
                "[bench] streaming read 2GiB: {:.2} GB/s (checksum {})",
                n_bytes as f64 / el / 1e9,
                sum & 1
            );
        }

        let ml = Tensor::zeros((1, 2, 2), DType::F32, &dev)?;
        let st = std::time::Instant::now();
        for _ in 0..100 {
            let _ = sinkhorn(&ml, 20)?;
        }
        println!("[bench] sinkhorn(20 iters): {:.3} ms/call", st.elapsed().as_secs_f64() * 10.0);
        return Ok(());
    }

    let tok = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("tokenizer: {e}"))?;

    if sv {
        let mut engine = server::Engine::new(model, tok, model_name);
        engine.prefix_cache = prefix_cache || std::env::var("SENSENOVA_PREFIX_CACHE").ok().as_deref() == Some("1");
        return server::serve(engine, port);
    }

    let text = if chat {
        let think_open = if thinking {
            "<|im_start|>assistant\n<think>\n".to_string()
        } else {
            "<|im_start|>assistant\n<think>\n\n</think>\n\n".to_string()
        };
        format!("<|im_start|>user\n{prompt}<|im_end|>\n{think_open}")
    } else {
        prompt.clone()
    };

    let enc = tok
        .encode(text.as_str(), true)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    print!("{prompt}");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let n_prompt = ids.len();

    if selfcheck {
        let a = model.forward(&ids, 0, &mut model.new_states())?;
        let la = a.narrow(0, n_prompt - 1, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let mut st2 = model.new_states();
        let mut lb = Vec::new();
        for (p, &tk1) in ids.iter().enumerate() {
            let lg = model.forward(&[tk1], p, &mut st2)?;
            lb = lg.squeeze(0)?.to_vec1::<f32>()?;
        }
        let mut idx: Vec<usize> = (0..la.len()).collect();
        idx.sort_by(|x, y| la[*y].partial_cmp(&la[*x]).unwrap());
        println!("[selfcheck] prefill top5:");
        for &j in idx.iter().take(5) {
            println!(
                "   {j:7} {:.4} {:?}",
                la[j],
                tok.decode(&[j as u32], false).unwrap_or_default()
            );
        }
        idx.sort_by(|x, y| lb[*y].partial_cmp(&lb[*x]).unwrap());
        println!("[selfcheck] stepwise top5:");
        for &j in idx.iter().take(5) {
            println!(
                "   {j:7} {:.4} {:?}",
                lb[j],
                tok.decode(&[j as u32], false).unwrap_or_default()
            );
        }
        let mx = la
            .iter()
            .zip(lb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("[selfcheck] max |logit diff| = {mx:.4}");
        return Ok(());
    }

    let mut states = model.new_states();

    let tp = std::time::Instant::now();
    let logits = model.forward(&ids, 0, &mut states)?;
    let dt = tp.elapsed().as_secs_f64();
    println!(
        "\n[prefill {n_prompt} tokens in {dt:.2}s = {:.2} tok/s]",
        n_prompt as f64 / dt
    );

    let last = logits.narrow(0, n_prompt - 1, 1)?.squeeze(0)?;
    let lv = last.to_vec1::<f32>()?;
    if topk > 0 {
        let mut idx: Vec<usize> = (0..lv.len()).collect();
        idx.sort_by(|a, b| lv[*b].partial_cmp(&lv[*a]).unwrap());
        println!("[top{topk}]");
        for &j in idx.iter().take(topk) {
            println!(
                "   {j:7} {:.4}  {:?}",
                lv[j],
                tok.decode(&[j as u32], false).unwrap_or_default()
            );
        }
    }
    let mut next = argmax(&lv);
    let mut gen = 0usize;
    let tg = std::time::Instant::now();
    loop {
        let txt = tok
            .decode(&[next as u32], false)
            .map_err(|e| anyhow!("decode: {e}"))?;
        print!("{txt}");
        std::io::stdout().flush().ok();
        gen += 1;
        if gen >= max_new {
            break;
        }
        let pos = ids.len();
        ids.push(next as u32);
        let logits = model.forward(&[next as u32], pos, &mut states)?;
        let mut v = logits.squeeze(0)?.to_vec1::<f32>()?;
        if temp > 0.0 {
            let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f64;
            for x in v.iter_mut() {
                *x = ((*x - mx) / temp).exp();
                sum += *x as f64;
            }
            let u = {
                let ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .subsec_nanos() as f64;
                (ns / 1e9).fract()
            };
            let target = u * sum;
            let mut acc = 0f64;
            let mut chosen = v.len() - 1;
            for (i, &p) in v.iter().enumerate() {
                acc += p as f64;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            next = chosen;
        } else {
            next = argmax(&v);
        }
    }
    let dt = tg.elapsed().as_secs_f64();
    println!(
        "\n[decode {gen} tokens in {dt:.2}s = {:.2} tok/s]",
        gen as f64 / dt
    );
    Ok(())
}
