use candle_core::quantized::gguf_file;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: stats <gguf>");
    let dev = Device::Cpu;
    let mut f = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut f)?;

    let probes = [
        "blk.4.ssm_a",
        "blk.10.ssm_a",
        "blk.4.ssm_dt.bias",
        "blk.4.ssm_conv1d.weight",
        "blk.0.attn_norm.weight",
        "blk.4.attn_norm.weight",
        "blk.4.post_attention_norm.weight",
        "blk.3.attn_q_norm.weight",
        "blk.3.attn_k_norm.weight",
        "blk.4.mhc_attn.bias",
        "blk.4.mhc_attn.weight",
        "blk.4.mhc_ffn.bias",
        "blk.4.ssm_norm.weight",
        "blk.4.ssm_beta.weight",
        "blk.4.ssm_alpha.weight",
        "blk.4.attn_qkv.weight",
    ];
    for p in probes {
        if !content.tensor_infos.contains_key(p) {
            println!("{p}: ABSENT");
            continue;
        }
        let t = content.tensor(&mut f, p, &dev)?.dequantize(&dev)?.to_dtype(DType::F32)?;
        let shape = content.tensor_infos[p].shape.dims().to_vec();
        let n = t.elem_count();
        let v = t.flatten_all()?.to_vec1::<f32>()?;
        let mean = v.iter().sum::<f32>() / n as f32;
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("{p:34} {shape:?} mean={mean:+.4} min={mn:+.4} max={mx:+.4}");
    }
    Ok(())
}
