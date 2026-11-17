use candle_core::quantized::gguf_file;
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: list <gguf>");
    let dev = Device::Cpu;
    let mut f = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut f)?;

    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort();
    println!("total tensors: {}", names.len());

    let mut blocks: Vec<String> = Vec::new();
    for n in &names {
        if let Some(rest) = n.strip_prefix("blk.") {
            if let Some((idx, _)) = rest.split_once('.') {
                if !blocks.contains(&idx.to_string()) {
                    blocks.push(idx.to_string());
                }
            }
        }
    }
    blocks.sort_by_key(|s| s.parse::<usize>().unwrap_or(0));
    println!("block indices ({}): {:?}", blocks.len(), blocks);

    println!("\n---- tensors of blk.4 / blk.5 / blk.27 / blk.28 ----");
    for n in &names {
        if n.starts_with("blk.4.") || n.starts_with("blk.5.") || n.starts_with("blk.27.") || n.starts_with("blk.28.") {
            let ti = &content.tensor_infos[*n];
            println!("  {n}  {:?} {:?}", ti.shape.dims(), ti.ggml_dtype);
        }
    }

    println!("\n---- suffix inventory ----");
    use std::collections::HashMap;
    let mut inv: HashMap<String, (usize, String)> = HashMap::new();
    for n in &names {
        let suffix = if let Some(rest) = n.strip_prefix("blk.") {
            rest.split_once('.').map(|(_, s)| s.to_string()).unwrap_or_default()
        } else {
            n.to_string()
        };
        let ti = &content.tensor_infos[*n];
        let e = inv.entry(suffix).or_insert((0, format!("{:?} {:?}", ti.shape.dims(), ti.ggml_dtype)));
        e.0 += 1;
    }
    let mut items: Vec<_> = inv.into_iter().collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, (c, shape)) in items {
        println!("  {c:3}x  {k:28} {shape}");
    }

    println!("\n---- norm weight means ----");
    for probe in ["output_norm.weight", "blk.0.attn_norm.weight", "blk.4.attn_norm.weight", "blk.0.ffn_norm.weight", "blk.0.ssm_norm.weight", "blk.3.attn_q_norm.weight"] {
        if content.tensor_infos.contains_key(probe) {
            let qt = content.tensor(&mut f, probe, &dev)?;
            let t = qt.dequantize(&dev)?.to_dtype(candle_core::DType::F32)?;
            let m = t.mean_all()?.to_scalar::<f32>()?;
            let mn = t.min_all()?.to_scalar::<f32>()?;
            let mx = t.max_all()?.to_scalar::<f32>()?;
            println!("  {probe:28} mean={m:.4} min={mn:.4} max={mx:.4}");
        }
    }
    Ok(())
}
