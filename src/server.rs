// Minimal OpenAI-compatible HTTP server for the SenseNova-Loop engine.
// Endpoints: GET /v1/models, POST /v1/chat/completions (stream or not).

use crate::argmax;
use crate::{Model, State};
use anyhow::{anyhow, Result};
use candle_core::Tensor;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct Engine {
    pub model: Model,
    pub tok: Tokenizer,
    pub name: String,
    pub stop_ids: Vec<u32>,
    /// Prefix cache: (tokens, KV/GDN state after those tokens). When the next
    /// request starts with exactly these tokens we only prefill the suffix.
    pub cache: std::sync::Mutex<Option<(Vec<u32>, Vec<State>)>>,
    pub prefix_cache: bool,
}

impl Engine {
    pub fn new(model: Model, tok: Tokenizer, name: String) -> Self {
        let mut stop_ids = Vec::new();
        for s in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"] {
            if let Ok(e) = tok.encode(s, false) {
                if e.get_ids().len() == 1 {
                    stop_ids.push(e.get_ids()[0]);
                }
            }
        }
        Self {
            model,
            tok,
            name,
            stop_ids,
            cache: std::sync::Mutex::new(None),
            prefix_cache: false,
        }
    }

    fn render_chat(&self, msgs: &Value, thinking: bool) -> String {
        let mut s = String::new();
        if let Some(arr) = msgs.as_array() {
            for m in arr {
                let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let content = match m.get("content") {
                    Some(Value::String(t)) => t.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };
                s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
        }
        if thinking {
            s.push_str("<|im_start|>assistant\n<think>\n");
        } else {
            s.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        }
        s
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temp: f32,
        cb: &mut dyn FnMut(&str),
    ) -> Result<(String, usize)> {
        let enc = self
            .tok
            .encode(prompt, true)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let n_prompt = ids.len();

        // A tokenizer is not prefix-stable: text that ends at an assistant
        // header re-tokenises differently once the reply follows it. Special
        // tokens are, though, so the cache is anchored on the end of the last
        // chat message ("<|im_end|>"), which lets the *next* turn reuse every
        // token up to that point and re-prefill only the new message+header.
        let boundary = {
            let mut b = 0usize;
            for i in (0..ids.len()).rev() {
                if self.stop_ids.contains(&ids[i]) {
                    b = i + 1;
                    break;
                }
            }
            b
        };

        let mut reused = 0usize;
        let mut reuse_states: Option<Vec<State>> = None;
        if self.prefix_cache && boundary > 0 {
            if let Ok(mut guard) = self.cache.lock() {
                let hit = match guard.as_ref() {
                    Some((ctok, _)) => ids.starts_with(ctok),
                    None => false,
                };
                if hit {
                    if let Some((ctok, st)) = guard.take() {
                        reused = ctok.len();
                        reuse_states = Some(st);
                    }
                } else if std::env::var("SENSENOVA_LOG").is_ok() {
                    if let Some((ctok, _)) = guard.as_ref() {
                        let lcp = ctok.iter().zip(ids.iter()).take_while(|(a, b)| a == b).count();
                        println!(
                            "[prefix-cache] miss: cached={} new={} lcp={}",
                            ctok.len(),
                            ids.len(),
                            lcp
                        );
                    }
                }
            }
        }
        if reused >= n_prompt {
            reused = 0;
            reuse_states = None;
        }
        let mut states: Vec<State> = match reuse_states {
            Some(st) => st,
            None => self.model.new_states(),
        };
        if reused > 0 {
            println!(
                "[prefix-cache] reused {reused} of {n_prompt} prompt tokens (prefill {})",
                n_prompt - reused
            );
        }

        // forward the message part, snapshot the state there, then the trailing
        // assistant header
        let mut tail_logits: Option<Tensor> = None;
        if boundary > reused {
            let t = self.model.forward(&ids[reused..boundary], reused, &mut states)?;
            if boundary == n_prompt {
                tail_logits = Some(t);
            }
        }
        let snapshot: Option<Vec<State>> = if self.prefix_cache && boundary > 0 {
            Some(states.clone())
        } else {
            None
        };
        if boundary < n_prompt {
            tail_logits = Some(self.model.forward(&ids[boundary..], boundary, &mut states)?);
        }
        let lg = tail_logits.expect("prompt always yields logits");
        let last = lg.narrow(0, lg.dims()[0] - 1, 1)?.squeeze(0)?;
        let mut next = pick(&last, temp)?;
        let mut out = String::new();
        let mut n_gen = 0usize;
        loop {
            if self.stop_ids.contains(&next) || n_gen >= max_tokens {
                break;
            }
            let txt = self
                .tok
                .decode(&[next], false)
                .map_err(|e| anyhow!("decode: {e}"))?;
            out.push_str(&txt);
            cb(&txt);
            n_gen += 1;
            let pos = ids.len();
            ids.push(next);
            let logits = self.model.forward(&[next], pos, &mut states)?;
            let row = logits.squeeze(0)?;
            next = pick(&row, temp)?;
        }

        if self.prefix_cache && boundary > 0 {
            if let (Some(snap), Ok(mut guard)) = (snapshot, self.cache.lock()) {
                *guard = Some((ids[..boundary].to_vec(), snap));
            }
        }
        Ok((out, n_gen))
    }
}

fn pick(t: &Tensor, temp: f32) -> Result<u32> {
    let v = t.to_vec1::<f32>()?;
    if temp <= 0.0 {
        return Ok(argmax(&v) as u32);
    }
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    let mut p = Vec::with_capacity(v.len());
    for x in &v {
        let e = ((*x - mx) / temp).exp() as f64;
        sum += e;
        p.push(e);
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
    for (i, e) in p.iter().enumerate() {
        acc += e;
        if acc >= target {
            return Ok(i as u32);
        }
    }
    Ok((v.len() - 1) as u32)
}

pub fn serve(engine: Engine, port: u16) -> Result<()> {
    let engine = Arc::new(engine);
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    println!("listening on http://127.0.0.1:{port}/v1  (model: {})", engine.name);
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let e = engine.clone();
                std::thread::spawn(move || {
                    if let Err(err) = handle(s, &e) {
                        eprintln!("request error: {err}");
                    }
                });
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }
    Ok(())
}

fn read_request(s: &mut TcpStream) -> Result<(String, String, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let header_end;
    loop {
        let n = s.read(&mut tmp)?;
        if n == 0 {
            return Err(anyhow!("eof"));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            header_end = p + 4;
            break;
        }
        if buf.len() > 1 << 20 {
            return Err(anyhow!("header too large"));
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or("").to_string();
    let mut content_length = 0usize;
    for l in lines {
        let lower = l.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        let n = s.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_length);
    Ok((request_line, head, body))
}

fn respond(s: &mut TcpStream, code: &str, ctype: &str, body: &[u8]) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes())?;
    s.write_all(body)?;
    s.flush()?;
    Ok(())
}

fn handle(mut s: TcpStream, e: &Engine) -> Result<()> {
    let t0 = std::time::Instant::now();
    let (line, head, body) = read_request(&mut s)?;
    if std::env::var("SENSENOVA_LOG").is_ok() {
        let cl = body.len();
        let tools = String::from_utf8_lossy(&body).contains("\"tools\"");
        eprintln!("[req] {line}  body={cl}B tools={tools}");
        if std::env::var("SENSENOVA_LOG").map(|v| v == "2").unwrap_or(false) {
            eprintln!("{head}");
        }
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    if method == "OPTIONS" {
        respond(&mut s, "204 No Content", "text/plain", b"")?;
        return Ok(());
    }
    if method == "GET" && (path.starts_with("/health") || path == "/") {
        respond(&mut s, "200 OK", "application/json", b"{\"status\":\"ok\"}")?;
        return Ok(());
    }

    if method == "GET" && path.starts_with("/v1/models") {
        let j = json!({
            "object":"list",
            "data":[{"id":e.name,"object":"model","owned_by":"local","created":0}]
        });
        respond(&mut s, "200 OK", "application/json", j.to_string().as_bytes())?;
        return Ok(());
    }

    if method == "POST" && (path.starts_with("/v1/chat/completions") || path.starts_with("/v1/completions")) {
        let req: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
        let max_tokens = req
            .get("max_tokens")
            .or_else(|| req.get("max_completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;
        let temp = req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
        let thinking = req
            .get("enable_thinking")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || req
                .get("chat_template_kwargs")
                .and_then(|v| v.get("enable_thinking"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

        let prompt = if let Some(msgs) = req.get("messages") {
            e.render_chat(msgs, thinking)
        } else {
            req.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string()
        };

        if !stream {
            let (text, ngen) = e.generate(&prompt, max_tokens, temp, &mut |_| {})?;
            let j = json!({
                "id":"chatcmpl-local","object":"chat.completion",
                "created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
                "model": e.name,
                "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":0,"completion_tokens":ngen,"total_tokens":ngen}
            });
            respond(&mut s, "200 OK", "application/json", j.to_string().as_bytes())?;
        } else {
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            s.write_all(head.as_bytes())?;
            let mut first = true;
            let model_name = e.name.clone();
            e.generate(&prompt, max_tokens, temp, &mut |tok| {
                let delta = if first {
                    first = false;
                    json!({"role":"assistant","content":tok})
                } else {
                    json!({"content":tok})
                };
                let chunk = json!({
                    "id":"chatcmpl-local","object":"chat.completion.chunk","model":model_name,
                    "choices":[{"index":0,"delta":delta,"finish_reason":null}]
                });
                let ev = format!("data: {chunk}\n\n");
                let _ = s.write_all(format!("{:x}\r\n{ev}\r\n", ev.len()).as_bytes());
                let _ = s.flush();
            })?;
            let done = json!({
                "id":"chatcmpl-local","object":"chat.completion.chunk","model":e.name,
                "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]
            });
            let ev = format!("data: {done}\n\ndata: [DONE]\n\n");
            s.write_all(format!("{:x}\r\n{ev}\r\n0\r\n\r\n", ev.len()).as_bytes())?;
            s.flush()?;
            if std::env::var("SENSENOVA_LOG").is_ok() {
                eprintln!("[done] stream finished in {:.2?}", t0.elapsed());
            }
        }
        return Ok(());
    }

    respond(&mut s, "404 Not Found", "application/json", b"{\"error\":\"not found\"}")?;
    Ok(())
}
