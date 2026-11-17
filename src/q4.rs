// Q4_0 weight matrix: rayon-parallel, AVX2-vectorised GEMV/GEMM.
//
// Design notes (decode hot path is M=1 GEMV with 4-12k rows):
//  * one fused accumulator over all blocks of a row -> a single horizontal sum
//  * rows processed in small groups per rayon task -> less task overhead
//  * unsigned-nibble unpack with AVX2, FMA against f32 activations

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use rayon::prelude::*;

pub struct Q4 {
    pub data: Vec<u8>,
    pub n: usize,
    pub k: usize,
}

const BLOCK: usize = 32;
const BYTES: usize = 18;

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_row_avx2(data: *const u8, base: usize, nb: usize, x: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let mask = _mm_set1_epi8(0x0f);
    let eight = _mm256_set1_epi32(8);
    let mut acc = _mm256_setzero_ps();
    let mut off = base;
    for b in 0..nb {
        let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(i16::from_le_bytes([
            *data.add(off),
            *data.add(off + 1),
        ]))));
        let dv = _mm256_set1_ps(d);
        let v = _mm_loadu_si128(data.add(off + 2) as *const __m128i);
        let lo = _mm_and_si128(v, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);

        let a = _mm256_sub_epi32(_mm256_cvtepu8_epi32(lo), eight);
        let bb = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)), eight);
        let c = _mm256_sub_epi32(_mm256_cvtepu8_epi32(hi), eight);
        let dd = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)), eight);

        let xp = x.add(b * 32);
        acc = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(a), dv),
            _mm256_loadu_ps(xp),
            acc,
        );
        acc = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(bb), dv),
            _mm256_loadu_ps(xp.add(8)),
            acc,
        );
        acc = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(c), dv),
            _mm256_loadu_ps(xp.add(16)),
            acc,
        );
        acc = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(dd), dv),
            _mm256_loadu_ps(xp.add(24)),
            acc,
        );
        off += BYTES;
    }
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum4 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn dequant_row_avx2(data: *const u8, base: usize, nb: usize, dst: *mut f32) {
    use std::arch::x86_64::*;
    let mask = _mm_set1_epi8(0x0f);
    let eight = _mm256_set1_epi32(8);
    let mut off = base;
    for b in 0..nb {
        let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(i16::from_le_bytes([
            *data.add(off),
            *data.add(off + 1),
        ]))));
        let dv = _mm256_set1_ps(d);
        let v = _mm_loadu_si128(data.add(off + 2) as *const __m128i);
        let lo = _mm_and_si128(v, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
        let a = _mm256_sub_epi32(_mm256_cvtepu8_epi32(lo), eight);
        let bb = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)), eight);
        let c = _mm256_sub_epi32(_mm256_cvtepu8_epi32(hi), eight);
        let dd = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)), eight);
        let o = dst.add(b * 32);
        _mm256_storeu_ps(o, _mm256_mul_ps(_mm256_cvtepi32_ps(a), dv));
        _mm256_storeu_ps(o.add(8), _mm256_mul_ps(_mm256_cvtepi32_ps(bb), dv));
        _mm256_storeu_ps(o.add(16), _mm256_mul_ps(_mm256_cvtepi32_ps(c), dv));
        _mm256_storeu_ps(o.add(24), _mm256_mul_ps(_mm256_cvtepi32_ps(dd), dv));
        off += BYTES;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s4 = _mm_add_ps(lo, hi);
    let sh = _mm_movehdup_ps(s4);
    let ss = _mm_add_ps(s4, sh);
    let sh2 = _mm_movehl_ps(sh, ss);
    _mm_cvtss_f32(_mm_add_ss(ss, sh2))
}

/// One weight row dotted against one int8-quantised activation block-row.
/// Nibbles are used as unsigned (0..15) so VNNI's u8 x i8 dot applies directly;
/// the missing `-8` offset is folded into a scalar correction `8 * sum(q)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")]
unsafe fn dot_row_vnni(
    data: *const u8,
    base: usize,
    nb: usize,
    xs: *const i8,
    dx: *const f32,
    sums: *const i32,
) -> f32 {
    use std::arch::x86_64::*;
    let mask = _mm_set1_epi8(0x0f);
    let mut accf = _mm256_setzero_ps();
    let mut corr = 0f32;
    let mut off = base;
    for b in 0..nb {
        let dw = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(i16::from_le_bytes([
            *data.add(off),
            *data.add(off + 1),
        ]))));
        let v = _mm_loadu_si128(data.add(off + 2) as *const __m128i);
        let lo = _mm_and_si128(v, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
        let a = _mm256_set_m128i(hi, lo);
        let xq = _mm256_loadu_si256(xs.add(b * 32) as *const __m256i);
        let p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), a, xq);
        let pf = _mm256_cvtepi32_ps(p);
        let s = dw * *dx.add(b);
        accf = _mm256_fmadd_ps(pf, _mm256_set1_ps(s), accf);
        corr += 8.0 * (*sums.add(b) as f32) * s;
        off += BYTES;
    }
    hsum256(accf) - corr
}

fn quantise_acts(xv: &[f32], m: usize, k: usize) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    let nb = k / BLOCK;
    let mut q = vec![0i8; m * k];
    let mut d = vec![0f32; m * nb];
    let mut s = vec![0i32; m * nb];
    q.par_chunks_mut(k)
        .zip(d.par_chunks_mut(nb))
        .zip(s.par_chunks_mut(nb))
        .enumerate()
        .for_each(|(mi, ((qr, dr), sr))| {
            let xr = &xv[mi * k..(mi + 1) * k];
            for b in 0..nb {
                let blk = &xr[b * BLOCK..b * BLOCK + BLOCK];
                let mx = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
                if mx == 0.0 {
                    continue;
                }
                let dd = mx / 127.0;
                let mut sum = 0i32;
                for (j, &v) in blk.iter().enumerate() {
                    let qi = (v / dd).round().clamp(-127.0, 127.0) as i32;
                    qr[b * BLOCK + j] = qi as i8;
                    sum += qi;
                }
                dr[b] = dd;
                sr[b] = sum;
            }
        });
    (q, d, s)
}

static VNNI: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[inline]
fn have_vnni() -> bool {
    if std::env::var("SENSENOVA_NOVNNI").is_ok() {
        return false;
    }
    *VNNI.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[inline]
fn have_avx2() -> bool {
    *AVX2.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

fn scalar_dot(data: &[u8], base: usize, nb: usize, x: &[f32]) -> f32 {
    let mut acc = 0f32;
    let mut off = base;
    for b in 0..nb {
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let qs = &data[off + 2..off + BYTES];
        let xb = &x[b * BLOCK..b * BLOCK + BLOCK];
        let mut s = 0f32;
        for j in 0..16 {
            let q = qs[j];
            s += ((q & 0x0f) as i32 - 8) as f32 * xb[j]
                + (((q >> 4) & 0x0f) as i32 - 8) as f32 * xb[j + 16];
        }
        acc += d * s;
        off += BYTES;
    }
    acc
}

const ROW_GROUP: usize = 8;

impl Q4 {
    pub fn from_bytes(data: Vec<u8>, n: usize, k: usize) -> Result<Self> {
        if k % BLOCK != 0 {
            bail!("k={k} not a multiple of {BLOCK}");
        }
        let expect = n * (k / BLOCK) * BYTES;
        if data.len() != expect {
            bail!(
                "Q4_0 size mismatch: got {} expected {expect} (n={n} k={k})",
                data.len()
            );
        }
        let q = Self { data, n, k };
        if crate::gpu::enabled() {
            crate::gpu::register_weight(&q);
        }
        Ok(q)
    }

    #[inline]
    fn row_off(&self, r: usize) -> usize {
        r * (self.k / BLOCK) * BYTES
    }

    #[inline]
    pub fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        let nb = self.k / BLOCK;
        let base = self.row_off(r);
        if have_avx2() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                return dot_row_avx2(self.data.as_ptr(), base, nb, x.as_ptr());
            }
        }
        scalar_dot(&self.data, base, nb, x)
    }

    pub fn dequant_row(&self, r: usize, dst: &mut [f32]) {
        let nb = self.k / BLOCK;
        let base = self.row_off(r);
        if have_avx2() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                dequant_row_avx2(self.data.as_ptr(), base, nb, dst.as_mut_ptr());
                return;
            }
        }
        let mut o = 0;
        for b in 0..nb {
            let off = base + b * BYTES;
            let d = f16_to_f32(u16::from_le_bytes([self.data[off], self.data[off + 1]]));
            let qs = &self.data[off + 2..off + BYTES];
            for j in 0..16 {
                let q = qs[j];
                dst[o + j] = d * (((q & 0x0f) as i32 - 8) as f32);
                dst[o + j + 16] = d * ((((q >> 4) & 0x0f) as i32 - 8) as f32);
            }
            o += 32;
        }
    }

    /// x: [m, k] -> y: [m, n]
    pub fn mat(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let m = dims[0];
        let k = dims[1];
        if k != self.k {
            bail!("matmul k mismatch: x has {k}, weight {}", self.k);
        }
        let xv = x
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let n = self.n;
        if crate::gpu::enabled() && (m > 8 || !crate::gpu::prefill_only()) {
            match crate::gpu::mat(self, &xv, m) {
                Ok(out) => return Ok(Tensor::from_vec(out, (m, n), &Device::Cpu)?),
                Err(e) => crate::gpu::disable_with(&format!("{e}")),
            }
        }
        let nb = k / BLOCK;
        let use_avx2 = have_avx2();
        let data = &self.data;

        let dot_at = |r: usize, xr: &[f32]| -> f32 {
            let base = r * nb * BYTES;
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    return dot_row_avx2(data.as_ptr(), base, nb, xr.as_ptr());
                }
            }
            scalar_dot(data, base, nb, xr)
        };

        if m == 1 {
            let xr = &xv[..k];
            let mut out = vec![0f32; n];
            out.par_chunks_mut(ROW_GROUP)
                .enumerate()
                .for_each(|(gi, rows)| {
                    let r0 = gi * ROW_GROUP;
                    for (ri, y) in rows.iter_mut().enumerate() {
                        *y = dot_at(r0 + ri, xr);
                    }
                });
            Ok(Tensor::from_vec(out, (1, n), &Device::Cpu)?)
        } else {
            // prefill: keep one weight row hot and dot it against every token,
            // so weight traffic is paid once per pass instead of once per token.
            let use_vnni = have_vnni();
            let (xq, xd, xs) = if use_vnni {
                quantise_acts(&xv, m, k)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
            let mut out_t = vec![0f32; n * m]; // [n, m]
            out_t.par_chunks_mut(m).enumerate().for_each(|(r, dst)| {
                let base = r * nb * BYTES;
                for mi in 0..m {
                    let v = if use_vnni {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            dot_row_vnni(
                                data.as_ptr(),
                                base,
                                nb,
                                xq.as_ptr().add(mi * k),
                                xd.as_ptr().add(mi * nb),
                                xs.as_ptr().add(mi * nb),
                            )
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            0.0
                        }
                    } else if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            dot_row_avx2(data.as_ptr(), base, nb, xv[mi * k..].as_ptr())
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            0.0
                        }
                    } else {
                        scalar_dot(data, base, nb, &xv[mi * k..(mi + 1) * k])
                    };
                    dst[mi] = v;
                }
            });
            let t = Tensor::from_vec(out_t, (n, m), &Device::Cpu)?;
            Ok(t.t()?.contiguous()?)
        }
    }

    /// CPU f32 reference (AVX2, no int8 activation quantisation) used by
    /// --bench-gpu to separate GPU kernel error from the CPU VNNI error.
    pub fn mat_cpu_ref(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let (m, k) = (dims[0], dims[1]);
        if k != self.k {
            bail!("matmul k mismatch");
        }
        let xv = x.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let n = self.n;
        let nb = k / BLOCK;
        let mut out = vec![0f32; m * n];
        out.par_chunks_mut(n).enumerate().for_each(|(mi, row)| {
            let xr = &xv[mi * k..(mi + 1) * k];
            for (r, y) in row.iter_mut().enumerate() {
                *y = self.dot_row(r, xr);
            }
            let _ = nb;
        });
        Ok(Tensor::from_vec(out, (m, n), &Device::Cpu)?)
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let k = self.k;
        let mut out = vec![0f32; ids.len() * k];
        out.par_chunks_mut(k)
            .zip(ids.par_iter())
            .for_each(|(row, &id)| self.dequant_row(id as usize, row));
        Ok(Tensor::from_vec(out, (ids.len(), k), &Device::Cpu)?)
    }
}
