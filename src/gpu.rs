#![allow(dead_code)] // FFI surface + diagnostic helpers kept for future backends
// Minimal OpenCL 2.0 compute backend for the Q4_0 matrix-multiply hot path.
//
// Design:
//  * OpenCL.dll is loaded at runtime (no import library needed on Windows).
//  * Every Q4 weight matrix is uploaded once into a device buffer that is kept
//    resident; `Q4::mat` only uploads the activations and reads back the result.
//  * Two kernels:
//      q4_gemv  - one work-group of 128 lanes per output row, used for decode
//                 (m == 1..4). Coalesced 18-byte block reads, tree reduce.
//      q4_gemm  - 8 rows x 32 tokens tile, 256 lanes, weights and activations
//                 staged in local memory (padded to avoid bank conflicts).
//                 Used for prefill, where re-reading the weight per token is
//                 what kills the CPU path.
//
// The CPU path remains the default; `gpu::set_enabled(true)` (or
// SENSENOVA_GPU=1 / --gpu) switches the matmul entry point over.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::q4::Q4;

pub static ENABLED: AtomicBool = AtomicBool::new(false);
static FAILED: AtomicBool = AtomicBool::new(false);

const CL_SUCCESS: i32 = 0;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_MEM_READ_ONLY: u64 = 1 << 2;
const CL_MEM_WRITE_ONLY: u64 = 1 << 1;
const CL_MEM_READ_WRITE: u64 = 1 << 0;
const CL_MEM_USE_HOST_PTR: u64 = 1 << 3;
const CL_MEM_COPY_HOST_PTR: u64 = 1 << 5;
const CL_TRUE: u32 = 1;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;

// ---- device info tokens
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
const CL_DEVICE_MAX_CLOCK_FREQUENCY: u32 = 0x100C;
const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
const CL_DEVICE_LOCAL_MEM_SIZE: u32 = 0x1023;
const CL_DEVICE_MEM_BASE_ADDR_ALIGN: u32 = 0x1019;
const CL_DEVICE_MAX_MEM_ALLOC_SIZE: u32 = 0x1010;

type ClInt = i32;

macro_rules! load {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let sym: libloading::Symbol<$ty> = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
            .map_err(|e| anyhow!("OpenCL symbol {} missing: {e}", $name))?;
        *sym
    }};
}

struct Cl {
    _lib: libloading::Library,
    get_platform_ids: unsafe extern "system" fn(u32, *mut *mut c_void, *mut u32) -> ClInt,
    get_platform_info: unsafe extern "system" fn(*mut c_void, u32, usize, *mut c_void, *mut usize) -> ClInt,
    get_device_ids: unsafe extern "system" fn(*mut c_void, u64, u32, *mut *mut c_void, *mut u32) -> ClInt,
    get_device_info: unsafe extern "system" fn(*mut c_void, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_context: unsafe extern "system" fn(*const usize, u32, *const *mut c_void, *const c_void, *mut c_void, *mut ClInt) -> *mut c_void,
    create_command_queue: unsafe extern "system" fn(*mut c_void, *mut c_void, u64, *mut ClInt) -> *mut c_void,
    create_program_with_source: unsafe extern "system" fn(*mut c_void, u32, *const *const i8, *const usize, *mut ClInt) -> *mut c_void,
    build_program: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *const i8, *const c_void, *mut c_void) -> ClInt,
    get_program_build_info: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_kernel: unsafe extern "system" fn(*mut c_void, *const i8, *mut ClInt) -> *mut c_void,
    set_kernel_arg: unsafe extern "system" fn(*mut c_void, u32, usize, *const c_void) -> ClInt,
    create_buffer: unsafe extern "system" fn(*mut c_void, u64, usize, *mut c_void, *mut ClInt) -> *mut c_void,
    enqueue_write_buffer: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, usize, usize, *const c_void, u32, *const *mut c_void, *mut *mut c_void) -> ClInt,
    enqueue_read_buffer: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, usize, usize, *mut c_void, u32, *const *mut c_void, *mut *mut c_void) -> ClInt,
    enqueue_ndrange_kernel: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const usize, *const usize, *const usize, u32, *const *mut c_void, *mut *mut c_void) -> ClInt,
    finish: unsafe extern "system" fn(*mut c_void) -> ClInt,
    release_mem_object: unsafe extern "system" fn(*mut c_void) -> ClInt,
}

unsafe impl Send for Cl {}
unsafe impl Sync for Cl {}

const KERNEL_SRC: &str = r#"
inline float h2f(ushort h) {
    uint s = (uint)(h >> 15) & 1u;
    uint e = (uint)(h >> 10) & 0x1Fu;
    uint f = (uint)h & 0x3FFu;
    uint bits;
    if (e == 0u) {
        if (f == 0u) {
            bits = s << 31;
        } else {
            uint ee = 127u - 15u + 1u;
            uint m = f;
            while ((m & 0x400u) == 0u) { m <<= 1; ee -= 1u; }
            bits = (s << 31) | (ee << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (e == 0x1Fu) {
        bits = (s << 31) | (0xFFu << 23) | (f << 13);
    } else {
        bits = (s << 31) | ((e + 127u - 15u) << 23) | (f << 13);
    }
    return as_float(bits);
}

// decode: one workgroup of 128 lanes per output row, one 18-byte block per
// lane; the tree reduction is shared across the group.
__kernel void q4_gemv(__global const uchar* w,
                      __global const float* x,
                      __global float* y,
                      const int nb,
                      const int xoff,
                      const int yoff) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    __local float red[128];
    float acc = 0.0f;
    __global const uchar* base = w + (size_t)row * (size_t)nb * 18u;
    for (int b = lid; b < nb; b += 128) {
        __global const uchar* p = base + b * 18;
        const float d = h2f(*(__global ushort*)p);
        uchar16 qv = vload16(0, p + 2);
        __global const float* xb = x + xoff + b * 32;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int byte = (int)qv[j];
            acc += d * ((float)((byte & 0x0F) - 8) * xb[j]
                      + (float)((byte >> 4) - 8) * xb[j + 16]);
        }
    }
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) y[yoff + row] = red[0];
}
// prefill: 128 rows x 64 tokens per workgroup, 256 lanes (16 row-groups x
// 16 token-groups). Each lane owns 8 output rows and 4 output tokens, so one
// dequantised weight word feeds 4 FMAs and one activation float4 feeds 8.
// Large tiles keep activation traffic down: with 8-row tiles every row group
// re-read the whole activation tensor, which dominated the first version.
#define G2T 64
#define G2R 128
#define G2K 16

#define STORE_S(S)                                                       \
    {                                                                    \
        const int tok = token0 + (S);                                    \
        const int row0 = r0 + rowbase;                                   \
        if (tok < m) {                                                   \
            __global float* base = &y[(size_t)tok * (size_t)n];          \
            if (row0 + 7 < n) {                                          \
                base[row0 + 0] = acc0.s##S;                              \
                base[row0 + 1] = acc1.s##S;                              \
                base[row0 + 2] = acc2.s##S;                              \
                base[row0 + 3] = acc3.s##S;                              \
                base[row0 + 4] = acc4.s##S;                              \
                base[row0 + 5] = acc5.s##S;                              \
                base[row0 + 6] = acc6.s##S;                              \
                base[row0 + 7] = acc7.s##S;                              \
            } else {                                                     \
                if (row0 + 0 < n) base[row0 + 0] = acc0.s##S;            \
                if (row0 + 1 < n) base[row0 + 1] = acc1.s##S;            \
                if (row0 + 2 < n) base[row0 + 2] = acc2.s##S;            \
                if (row0 + 3 < n) base[row0 + 3] = acc3.s##S;            \
                if (row0 + 4 < n) base[row0 + 4] = acc4.s##S;            \
                if (row0 + 5 < n) base[row0 + 5] = acc5.s##S;            \
                if (row0 + 6 < n) base[row0 + 6] = acc6.s##S;            \
                if (row0 + 7 < n) base[row0 + 7] = acc7.s##S;            \
            }                                                            \
        }                                                                \
    }

__kernel void q4_gemm(__global const uchar* w,
                      __global const float* x,
                      __global float* y,
                      const int m,
                      const int n,
                      const int nb) {
    const int tg = get_local_id(0);  // 0..15 -> 4 tokens each
    const int rr = get_local_id(1);  // 0..15 -> 8 rows each
    const int t0 = get_group_id(0) * G2T;
    const int r0 = get_group_id(1) * G2R;
    const int lid = rr * 16 + tg;
    const int token0 = t0 + tg * 4;
    const int rowbase = rr * 8;
    const int k = nb * 32;
    __local float wl[G2K][G2R];      // transposed: [k][row]
    __local float al[G2K][G2T + 4];  // [k][token]

    float4 acc0 = (float4)(0.0f), acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f), acc3 = (float4)(0.0f);
    float4 acc4 = (float4)(0.0f), acc5 = (float4)(0.0f);
    float4 acc6 = (float4)(0.0f), acc7 = (float4)(0.0f);

    for (int k0 = 0; k0 < k; k0 += G2K) {
        barrier(CLK_LOCAL_MEM_FENCE);
        // ---- stage 128 weight rows x 16 k values (2 lanes per row)
        {
            const int row = lid >> 1;
            const int jbase = (lid & 1) * 8;
            const int blk = k0 >> 5;
            const int part = (k0 >> 4) & 1;
            const int gr = r0 + row;
            if (blk < nb && gr < n) {
                __global const uchar* p = w + ((size_t)gr * (size_t)nb + (size_t)blk) * 18u;
                const float d = h2f(*(__global ushort*)p);
                uchar8 qv = vload8(0, p + 2 + jbase);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int jj = jbase + j;
                    int byte = (int)qv[j];
                    int v = part ? (int)(byte >> 4) : (int)(byte & 0x0F);
                    wl[jj][row] = d * (float)(v - 8);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) wl[jbase + j][row] = 0.0f;
            }
        }
        // ---- stage 16 k values x 64 tokens, transposed for conflict-free
        //      float4 reads along the token axis
        #pragma unroll
        for (int rep = 0; rep < 4; ++rep) {
            const int i = lid + rep * 256;
            const int tt = i >> 4;
            const int j = i & 15;
            const int gtok = t0 + tt;
            const int kk = k0 + j;
            al[j][tt] = (gtok < m && kk < k) ? x[(size_t)gtok * (size_t)k + kk] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll 4
        for (int j = 0; j < G2K; ++j) {
            const float4 a = *(__local float4*)&al[j][tg * 4];
            const float4 w0 = *(__local float4*)&wl[j][rowbase + 0];
            const float4 w1 = *(__local float4*)&wl[j][rowbase + 4];
            acc0 = fma(a, (float4)(w0.x), acc0);
            acc1 = fma(a, (float4)(w0.y), acc1);
            acc2 = fma(a, (float4)(w0.z), acc2);
            acc3 = fma(a, (float4)(w0.w), acc3);
            acc4 = fma(a, (float4)(w1.x), acc4);
            acc5 = fma(a, (float4)(w1.y), acc5);
            acc6 = fma(a, (float4)(w1.z), acc6);
            acc7 = fma(a, (float4)(w1.w), acc7);
        }
    }

    // ---- epilogue: y is [m][n]
    STORE_S(0)
    STORE_S(1)
    STORE_S(2)
    STORE_S(3)
}
// micro-benchmarks used to sanity-check what this iGPU can do at all
__kernel void bench_fma(__global float* out, const int iters) {
    const int gid = get_global_id(0);
    float a = (float)gid * 1e-7f + 1.0f;
    float4 acc0 = (float4)(a);
    float4 acc1 = (float4)(a * 1.1f);
    for (int i = 0; i < iters; ++i) {
        acc0 = fma(acc0, (float4)(1.000001f), (float4)(a));
        acc1 = fma(acc1, (float4)(1.000001f), (float4)(a));
    }
    out[gid] = acc0.x + acc0.y + acc0.z + acc0.w + acc1.x + acc1.y + acc1.z + acc1.w;
}

__kernel void bench_bw(__global const float4* in, __global float4* out, const int n) {
    const int gid = get_global_id(0);
    const int stride = get_global_size(0);
    float4 s = (float4)(0.0f);
    for (int i = gid; i < n; i += stride) {
        s += in[i];
    }
    out[gid] = s;
}
"#;

struct GpuBuf {
    mem: *mut c_void,
    bytes: usize,
}

unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

struct Scratch {
    x: *mut c_void,
    xcap: usize,
    y: *mut c_void,
    ycap: usize,
}

struct GpuCtx {
    cl: Cl,
    device: *mut c_void,
    context: *mut c_void,
    queue: *mut c_void,
    k_gemv: *mut c_void,
    k_gemm: *mut c_void,
    k_fma: *mut c_void,
    k_bw: *mut c_void,
    weights: Mutex<HashMap<usize, Arc<GpuBuf>>>,
    scratch: Mutex<Scratch>,
    use_host_ptr: bool,
    name: String,
    cu: u32,
}

unsafe impl Send for GpuCtx {}
unsafe impl Sync for GpuCtx {}

static CTX: OnceLock<Option<Arc<GpuCtx>>> = OnceLock::new();

fn check(rc: i32, what: &str) -> Result<()> {
    if rc == CL_SUCCESS {
        Ok(())
    } else {
        bail!("OpenCL {what} failed: {rc}")
    }
}

impl Cl {
    fn load() -> Result<Self> {
        // The Khronos ICD loader ships with the driver on every major platform;
        // no import library / SDK is required.
        fn candidates() -> &'static [&'static str] {
            #[cfg(target_os = "windows")]
            {
                &["OpenCL.dll", "C:\\Windows\\System32\\OpenCL.dll"]
            }
            #[cfg(target_os = "linux")]
            {
                &["libOpenCL.so", "libOpenCL.so.1", "libOpenCL.so.2"]
            }
            #[cfg(target_os = "macos")]
            {
                &["/System/Library/Frameworks/OpenCL.framework/OpenCL"]
            }
            #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
            {
                &["OpenCL.dll"]
            }
        }
        let names = candidates();
        let mut last_err = String::from("no candidate");
        let mut loaded = None;
        for name in names {
            match unsafe { libloading::Library::new(*name) } {
                Ok(lib) => {
                    loaded = Some(lib);
                    break;
                }
                Err(e) => last_err = format!("{name}: {e}"),
            }
        }
        let lib = loaded.ok_or_else(|| {
            anyhow!("cannot load the OpenCL ICD loader (tried {names:?}): {last_err}")
        })?;
        Ok(Cl {
            get_platform_ids: load!(lib, "clGetPlatformIDs", unsafe extern "system" fn(u32, *mut *mut c_void, *mut u32) -> ClInt),
            get_platform_info: load!(lib, "clGetPlatformInfo", unsafe extern "system" fn(*mut c_void, u32, usize, *mut c_void, *mut usize) -> ClInt),
            get_device_ids: load!(lib, "clGetDeviceIDs", unsafe extern "system" fn(*mut c_void, u64, u32, *mut *mut c_void, *mut u32) -> ClInt),
            get_device_info: load!(lib, "clGetDeviceInfo", unsafe extern "system" fn(*mut c_void, u32, usize, *mut c_void, *mut usize) -> ClInt),
            create_context: load!(lib, "clCreateContext", unsafe extern "system" fn(*const usize, u32, *const *mut c_void, *const c_void, *mut c_void, *mut ClInt) -> *mut c_void),
            create_command_queue: load!(lib, "clCreateCommandQueue", unsafe extern "system" fn(*mut c_void, *mut c_void, u64, *mut ClInt) -> *mut c_void),
            create_program_with_source: load!(lib, "clCreateProgramWithSource", unsafe extern "system" fn(*mut c_void, u32, *const *const i8, *const usize, *mut ClInt) -> *mut c_void),
            build_program: load!(lib, "clBuildProgram", unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *const i8, *const c_void, *mut c_void) -> ClInt),
            get_program_build_info: load!(lib, "clGetProgramBuildInfo", unsafe extern "system" fn(*mut c_void, *mut c_void, u32, usize, *mut c_void, *mut usize) -> ClInt),
            create_kernel: load!(lib, "clCreateKernel", unsafe extern "system" fn(*mut c_void, *const i8, *mut ClInt) -> *mut c_void),
            set_kernel_arg: load!(lib, "clSetKernelArg", unsafe extern "system" fn(*mut c_void, u32, usize, *const c_void) -> ClInt),
            create_buffer: load!(lib, "clCreateBuffer", unsafe extern "system" fn(*mut c_void, u64, usize, *mut c_void, *mut ClInt) -> *mut c_void),
            enqueue_write_buffer: load!(lib, "clEnqueueWriteBuffer", unsafe extern "system" fn(*mut c_void, *mut c_void, u32, usize, usize, *const c_void, u32, *const *mut c_void, *mut *mut c_void) -> ClInt),
            enqueue_read_buffer: load!(lib, "clEnqueueReadBuffer", unsafe extern "system" fn(*mut c_void, *mut c_void, u32, usize, usize, *mut c_void, u32, *const *mut c_void, *mut *mut c_void) -> ClInt),
            enqueue_ndrange_kernel: load!(lib, "clEnqueueNDRangeKernel", unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const usize, *const usize, *const usize, u32, *const *mut c_void, *mut *mut c_void) -> ClInt),
            finish: load!(lib, "clFinish", unsafe extern "system" fn(*mut c_void) -> ClInt),
            release_mem_object: load!(lib, "clReleaseMemObject", unsafe extern "system" fn(*mut c_void) -> ClInt),
            _lib: lib,
        })
    }
}

impl GpuCtx {
    fn init() -> Result<Self> {
        let cl = Cl::load()?;
        unsafe {
            let mut nplat = 0u32;
            check((cl.get_platform_ids)(0, std::ptr::null_mut(), &mut nplat), "clGetPlatformIDs")?;
            if nplat == 0 {
                bail!("no OpenCL platform");
            }
            let mut plats = vec![std::ptr::null_mut::<c_void>(); nplat as usize];
            check((cl.get_platform_ids)(nplat, plats.as_mut_ptr(), &mut nplat), "clGetPlatformIDs")?;

            let mut device: *mut c_void = std::ptr::null_mut();
            for &p in &plats {
                let mut ndev = 0u32;
                if (cl.get_device_ids)(p, CL_DEVICE_TYPE_GPU, 0, std::ptr::null_mut(), &mut ndev) == CL_SUCCESS && ndev > 0 {
                    let mut devs = vec![std::ptr::null_mut::<c_void>(); ndev as usize];
                    if (cl.get_device_ids)(p, CL_DEVICE_TYPE_GPU, ndev, devs.as_mut_ptr(), &mut ndev) == CL_SUCCESS {
                        device = devs[0];
                        break;
                    }
                }
            }
            if device.is_null() {
                bail!("no OpenCL GPU device");
            }

            let mut namebuf = [0u8; 256];
            let mut got = 0usize;
            (cl.get_device_info)(device, CL_DEVICE_NAME, namebuf.len(), namebuf.as_mut_ptr() as *mut c_void, &mut got);
            let name = String::from_utf8_lossy(&namebuf[..got.saturating_sub(1)]).to_string();

            let mut cu = 0u32;
            (cl.get_device_info)(device, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &mut cu as *mut u32 as *mut c_void, std::ptr::null_mut());
            let mut align_bits = 0u32;
            (cl.get_device_info)(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, 4, &mut align_bits as *mut u32 as *mut c_void, std::ptr::null_mut());
            let align_bytes = ((align_bits / 8) as usize).max(64);

            let mut err = 0i32;
            let ctx = (cl.create_context)(std::ptr::null(), 1, &device, std::ptr::null(), std::ptr::null_mut(), &mut err);
            check(err, "clCreateContext")?;
            let queue = (cl.create_command_queue)(ctx, device, 0, &mut err);
            check(err, "clCreateCommandQueue")?;

            let src = CString::new(KERNEL_SRC).unwrap();
            let sptr = src.as_ptr();
            let prog = (cl.create_program_with_source)(ctx, 1, &sptr, std::ptr::null(), &mut err);
            check(err, "clCreateProgramWithSource")?;
            let rc = (cl.build_program)(prog, 1, &device, std::ptr::null(), std::ptr::null(), std::ptr::null_mut());
            if rc != CL_SUCCESS {
                let mut log = vec![0u8; 16384];
                let mut got = 0usize;
                (cl.get_program_build_info)(prog, device, CL_PROGRAM_BUILD_LOG, log.len(), log.as_mut_ptr() as *mut c_void, &mut got);
                bail!("OpenCL build failed ({rc}): {}", String::from_utf8_lossy(&log[..got]));
            }
            let k_gemv = (cl.create_kernel)(prog, b"q4_gemv\0".as_ptr() as *const i8, &mut err);
            check(err, "clCreateKernel(q4_gemv)")?;
            let k_gemm = (cl.create_kernel)(prog, b"q4_gemm\0".as_ptr() as *const i8, &mut err);
            check(err, "clCreateKernel(q4_gemm)")?;
            let k_fma = (cl.create_kernel)(prog, b"bench_fma\0".as_ptr() as *const i8, &mut err);
            check(err, "clCreateKernel(bench_fma)")?;
            let k_bw = (cl.create_kernel)(prog, b"bench_bw\0".as_ptr() as *const i8, &mut err);
            check(err, "clCreateKernel(bench_bw)")?;

            let use_host_ptr = std::env::var("SENSENOVA_CL_COPY").is_err();
            let mem_base = {
                let mut v = 0u64;
                (cl.get_device_info)(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 8, &mut v as *mut u64 as *mut c_void, std::ptr::null_mut());
                v
            };
            if std::env::var("SENSENOVA_GPU_VERBOSE").is_ok() {
                println!(
                    "[gpu] device: {name} | CU={cu} | align={align_bytes}B | max_alloc={:.1} MiB | host_ptr={use_host_ptr}",
                    mem_base as f64 / 1048576.0
                );
            }
            Ok(GpuCtx {
                cl,
                device,
                context: ctx,
                queue,
                k_gemv,
                k_gemm,
                k_fma,
                k_bw,
                weights: Mutex::new(HashMap::new()),
                scratch: Mutex::new(Scratch {
                    x: std::ptr::null_mut(),
                    xcap: 0,
                    y: std::ptr::null_mut(),
                    ycap: 0,
                }),
                use_host_ptr,
                name,
                cu,
            })
        }
    }

    pub fn info_line(&self) -> String {
        format!("{} ({} CU)", self.name, self.cu)
    }

    fn create_weight(&self, data: &[u8]) -> Result<Arc<GpuBuf>> {
        let cl = &self.cl;
        let mut err = 0i32;
        let bytes = data.len();
        let aligned = data.as_ptr() as usize % 64 == 0;
        let mut flags = CL_MEM_READ_ONLY;
        let mut host = data.as_ptr() as *mut c_void;
        if self.use_host_ptr && aligned {
            flags |= CL_MEM_USE_HOST_PTR;
        } else {
            flags |= CL_MEM_COPY_HOST_PTR;
        }
        let mut mem = unsafe { (cl.create_buffer)(self.context, flags, bytes, host, &mut err) };
        if err != CL_SUCCESS && (flags & CL_MEM_USE_HOST_PTR) != 0 {
            // Fall back to a real copy if the driver rejects the host pointer.
            flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
            host = data.as_ptr() as *mut c_void;
            mem = unsafe { (cl.create_buffer)(self.context, flags, bytes, host, &mut err) };
        }
        check(err, "clCreateBuffer(weight)")?;
        Ok(Arc::new(GpuBuf { mem, bytes }))
    }

    fn ensure_scratch(&self, s: &mut Scratch, xbytes: usize, ybytes: usize) -> Result<()> {
        let cl = &self.cl;
        if s.xcap < xbytes {
            if !s.x.is_null() {
                unsafe { (cl.release_mem_object)(s.x) };
            }
            let mut err = 0i32;
            s.x = unsafe { (cl.create_buffer)(self.context, CL_MEM_READ_ONLY, xbytes, std::ptr::null_mut(), &mut err) };
            check(err, "clCreateBuffer(x)")?;
            s.xcap = xbytes;
        }
        if s.ycap < ybytes {
            if !s.y.is_null() {
                unsafe { (cl.release_mem_object)(s.y) };
            }
            let mut err = 0i32;
            s.y = unsafe { (cl.create_buffer)(self.context, CL_MEM_READ_WRITE, ybytes, std::ptr::null_mut(), &mut err) };
            check(err, "clCreateBuffer(y)")?;
            s.ycap = ybytes;
        }
        Ok(())
    }

    fn set_mem_arg(&self, kernel: *mut c_void, idx: u32, mem: *mut c_void) -> Result<()> {
        let h = mem as usize;
        let rc = unsafe { (self.cl.set_kernel_arg)(kernel, idx, 8, &h as *const usize as *const c_void) };
        check(rc, "clSetKernelArg(mem)")
    }

    fn set_i32_arg(&self, kernel: *mut c_void, idx: u32, v: i32) -> Result<()> {
        let rc = unsafe { (self.cl.set_kernel_arg)(kernel, idx, 4, &v as *const i32 as *const c_void) };
        check(rc, "clSetKernelArg(int)")
    }

    fn enqueue(&self, kernel: *mut c_void, global: &[usize], local: &[usize]) -> Result<()> {
        let rc = unsafe {
            (self.cl.enqueue_ndrange_kernel)(
                self.queue,
                kernel,
                global.len() as u32,
                std::ptr::null(),
                global.as_ptr(),
                local.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        check(rc, "clEnqueueNDRangeKernel")
    }

    fn finish(&self) -> Result<()> {
        check(unsafe { (self.cl.finish)(self.queue) }, "clFinish")
    }

    fn matmul(&self, q: &Q4, xv: &[f32], m: usize) -> Result<Vec<f32>> {
        let k = q.k;
        let n = q.n;
        let nb = k / 32;
        let w = weight_for(&self.weights, q).ok_or_else(|| anyhow!("weight not resident"))?;

        let mut s = self.scratch.lock().unwrap();
        self.ensure_scratch(&mut s, m * k * 4, m * n * 4)?;
        let cl = &self.cl;
        unsafe {
            let rc = (cl.enqueue_write_buffer)(
                self.queue,
                s.x,
                CL_TRUE,
                0,
                m * k * 4,
                xv.as_ptr() as *const c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            check(rc, "clEnqueueWriteBuffer(x)")?;

            if m <= 8 {
                let global = [n * 128];
                let local = [128];
                for t in 0..m {
                    self.set_mem_arg(self.k_gemv, 0, w.mem)?;
                    self.set_mem_arg(self.k_gemv, 1, s.x)?;
                    self.set_mem_arg(self.k_gemv, 2, s.y)?;
                    self.set_i32_arg(self.k_gemv, 3, nb as i32)?;
                    self.set_i32_arg(self.k_gemv, 4, (t * k) as i32)?;
                    self.set_i32_arg(self.k_gemv, 5, (t * n) as i32)?;
                    self.enqueue(self.k_gemv, &global, &local)?;
                }
            } else {
                // 128-row x 64-token tiles; local dims are 16 x 16, so the
                // global size is expressed in local work-items, not tiles.
                let global = [((m + 63) / 64) * 16, ((n + 127) / 128) * 16];
                let local = [16, 16];
                self.set_mem_arg(self.k_gemm, 0, w.mem)?;
                self.set_mem_arg(self.k_gemm, 1, s.x)?;
                self.set_mem_arg(self.k_gemm, 2, s.y)?;
                self.set_i32_arg(self.k_gemm, 3, m as i32)?;
                self.set_i32_arg(self.k_gemm, 4, n as i32)?;
                self.set_i32_arg(self.k_gemm, 5, nb as i32)?;
                self.enqueue(self.k_gemm, &global, &local)?;
            }
            self.finish()?;

            let mut out = vec![0f32; m * n];
            let rc = (cl.enqueue_read_buffer)(
                self.queue,
                s.y,
                CL_TRUE,
                0,
                m * n * 4,
                out.as_mut_ptr() as *mut c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            check(rc, "clEnqueueReadBuffer(y)")?;
            Ok(out)
        }
    }
}

fn weight_for(map: &Mutex<HashMap<usize, Arc<GpuBuf>>>, q: &Q4) -> Option<Arc<GpuBuf>> {
    let key = q.data.as_ptr() as usize;
    if let Some(b) = map.lock().unwrap().get(&key) {
        return Some(b.clone());
    }
    None
}

pub fn ctx() -> Option<&'static Arc<GpuCtx>> {
    CTX.get_or_init(|| match GpuCtx::init() {
        Ok(c) => Some(Arc::new(c)),
        Err(e) => {
            eprintln!("[gpu] init failed: {e} (falling back to CPU)");
            None
        }
    })
    .as_ref()
}

/// Upload one weight matrix to the device (idempotent). Returns the device
/// buffer size, or None when the GPU backend is unavailable.
pub fn register_weight(q: &Q4) -> Option<usize> {
    if !ENABLED.load(Ordering::Relaxed) || FAILED.load(Ordering::Relaxed) {
        return None;
    }
    let c = ctx()?;
    let key = q.data.as_ptr() as usize;
    let mut map = c.weights.lock().unwrap();
    if let Some(b) = map.get(&key) {
        return Some(b.bytes);
    }
    match c.create_weight(&q.data) {
        Ok(b) => {
            let bytes = b.bytes;
            map.insert(key, b);
            Some(bytes)
        }
        Err(e) => {
            eprintln!("[gpu] weight upload failed: {e}");
            FAILED.store(true, Ordering::Relaxed);
            None
        }
    }
}

pub fn set_enabled(v: bool) {
    ENABLED.store(v, Ordering::Relaxed);
    if v {
        match ctx() {
            Some(c) => {
                println!("[gpu] enabled: {}", c.info_line());
            }
            None => {
                eprintln!("[gpu] requested but unavailable; staying on CPU");
                ENABLED.store(false, Ordering::Relaxed);
            }
        }
    }
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed) && !FAILED.load(Ordering::Relaxed)
}

/// Decode (m == 1) is a pure bandwidth problem on a shared-memory APU, where
/// the CPU path is already near the machine's 50 GB/s. Setting
/// SENSENOVA_GPU_PREFILL_ONLY=1 keeps those matmuls on the CPU and only
/// accelerates prefill, which is the compute-bound half.
pub fn prefill_only() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("SENSENOVA_GPU_PREFILL_ONLY").ok().as_deref() == Some("1"))
}

/// GPU matmul: x is [m, k] row-major, result [m, n] row-major.
pub fn mat(q: &Q4, xv: &[f32], m: usize) -> Result<Vec<f32>> {
    let c = ctx().ok_or_else(|| anyhow!("no GPU context"))?;
    let key = q.data.as_ptr() as usize;
    let w = {
        let mut map = c.weights.lock().unwrap();
        match map.get(&key) {
            Some(b) => b.clone(),
            None => match c.create_weight(&q.data) {
                Ok(b) => {
                    let b = b;
                    let bytes = b.bytes;
                    map.insert(key, b.clone());
                    if std::env::var("SENSENOVA_GPU_VERBOSE").is_ok() {
                        println!("[gpu] lazily uploaded {} KiB", bytes / 1024);
                    }
                    b
                }
                Err(e) => return Err(e),
            },
        }
    };
    debug_assert!(w.bytes == q.data.len());
    c.matmul(q, xv, m)
}

pub fn disable_with(reason: &str) {
    if !FAILED.swap(true, Ordering::Relaxed) {
        eprintln!("[gpu] disabling GPU path: {reason} (falling back to CPU)");
    }
}

/// Raw device capability probes: fp32 FMA throughput and read bandwidth.
pub fn peak_bench() -> Result<()> {
    use std::time::Instant;
    let c = ctx().ok_or_else(|| anyhow!("no GPU context"))?;
    let cl = &c.cl;
    unsafe {
        let mut err = 0i32;
        let wi = 65536usize;
        let outbuf = (cl.create_buffer)(c.context, CL_MEM_READ_WRITE, wi * 4, std::ptr::null_mut(), &mut err);
        check(err, "clCreateBuffer(out)")?;
        let iters: i32 = 20000;
        c.set_mem_arg(c.k_fma, 0, outbuf)?;
        c.set_i32_arg(c.k_fma, 1, iters)?;
        c.enqueue(c.k_fma, &[wi], &[256])?;
        c.finish()?;
        let t = Instant::now();
        c.enqueue(c.k_fma, &[wi], &[256])?;
        c.finish()?;
        let s = t.elapsed().as_secs_f64();
        let flops = wi as f64 * iters as f64 * 16.0;
        println!("[gpu-peak] fp32 FMA: {:.0} GFLOP/s ({s:.3}s, {} threads)", flops / s / 1e9, wi);

        // read bandwidth: 64 MiB buffer, 16 passes
        let n4 = (64usize << 20) / 16;
        let inbuf = (cl.create_buffer)(c.context, CL_MEM_READ_ONLY, n4 * 16, std::ptr::null_mut(), &mut err);
        check(err, "clCreateBuffer(in)")?;
        c.set_mem_arg(c.k_bw, 0, inbuf)?;
        c.set_mem_arg(c.k_bw, 1, outbuf)?;
        c.set_i32_arg(c.k_bw, 2, n4 as i32)?;
        c.enqueue(c.k_bw, &[wi], &[256])?;
        c.finish()?;
        let passes = 16;
        let t = Instant::now();
        for _ in 0..passes {
            c.enqueue(c.k_bw, &[wi], &[256])?;
        }
        c.finish()?;
        let s = t.elapsed().as_secs_f64();
        let bytes = n4 * 16 * passes;
        println!("[gpu-peak] read: {:.2} GB/s ({} MiB x{passes} in {s:.3}s)", bytes as f64 / s / 1e9, bytes >> 20);
        (cl.release_mem_object)(outbuf);
        (cl.release_mem_object)(inbuf);
    }
    Ok(())
}

/// Enumerate the device without touching weights; used by --gpu-info.
pub fn info() -> Result<()> {
    let c = ctx().ok_or_else(|| anyhow!("no GPU context"))?;
    println!("[gpu] device: {}", c.info_line());
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn src_parses() {
        assert!(super::KERNEL_SRC.contains("q4_gemm"));
    }
}
