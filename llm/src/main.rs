mod cli;
use cli::Config;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::fs::File;
use std::path::Path;
use byteorder::{ReadBytesExt, LittleEndian};
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;
use cblas::{sgemm, Layout, Transpose};
use tokenizers::tokenizer::Tokenizer;

use std::io::{Write, BufWriter};
use std::error::Error;

/*Explanation for mutex
 rayon expects the data being mutated to be isolated per iteration to avoid data races.
 Since out is a mutable reference that multiple threads attempt to access simultaneously, Rust prevents this to ensure safety.

 use rayon's Mutex or Atomic types to handle concurrent writes to shared data. Here, using Mutex around the out array is appropriate since we're writing to different parts of the array concurrently. Each thread will lock the Mutex to write its results.
*/

const NUM_PARAMETER_TENSORS: usize = 16;
const GPT2_EOT: u32 = 50256;


//***** UTILITY FUNCTION **** */
fn encoder_forward(out: &mut [f32], inp: &[i32], wte: &[f32], wpe: &[f32], b: usize, t: usize, c: usize) {

    for b_idx in 0..b {
        for t_idx in 0..t {
            let out_start_idx = b_idx * t * c + t_idx * c;
            let out_bt = &mut out[out_start_idx..out_start_idx + c];
            // Get the index of the token at inp[b, t]
            let ix = inp[b_idx * t + t_idx] as usize;  // Convert to usize for safe indexing
            let wte_start_idx = ix * c;
            let wte_ix = &wte[wte_start_idx..wte_start_idx + c];
            let wpe_start_idx = t_idx * c;
            let wpe_t = &wpe[wpe_start_idx..wpe_start_idx + c];
            // Add the two vectors and store the result in out[b, t, :]
            for i in 0..c {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

fn encoder_backward(dwte: &mut [f32], dwpe: &mut [f32], dout: &[f32], inp: &[i32], b: usize, t: usize, c: usize) {
    for b_idx in 0..b {
        for t_idx in 0..t {
            let out_start_idx = b_idx * t * c + t_idx * c;
            let out_bt = &dout[out_start_idx..out_start_idx + c];
            let ix = inp[b_idx * t + t_idx] as usize;
            let dwte_start_idx = ix * c;
            let dwte_ix = &mut dwte[dwte_start_idx..dwte_start_idx + c];
            let dwpe_start_idx = t_idx * c;
            let dwpe_t = &mut dwpe[dwpe_start_idx..dwpe_start_idx + c];
            for i in 0..c {
                dwte_ix[i] += out_bt[i];
                dwpe_t[i] += out_bt[i];
            }
        }
    }
}

//******* GPT2 functions *****/
fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    b: usize,
    t: usize,
    c: usize,
) {
    let eps = 1e-5f32;

    for b_idx in 0..b {
        for t_idx in 0..t {
            let start_idx = b_idx * t * c + t_idx * c;
            let x = &inp[start_idx..start_idx + c];
            let out_bt = &mut out[start_idx..start_idx + c];

            // Calculate the mean
            let m: f32 = x.iter().sum::<f32>() / c as f32;

            // Calculate the variance (without any bias correction)
            let v: f32 = x.iter()
                .map(|&xi| {
                    let xshift = xi - m;
                    xshift * xshift
                })
                .sum::<f32>() / c as f32;

            // Calculate the reciprocal of the standard deviation (rstd)
            let s = 1.0 / (v + eps).sqrt();

            // Apply normalization, scale, and shift
            for i in 0..c {
                let n = (x[i] - m) * s; // normalized output
                let o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write result to output
            }

            // Cache the mean and rstd for the backward pass later
            mean[b_idx * t + t_idx] = m;
            rstd[b_idx * t + t_idx] = s;
        }
    }
}


fn layernorm_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    b: usize,
    t: usize,
    c: usize,
) {
    for b_idx in 0..b {
        for t_idx in 0..t {
            let start = b_idx * t * c + t_idx * c;
            let dout_bt = &dout[start..start + c];
            let inp_bt = &inp[start..start + c];
            let dinp_bt = &mut dinp[start..start + c];
            let mean_bt = mean[b_idx * t + t_idx];
            let rstd_bt = rstd[b_idx * t + t_idx];

            // First pass: compute dnorm_mean and dnorm_norm_mean
            let mut dnorm_mean = 0.0f32;
            let mut dnorm_norm_mean = 0.0f32;
            for i in 0..c {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= c as f32;
            dnorm_norm_mean /= c as f32;

            // Second pass: compute gradients
            for i in 0..c {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                dbias[i] += dout_bt[i];
                dweight[i] += norm_bti * dout_bt[i];

                let mut dval = dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp_bt[i] += dval;
            }
        }
    }
}


fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    t: usize,
    c: usize,
    oc: usize,
) {
    let m = (b * t) as i32; // Number of rows of the output matrix
    let k = c as i32;       // Number of columns of the input matrix / rows of the weight matrix
    let n = oc as i32;      // Number of columns of the output matrix

    // Leading dimensions for Row-Major layout
    let lda = k; // lda >= K
    let ldb = k; // ldb >= N
    let ldc = n; // ldc >= N

    //println!("m: {m}, k: {k}, n: {n}, lda: {lda}, ldb: {ldb}, ldc: {ldc}");

    // Perform the matrix multiplication using BLAS sgemm
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None, // A
            Transpose::Ordinary, // Transpose of B
            m,
            n,
            k,
            1.0,
            inp,
            lda,
            weight,
            ldb,
            0.0,
            out,
            ldc,
        );
    }

    // Add bias if present
    if let Some(bias) = bias {
        out.par_chunks_mut(oc)
            .for_each(|row| {
                for (o, val) in row.iter_mut().enumerate() {
                    *val += bias[o];
                }
            });
    }
}


fn matmul_backward_blas(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    b: usize,
    t: usize,
    c: usize,
    oc: usize,
) {
    use cblas::{sgemm, Layout, Transpose};

    let m = (b * t) as i32; // Number of rows in dout and dinp
    let k = oc as i32;      // Number of columns in dout and rows in weight
    let n = c as i32;       // Number of columns in weight and dinp

    // Compute dinp = dout * weight
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,     // No transpose for dout
            Transpose::None,     // No transpose for weight
            m,
            n,
            k,
            1.0,
            dout,
            k,                   // lda >= K (set to k)
            weight,
            n,                   // ldb >= N (set to n)
            0.0,
            dinp,
            n,                   // ldc >= N (set to n)
        );
    }

    let m_dw = oc as i32;      // Number of rows in dweight and dout^T
    let k_dw = (b * t) as i32; // Number of columns in dout^T and rows in inp
    let n_dw = c as i32;       // Number of columns in inp and dweight

    // Compute dweight = dout^T * inp
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::Ordinary, // Transpose dout
            Transpose::None,     // No transpose for inp
            m_dw,
            n_dw,
            k_dw,
            1.0,
            dout,
            m_dw,                // lda >= M (set to m_dw)
            inp,
            n_dw,                // ldb >= N (set to n_dw)
            0.0,
            dweight,
            n_dw,                // ldc >= N (set to n_dw)
        );
    }

    // Compute dbias = sum over batches and time steps of dout
    if let Some(dbias) = dbias {
        for o in 0..oc {
            let mut sum = 0.0f32;
            for bt in 0..(b * t) {
                sum += dout[bt * oc + o];
            }
            dbias[o] += sum;
        }
    }
}


fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
) {
    let c3 = c * 3;
    let hs = c / nh; // head size
    let scale = 1.0 / (hs as f32).sqrt();

    for b_idx in 0..b {
        for t_idx in 0..t {
            for h in 0..nh {
                let query_start = b_idx * t * c3 + t_idx * c3 + h * hs;
                let preatt_start = b_idx * nh * t * t + h * t * t + t_idx * t;
                let att_start = b_idx * nh * t * t + h * t * t + t_idx * t;

                // Extract the query vector Q: shape (hs)
                let query_vec = &inp[query_start..query_start + hs]; // shape: hs

                // Construct keys_mat: (t_idx+1) x hs
                let mut keys_mat = Vec::with_capacity((t_idx + 1) * hs);
                for t2 in 0..=t_idx {
                    let key_start = b_idx * t * c3 + t2 * c3 + h * hs + c; // +c for keys
                    keys_mat.extend_from_slice(&inp[key_start..key_start + hs]);
                }

                // We'll now compute preatt_row = keys_mat * query_vec
                // keys_mat: (t_idx+1) x hs
                // query_vec: hs x 1
                // result = preatt_row: (t_idx+1) x 1
                let mut preatt_row = vec![0.0f32; t_idx + 1];

                unsafe {
                    sgemm(
                        Layout::RowMajor,
                        Transpose::None,   // A as is: (t_idx+1) x hs
                        Transpose::None,   // B as is: hs x 1
                        (t_idx + 1) as i32, // m
                        1,                  // n
                        hs as i32,          // k
                        1.0,                // alpha
                        &keys_mat,          // A
                        hs as i32,          // lda = hs
                        query_vec,          // B
                        1,                  // ldb = 1 (because B is hs x 1)
                        0.0,                // beta
                        &mut preatt_row,    // C
                        1,                  // ldc = 1
                    );
                }

                // Apply scaling
                for val in &mut preatt_row {
                    *val *= scale;
                }

                // Softmax over preatt_row[0..=t_idx]
                let maxval = preatt_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let mut expsum = 0.0;
                for t2 in 0..=t_idx {
                    let expv = (preatt_row[t2] - maxval).exp().min(f32::MAX);
                    expsum += expv;
                    att[att_start + t2] = expv;
                    preatt[preatt_start + t2] = preatt_row[t2];
                }

                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                for t2 in 0..t {
                    if t2 <= t_idx {
                        att[att_start + t2] *= expsum_inv;
                    } else {
                        att[att_start + t2] = 0.0;
                    }
                }

                let out_start = b_idx * t * c + t_idx * c + h * hs;
                for i in 0..hs {
                    out[out_start + i] = 0.0;
                }

                // Construct values_mat: (t_idx+1) x hs
                let att_row = &att[att_start..att_start + (t_idx + 1)];
                let mut values_mat = Vec::with_capacity((t_idx + 1) * hs);
                for t2 in 0..=t_idx {
                    let value_start = b_idx * t * c3 + t2 * c3 + h * hs + c * 2;
                    values_mat.extend_from_slice(&inp[value_start..value_start + hs]);
                }

                let mut out_vec = vec![0.0f32; hs];

                // Now out_vec = att_row(1 x (t_idx+1)) * values_mat((t_idx+1) x hs)
                // Dimensions for sgemm:
                // A = att_row: (1 x (t_idx+1))
                // B = values_mat: ((t_idx+1) x hs)
                // C = out_vec: (1 x hs)
                // M=1, N=hs, K=(t_idx+1)
                unsafe {
                    sgemm(
                        Layout::RowMajor,
                        Transpose::None,   // A is (1x(t_idx+1))
                        Transpose::None,   // B is ((t_idx+1)xhs)
                        1,                 // M=1
                        hs as i32,         // N=hs
                        (t_idx + 1) as i32, // K=(t_idx+1)
                        1.0,               // alpha
                        att_row,           // A, lda=(t_idx+1)
                        (t_idx + 1) as i32,
                        &values_mat,       // B, ldb=hs
                        hs as i32,
                        0.0,               // beta
                        &mut out_vec,      // C, ldc=hs
                        hs as i32,
                    );
                }

                out[out_start..out_start + hs].copy_from_slice(&out_vec);
            }
        }
    }
}

fn attention_backward(
    dinp: &mut [f32],
    dpreatt: &mut [f32],
    datt: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
) {
    let c3 = c * 3;
    let hs = c / nh;
    let scale = 1.0 / (hs as f32).sqrt();

    let global_dinp = Mutex::new(vec![0.0f32; dinp.len()]);
    let global_datt = Mutex::new(vec![0.0f32; datt.len()]);
    let global_dpreatt = Mutex::new(vec![0.0f32; dpreatt.len()]);

    (0..b).into_par_iter().for_each(|b_idx| {
        let mut local_dinp = vec![0.0f32; dinp.len()];
        let mut local_datt = vec![0.0f32; datt.len()];
        let mut local_dpreatt = vec![0.0f32; dpreatt.len()];

        for t_idx in 0..t {
            for h_idx in 0..nh {
                let valid_len = t_idx + 1;

                let att_start = b_idx * nh * t * t + h_idx * t * t + t_idx * t;
                let preatt_start = att_start;
                let query_start = b_idx * t * c3 + t_idx * c3 + h_idx * hs;
                let dout_offset = b_idx * t * c + t_idx * c + h_idx * hs;

                let att_bth = &att[att_start..att_start + valid_len];
                let datt_bth = &mut local_datt[att_start..att_start + valid_len];
                let dpreatt_bth = &mut local_dpreatt[preatt_start..preatt_start + valid_len];
                let dout_bth = &dout[dout_offset..dout_offset + hs];

                let mut k_mat = vec![0.0f32; valid_len * hs];
                let mut v_mat = vec![0.0f32; valid_len * hs];

                for t2 in 0..valid_len {
                    let base = b_idx * t * c3 + t2 * c3 + h_idx * hs;
                    let key_row = &inp[base + c..base + c + hs];
                    let val_row = &inp[base + 2 * c..base + 2 * c + hs];
                    k_mat[t2 * hs..(t2 + 1) * hs].copy_from_slice(key_row);
                    v_mat[t2 * hs..(t2 + 1) * hs].copy_from_slice(val_row);
                }

                for t2 in 0..valid_len {
                    let value_t2_start = b_idx * t * c3 + t2 * c3 + h_idx * hs + 2 * c;
                    for i in 0..hs {
                        datt_bth[t2] += inp[value_t2_start + i] * dout_bth[i];
                        local_dinp[value_t2_start + i] += att_bth[t2] * dout_bth[i];
                    }
                }

                for t3 in 0..valid_len {
                    let mut sum = 0.0f32;
                    for t2 in 0..valid_len {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        sum += local_derivative * datt_bth[t2];
                    }
                    dpreatt_bth[t3] += sum;
                }

                let mut dpreatt_scaled = dpreatt_bth.to_vec();
                dpreatt_scaled.iter_mut().for_each(|x| *x *= scale);

                let mut dq = vec![0.0f32; hs];
                for t2 in 0..valid_len {
                    let dp = dpreatt_scaled[t2];
                    for i in 0..hs {
                        dq[i] += k_mat[t2 * hs + i] * dp;
                    }
                }
                for i in 0..hs {
                    local_dinp[query_start + i] += dq[i];
                }

                for t2 in 0..valid_len {
                    let dp = dpreatt_scaled[t2];
                    let key_t2_start = b_idx * t * c3 + t2 * c3 + h_idx * hs + c;
                    for i in 0..hs {
                        local_dinp[key_t2_start + i] += inp[query_start + i] * dp;
                    }
                }
            }
        }

        // Safely merge local results into global arrays
        {
            let mut g_dinp = global_dinp.lock().unwrap();
            g_dinp
                .iter_mut()
                .zip(local_dinp.iter())
                .for_each(|(g, l)| *g += l);
        }
        {
            let mut g_datt = global_datt.lock().unwrap();
            g_datt
                .iter_mut()
                .zip(local_datt.iter())
                .for_each(|(g, l)| *g += l);
        }
        {
            let mut g_dpreatt = global_dpreatt.lock().unwrap();
            g_dpreatt
                .iter_mut()
                .zip(local_dpreatt.iter())
                .for_each(|(g, l)| *g += l);
        }
    });

    // Copy results back to the original mutable references
    dinp.copy_from_slice(&global_dinp.lock().unwrap());
    datt.copy_from_slice(&global_datt.lock().unwrap());
    dpreatt.copy_from_slice(&global_dpreatt.lock().unwrap());
}


fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], n: usize) {
    // Ensure that all slices have at least 'n' elements
    assert!(out.len() >= n && inp1.len() >= n && inp2.len() >= n, "Input slices must be of at least size n");

    for i in 0..n {
        out[i] = inp1[i] + inp2[i];
    }
    /* Iterator implementation
    fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32]) {
    // Iterator-based approach, automatically handling bounds and potentially more idiomatic
    for ((o, i1), i2) in out.iter_mut().zip(inp1.iter()).zip(inp2.iter()) {
        *o = i1 + i2;
            }
    }
    */
}

fn residual_backward(dinp1: &mut [f32], dinp2: &mut [f32], dout: &mut [f32], n: usize){
    for i in 0..n {
        dinp1[i] = dout[i];
        dinp2[i] = dout[i];
    }
}

fn gelu_forward(out: &mut [f32], inp: &[f32], n: usize) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    for i in 0..n {
        let x = inp[i];
        let cube: f32 = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + (s * (x + cube)).tanh());
    }
}

fn gelu_backward(dinp: &mut [f32], inp: &[f32], dout: &mut [f32], n: usize) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    for i in 0..n {
        let x = inp[i];
        let cube: f32 = 0.044715 * x * x * x;
        let tanh_arg = s * (x + cube);
        let tanh_out = (tanh_arg).tanh();
        let coshf_out = (tanh_arg).cosh();
        let sech_out = 1.0 / (coshf_out*coshf_out);
        let local_grad = 0.5  * ( 1.0 + tanh_out) + x * 0.5 * sech_out * s * (1.0 + 3.0 * 0.044715*x*x);
        dinp[i] += local_grad * dout[i];
    }
}

fn softmax_forward(out: &mut [f32], logits: &[f32], b: usize, t: usize, v: usize){
    // here Karpathy uses pragma
    for b_idx in 0..b{
        for t_idx in 0..t {
            let start_idx = b_idx * t * v + t_idx * v;
            let logits_bt = &logits[start_idx..start_idx + v];
            let out_bt = &mut out[start_idx..start_idx + v];
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..v {
                if logits_bt[i] > max_val {
                    max_val = logits_bt[i];
                }
            }
            let mut sum = 0.0;
            for i in 0..v {
                let val = (logits_bt[i] - max_val).exp();
                out_bt[i] = val;
                sum += val;
            }
            let sum_inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            for i in 0..v {
                out_bt[i] *= sum_inv;
            }
        }
    }
}

fn crossentropy_forward(out: &mut [f32], probs: &[f32], targets: &[i32], b: usize, t: usize, v: usize){
    for b_idx in 0..b {
        for t_idx in 0..t {
            let target = targets[b_idx * t + t_idx] as usize; // int ix
            let start_idx = b_idx * t * v + t_idx * v; // index
            let probs_bt = &probs[start_idx..start_idx + v]; //  probs_bt
            out[b_idx * t + t_idx] = -probs_bt[target].ln();
        }
    }
}


fn crossentropy_softmax_backward(
    dlogits: &mut [f32],
    dlosses: &mut [f32],
    probs: &[f32],
    targets: &[i32],
    b: usize,
    t: usize,
    v: usize,
) {
    for b_idx in 0..b {
        for t_idx in 0..t {
            let idx_start = b_idx * t * v + t_idx * v;
            let dlogits_bt = &mut dlogits[idx_start..idx_start + v];
            let probs_bt = &probs[idx_start..idx_start + v];
            let dloss = dlosses[b_idx * t + t_idx];
            let target_idx = targets[b_idx * t + t_idx] as usize;  // Convert to usize for indexing

            for i in 0..v {
                let p = probs_bt[i];
                let indicator = (i == target_idx) as i32 as f32;  // 1.0 if true, 0.0 otherwise
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

//******** DATALOADER CONFIGURATIONS ******//
struct DataLoader {
    b: usize,
    t: usize,
    tokens_file: BufReader<File>,
    file_size: u64,
    current_position: u64,
    batch: Vec<i32>,
    inputs: Vec<i32>,
    targets: Vec<i32>,
    num_batches: usize,
}

impl DataLoader {
    fn new(file_path: &Path, b: usize, t: usize) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        let file_size = reader.seek(io::SeekFrom::End(0))?;
        reader.seek(io::SeekFrom::Start(0))?;

        if file_size < ((b * t + 1) * std::mem::size_of::<i32>()) as u64 {
            return Err(io::Error::new(io::ErrorKind::Other, "File too small"));
        }

        let mut loader = DataLoader {
            b,
            t,
            tokens_file: reader,
            file_size,
            current_position: 0,
            batch: vec![0; b * t + 1],
            inputs: vec![],  // Temporary placeholder
            targets: vec![], // Temporary placeholder
            num_batches: (file_size / (b * t * std::mem::size_of::<i32>() as usize) as u64) as usize,
        };

        loader.update_slices();
        Ok(loader)
    }

    fn reset(&mut self) -> io::Result<()> {
        self.current_position = 0;
        Ok(())
    }

    fn next_batch(&mut self) -> io::Result<()> {
        if self.current_position + (self.b * self.t + 1) as u64 * std::mem::size_of::<i32>() as u64 > self.file_size {
            self.current_position = 0;
        }

        self.tokens_file.seek(SeekFrom::Start(self.current_position))?;
        let buffer = self.batch.as_mut_slice();
        self.tokens_file.read_exact(bytemuck::cast_slice_mut(buffer))?;
        self.current_position += (self.b * self.t) as u64 * std::mem::size_of::<i32>() as u64;

        // Update slices to point to the new data
        self.update_slices();
        Ok(())
    }

    fn update_slices(&mut self) {
        self.inputs = self.batch[0..self.b * self.t].to_vec();
        self.targets = self.batch[1..=self.b * self.t].to_vec();
    }
}
/* END OF DATALOADER CONFIGURATION */

//****** GPT2 CONFIGURATIONS ********//
struct GPT2Config {
    max_seq_len: usize,
    vocab_size: usize,
    num_layers: usize,
    num_heads: usize,
    channels: usize,
}

struct ParameterTensors {
    wte: Vec<f32>, // (V, C)
    wpe: Vec<f32>, // (maxT, C)
    ln1w: Vec<f32>, // (L, C)
    ln1b: Vec<f32>, // (L, C)
    qkvw: Vec<f32>, // (L, 3*C, C)
    qkvb: Vec<f32>, // (L, 3*C)
    attprojw: Vec<f32>, // (L, C, C)
    attprojb: Vec<f32>, // (L, C)
    ln2w: Vec<f32>, // (L, C)
    ln2b: Vec<f32>, // (L, C)
    fcw: Vec<f32>, // (L, 4*C, C)
    fcb: Vec<f32>, // (L, 4*C)
    fcprojw: Vec<f32>, // (L, C, 4*C)
    fcprojb: Vec<f32>, // (L, C)
    lnfw: Vec<f32>, // (C)
    lnfb: Vec<f32>, // (C)
}

struct ActivationTensors {
    encoded: Vec<f32>, // (B, T, C)
    ln1: Vec<f32>, // (L, B, T, C)
    ln1_mean: Vec<f32>, // (L, B, T)
    ln1_rstd: Vec<f32>, // (L, B, T)
    qkv: Vec<f32>, // (L, B, T, 3*C)
    atty: Vec<f32>, // (L, B, T, C)
    preatt: Vec<f32>, // (L, B, NH, T, T)
    att: Vec<f32>, // (L, B, NH, T, T)
    attproj: Vec<f32>, // (L, B, T, C)
    residual2: Vec<f32>, // (L, B, T, C)
    ln2: Vec<f32>, // (L, B, T, C)
    ln2_mean: Vec<f32>, // (L, B, T)
    ln2_rstd: Vec<f32>, // (L, B, T)
    fch: Vec<f32>, // (L, B, T, 4*C)
    fch_gelu: Vec<f32>, // (L, B, T, 4*C)
    fcproj: Vec<f32>, // (L, B, T, C)
    residual3: Vec<f32>, // (L, B, T, C)
    lnf: Vec<f32>, // (B, T, C)
    lnf_mean: Vec<f32>, // (B, T)
    lnf_rstd: Vec<f32>, // (B, T)
    logits: Vec<f32>, // (B, T, V)
    probs: Vec<f32>, // (B, T, V)
    losses: Vec<f32>, // (B, T)
}


/*Since Rust doesn't have implicit nullability and raw pointers, we often use owned types like Vec<T> for dynamic arrays and manage explicit lifetimes where necessary.
*/
struct GPT2 {
    config: GPT2Config,
    params: ParameterTensors,
    param_sizes: Vec<usize>,
    params_memory: Vec<f32>,
    num_parameters: usize,
    grads: ParameterTensors,
    grads_memory: Vec<f32>,
    m_memory: Vec<f32>,
    v_memory: Vec<f32>,
    acts: ActivationTensors,
    acts_memory: Vec<f32>,
    num_activations: usize,
    grads_acts: ActivationTensors,
    grads_acts_memory: Vec<f32>,
    batch_size: usize,
    seq_len: usize,
    inputs: Vec<i32>,
    targets: Vec<i32>,
    mean_loss: f32,
}

impl GPT2 {
    fn new() -> Self {
        GPT2 {
            config: GPT2Config {
                max_seq_len: 0,
                vocab_size: 0,
                num_layers: 0,
                num_heads: 0,
                channels: 0,
            },
            params: ParameterTensors {
                wte: Vec::new(),
                wpe: Vec::new(),
                ln1w: Vec::new(),
                ln1b: Vec::new(),
                qkvw: Vec::new(),
                qkvb: Vec::new(),
                attprojw: Vec::new(),
                attprojb: Vec::new(),
                ln2w: Vec::new(),
                ln2b: Vec::new(),
                fcw: Vec::new(),
                fcb: Vec::new(),
                fcprojw: Vec::new(),
                fcprojb: Vec::new(),
                lnfw: Vec::new(),
                lnfb: Vec::new(),
            },
            param_sizes: vec![0; NUM_PARAMETER_TENSORS],
            params_memory: Vec::new(),
            num_parameters: 0,
            grads: ParameterTensors {
                wte: Vec::new(),
                wpe: Vec::new(),
                ln1w: Vec::new(),
                ln1b: Vec::new(),
                qkvw: Vec::new(),
                qkvb: Vec::new(),
                attprojw: Vec::new(),
                attprojb: Vec::new(),
                ln2w: Vec::new(),
                ln2b: Vec::new(),
                fcw: Vec::new(),
                fcb: Vec::new(),
                fcprojw: Vec::new(),
                fcprojb: Vec::new(),
                lnfw: Vec::new(),
                lnfb: Vec::new(),
            },
            grads_memory: Vec::new(),
            m_memory: Vec::new(),
            v_memory: Vec::new(),
            acts: ActivationTensors {
                encoded: Vec::new(),
                ln1: Vec::new(),
                ln1_mean: Vec::new(),
                ln1_rstd: Vec::new(),
                qkv: Vec::new(),
                atty: Vec::new(),
                preatt: Vec::new(),
                att: Vec::new(),
                attproj: Vec::new(),
                residual2: Vec::new(),
                ln2: Vec::new(),
                ln2_mean: Vec::new(),
                ln2_rstd: Vec::new(),
                fch: Vec::new(),
                fch_gelu: Vec::new(),
                fcproj: Vec::new(),
                residual3: Vec::new(),
                lnf: Vec::new(),
                lnf_mean: Vec::new(),
                lnf_rstd: Vec::new(),
                logits: Vec::new(),
                probs: Vec::new(),
                losses: Vec::new(),
            },
            acts_memory: Vec::new(),
            num_activations: 0,
            grads_acts: ActivationTensors {
                encoded: Vec::new(),
                ln1: Vec::new(),
                ln1_mean: Vec::new(),
                ln1_rstd: Vec::new(),
                qkv: Vec::new(),
                atty: Vec::new(),
                preatt: Vec::new(),
                att: Vec::new(),
                attproj: Vec::new(),
                residual2: Vec::new(),
                ln2: Vec::new(),
                ln2_mean: Vec::new(),
                ln2_rstd: Vec::new(),
                fch: Vec::new(),
                fch_gelu: Vec::new(),
                fcproj: Vec::new(),
                residual3: Vec::new(),
                lnf: Vec::new(),
                lnf_mean: Vec::new(),
                lnf_rstd: Vec::new(),
                logits: Vec::new(),
                probs: Vec::new(),
                losses: Vec::new(),
            },
            grads_acts_memory: Vec::new(),
            batch_size: 0,
            seq_len: 0,
            inputs: Vec::new(),
            targets: Vec::new(),
            mean_loss: -1.0,
        }
    }
    /* UTILITY FUNCTION TO INITIALIZE ACTIVITY TENSOR */
    fn allocate_activation_tensors(&mut self, b: usize, t: usize, l: usize, nh: usize, c: usize, v: usize) {
        self.acts.encoded.resize(b * t * c, 0.0);
        self.acts.ln1.resize(l * b * t * c, 0.0);
        self.acts.ln1_mean.resize(l * b * t, 0.0);
        self.acts.ln1_rstd.resize(l * b * t, 0.0);
        self.acts.qkv.resize(l * b * t * 3 * c, 0.0);
        self.acts.atty.resize(l * b * t * c, 0.0);
        self.acts.preatt.resize(l * b * nh * t * t, 0.0);
        self.acts.att.resize(l * b * nh * t * t, 0.0);
        self.acts.attproj.resize(l * b * t * c, 0.0);
        self.acts.residual2.resize(l * b * t * c, 0.0);
        self.acts.ln2.resize(l * b * t * c, 0.0);
        self.acts.ln2_mean.resize(l * b * t, 0.0);
        self.acts.ln2_rstd.resize(l * b * t, 0.0);
        self.acts.fch.resize(l * b * t * 4 * c, 0.0);
        self.acts.fch_gelu.resize(l * b * t * 4 * c, 0.0);
        self.acts.fcproj.resize(l * b * t * c, 0.0);
        self.acts.residual3.resize(l * b * t * c, 0.0);
        self.acts.lnf.resize(b * t * c, 0.0);
        self.acts.lnf_mean.resize(b * t, 0.0);
        self.acts.lnf_rstd.resize(b * t, 0.0);
        self.acts.logits.resize(b * t * v, 0.0);
        self.acts.probs.resize(b * t * v, 0.0);
        self.acts.losses.resize(b * t, 0.0);

        self.num_activations =
            self.acts.encoded.len()
            + self.acts.ln1.len()
            + self.acts.ln1_mean.len()
            + self.acts.ln1_rstd.len()
            + self.acts.qkv.len()
            + self.acts.atty.len()
            + self.acts.preatt.len()
            + self.acts.att.len()
            + self.acts.attproj.len()
            + self.acts.residual2.len()
            + self.acts.ln2.len()
            + self.acts.ln2_mean.len()
            + self.acts.ln2_rstd.len()
            + self.acts.fch.len()
            + self.acts.fch_gelu.len()
            + self.acts.fcproj.len()
            + self.acts.residual3.len()
            + self.acts.lnf.len()
            + self.acts.lnf_mean.len()
            + self.acts.lnf_rstd.len()
            + self.acts.logits.len()
            + self.acts.probs.len()
            + self.acts.losses.len();
    }
    /* Allocate grad tensors */
    fn allocate_grad_activation_tensors(&mut self, b: usize, t: usize, l: usize, nh: usize, c: usize, v: usize) {
        self.grads_acts.encoded.resize(b * t * c, 0.0);
        self.grads_acts.ln1.resize(l * b * t * c, 0.0);
        self.grads_acts.ln1_mean.resize(l * b * t, 0.0);
        self.grads_acts.ln1_rstd.resize(l * b * t, 0.0);
        self.grads_acts.qkv.resize(l * b * t * 3 * c, 0.0);
        self.grads_acts.atty.resize(l * b * t * c, 0.0);
        self.grads_acts.preatt.resize(l * b * nh * t * t, 0.0);
        self.grads_acts.att.resize(l * b * nh * t * t, 0.0);
        self.grads_acts.attproj.resize(l * b * t * c, 0.0);
        self.grads_acts.residual2.resize(l * b * t * c, 0.0);
        self.grads_acts.ln2.resize(l * b * t * c, 0.0);
        self.grads_acts.ln2_mean.resize(l * b * t, 0.0);
        self.grads_acts.ln2_rstd.resize(l * b * t, 0.0);
        self.grads_acts.fch.resize(l * b * t * 4 * c, 0.0);
        self.grads_acts.fch_gelu.resize(l * b * t * 4 * c, 0.0);
        self.grads_acts.fcproj.resize(l * b * t * c, 0.0);
        self.grads_acts.residual3.resize(l * b * t * c, 0.0);
        self.grads_acts.lnf.resize(b * t * c, 0.0);
        self.grads_acts.lnf_mean.resize(b * t, 0.0);
        self.grads_acts.lnf_rstd.resize(b * t, 0.0);
        self.grads_acts.logits.resize(b * t * v, 0.0);
        self.grads_acts.probs.resize(b * t * v, 0.0);
        self.grads_acts.losses.resize(b * t, 0.0);
    }

    /* FORWARD PASS */
    pub fn forward(&mut self, inputs: &[i32], targets: Option<&[i32]>, b: usize, t: usize) -> io::Result<()> {
        // Ensure the model is properly initialized
        if self.params_memory.is_empty() {
            self.batch_size = b;
            self.seq_len = t;
            return Err(io::Error::new(io::ErrorKind::Other, "Error: model was not initialized properly."));
        }

        let v = self.config.vocab_size;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let c = self.config.channels;
        // allocate space for all the activations if needed
        if self.acts_memory.is_empty() {
            self.batch_size = b;
            self.seq_len = t;
            // Resize activation tensors based on the current configuration and batch settings
            self.allocate_activation_tensors(b, t, l, nh, c, v);
            self.allocate_grad_activation_tensors(b, t, l, nh, c, v)
        } else {
            // Ensure B and T are not larger than what was previously allocated
            if b > self.batch_size || t > self.seq_len {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "Batch size or sequence length is too large."));
            }
        }

        // Cache the inputs and optionally the targets
        self.inputs = inputs.to_vec();
        //println!("inputs size: {}", self.inputs.len());

        if let Some(targets) = targets {
            self.targets = targets.to_vec();
        }

        // Call encoder_forward
        //let out = vec![0.0; b * t * c]; // Output tensor for the encoder
        let wte = &self.params.wte;
        let wpe = &self.params.wpe;
        // first 10 values of input
        //println!("inputs: {:?}", &inputs[0..10]);
        // print size of wte and wpe
        encoder_forward(&mut self.acts.encoded, &inputs, &wte, &wpe, b, t, c);

        // CHECK RUST VS C
        // let mean_encoded = self.acts.encoded.iter().sum::<f32>() / self.acts.encoded.len() as f32;
        // println!("Mean encoded activation (Rust): {:.6}", mean_encoded);
        // Process each layer
        for l in 0..self.config.num_layers {
            // Get the residual from the previous layer
            let index_base = l * self.batch_size * self.seq_len * self.config.channels; // L*B*T*C

            let next_index_base = (l + 1) * self.batch_size * self.seq_len * self.config.channels; // (L+1)*B*T*C

            let mut residual: Vec<f32> = if l == 0 {
                self.acts.encoded.clone()
            } else {
                // review thig part
                self.acts.residual3[(l - 1)*self.batch_size * self.seq_len * self.config.channels..index_base].to_vec()

            };

            // Access layer-specific parameters
            /* Lesson to write on Medium
            In C we perform the pointer arithmetic
            performs pointer arithmetic to obtain the address of a segment within the ln1w array. This effectively moves the pointer l * C positions forward from the start of the array, which corresponds to the start of the weight matrix for the l-th layer.

            In Rust, direct pointer manipulation like this is generally avoided to maintain safety. Instead, Rust uses slices, which are safer because they maintain bounds-checking and other safety properties. When you write:

            */
            let l_ln1w = &self.params.ln1w[l * self.config.channels..(l + 1) * self.config.channels];
            let l_ln1b = &self.params.ln1b[l * self.config.channels..(l + 1) * self.config.channels];
            let l_qkvw = &self.params.qkvw[l * 3 * self.config.channels * self.config.channels..(l + 1) * 3 * self.config.channels * self.config.channels];
            let l_qkvb = &self.params.qkvb[l * 3 * self.config.channels..(l + 1) * 3 * self.config.channels];
            let l_attprojw = &self.params.attprojw[l * self.config.channels * self.config.channels..(l + 1) * self.config.channels * self.config.channels];
            let l_attprojb = &self.params.attprojb[l * self.config.channels..(l + 1) * self.config.channels];
            let l_ln2w = &self.params.ln2w[l * self.config.channels..(l + 1) * self.config.channels];
            let l_ln2b = &self.params.ln2b[l * self.config.channels..(l + 1) * self.config.channels];
            let l_fcw = &self.params.fcw[l * 4 * self.config.channels * self.config.channels..(l + 1) * 4 * self.config.channels * self.config.channels];
            let l_fcb = &self.params.fcb[l * 4 * self.config.channels..(l + 1) * 4 * self.config.channels];
            let l_fcprojw = &self.params.fcprojw[l * self.config.channels * 4 * self.config.channels..(l + 1) * self.config.channels * 4 * self.config.channels];
            let l_fcprojb = &self.params.fcprojb[l * self.config.channels..(l + 1) * self.config.channels];

            //let base_idx = l * self.batch_size * self.seq_len;
            //let c = self.config.channels;
            let nh = self.config.num_heads;

            // Activation slices for this layer
            let l_ln1 = &mut self.acts.ln1[index_base..next_index_base];
            let l_ln1_mean = &mut self.acts.ln1_mean[l*self.batch_size*self.seq_len..(l+1)*self.batch_size*self.seq_len];
            let l_ln1_rstd = &mut self.acts.ln1_rstd[l*self.batch_size*self.seq_len..(l+1)*self.batch_size*self.seq_len];
            let l_qkv = &mut self.acts.qkv[index_base*3..next_index_base*3];
            let l_atty = &mut self.acts.atty[index_base..next_index_base];
            let l_preatt = &mut self.acts.preatt[l*self.batch_size*nh*self.seq_len*self.seq_len..(l+1)*self.batch_size*nh*self.seq_len*self.seq_len];
            let l_att = &mut self.acts.att[l*self.batch_size*nh*self.seq_len*self.seq_len..(l+1)*self.batch_size*nh*self.seq_len*self.seq_len];
            let l_attproj = &mut self.acts.attproj[index_base..next_index_base];
            let l_residual2 = &mut self.acts.residual2[index_base..next_index_base];
            let l_ln2 = &mut self.acts.ln2[index_base..next_index_base];
            let l_ln2_mean = &mut self.acts.ln2_mean[l*self.batch_size*self.seq_len..(l+1)*self.batch_size*self.seq_len];
            let l_ln2_rstd = &mut self.acts.ln2_rstd[l*self.batch_size*self.seq_len..(l+1)*self.batch_size*self.seq_len];
            let l_fch = &mut self.acts.fch[index_base*4..next_index_base*4];
            let l_fch_gelu = &mut self.acts.fch_gelu[index_base*4..next_index_base*4];
            let l_fcproj: &mut [f32] = &mut self.acts.fcproj[index_base..next_index_base];
            let l_residual3 = &mut self.acts.residual3[index_base..next_index_base];

            // FORWARD PASS
            // println!("Executing layernorm foward pass");
            layernorm_forward(
                 l_ln1,
                 l_ln1_mean,
                 l_ln1_rstd,
                & mut residual,
                &l_ln1w,  // weight for layernorm
                &l_ln1b,  // bias for layernorm
                b,
                t,
                c
            );
            // print the mean value of the l_ln1
            // let mean_ln1 = l_ln1.iter().sum::<f32>() / l_ln1.len() as f32;
            // println!("Mean ln1 activation (Rust): {:.6}", mean_ln1);
            // println!("Executing matmul forward pass");
            // let start = Instant::now();
            matmul_forward(
                l_qkv,
                l_ln1,      // Input
                l_qkvw,     // Weights
                Some(l_qkvb),     // Bias
                b,
                t,
                c,
                3*c
            );
            //print the mean value of l_qkv
            // let mean_qkv = l_qkv.iter().sum::<f32>() / l_qkv.len() as f32;
            // println!("Mean qkv activation (Rust): {:.6}", mean_qkv);
            // let duration = start.elapsed();
            // println!("Function took: {:?}", duration);
            // println!("Executing attention forward pass");
            attention_forward(
                l_atty,
                l_preatt,
                l_att,
                l_qkv,
                b,
                t,
                c,
                nh);
            // print mean of l_atty
            // let mean_atty = l_atty.iter().sum::<f32>() / l_atty.len() as f32;
            // println!("Mean atty activation (Rust): {:.6}", mean_atty);
            // println!("Executing matmul forward pass");
            // let start = Instant::now();
            matmul_forward(
                l_attproj,
                l_atty,
                l_attprojw,
                Some(l_attprojb),
                b,
                t,
                c,
                c);
            //print mean of l_attproj
            // let mean_attproj = l_attproj.iter().sum::<f32>() / l_attproj.len() as f32;
            // println!("Mean attproj activation (Rust): {:.6}", mean_attproj);
            // let duration = start.elapsed();
            // println!("Function took: {:?}", duration);
            // println!("Executing residual forward pass");
            residual_forward(
                l_residual2,
                &residual,
                l_attproj,
                b*t*c);
            //print mean of l_residual2
            // let mean_residual2 = l_residual2.iter().sum::<f32>() / l_residual2.len() as f32;
            // println!("Mean residual2 activation (Rust): {:.6}", mean_residual2);
            // println!("Executing layernorm forward pass");
            layernorm_forward(
                l_ln2,
                l_ln2_mean,
                l_ln2_rstd,
                l_residual2,
                l_ln2w,
                l_ln2b,
                b,
                t,
                c);
            //print mean of l_ln2
            // let mean_ln2 = l_ln2.iter().sum::<f32>() / l_ln2.len() as f32;
            // println!("Mean ln2 activation (Rust): {:.6}", mean_ln2);

            // println!("Executing matmul forward pass");
            // let start = Instant::now();
            // print first 10 elemetns of l_fch
            matmul_forward(
                l_fch,
                l_ln2,
                l_fcw,
                Some(l_fcb),
                b,
                t,
                c,
                4*c);

            // print mean of l_fch
            // let mean_fch = l_fch.iter().sum::<f32>() / (b*t*4*c) as f32;
            // println!("Mean fch activation (Rust): {:.6}", mean_fch);
            //process::exit(0x0100);
            // let duration = start.elapsed();
            // println!("Function took: {:?}", duration);
            // println!("Executing gelu forward pass");
            gelu_forward(
                l_fch_gelu,
                l_fch,
                b*t*4*c);
            // print mean of l_fch_gelu
            // let mean_fch_gelu = l_fch_gelu.iter().sum::<f32>() / l_fch_gelu.len() as f32;
            // println!("Mean fch_gelu activation (Rust): {:.6}", mean_fch_gelu);

            // println!("Executing matmul forward pass");
            // let start = Instant::now();
            matmul_forward(
                l_fcproj,
                l_fch_gelu,
                l_fcprojw,
                Some(l_fcprojb),
                b,
                t,
                4*c,
                c);
            // print mean of l_fcproj
            // let mean_fcproj = l_fcproj.iter().sum::<f32>() / l_fcproj.len() as f32;
            // println!("Mean fcproj activation (Rust): {:.6}", mean_fcproj);

            // let duration = start.elapsed();
            // println!("Function took: {:?}", duration);
            // println!("Executing residual forward pass");
            // print the mean of l_residual3
            // let mean_residual3 = l_residual3.iter().sum::<f32>() / l_residual3.len() as f32;
            // println!("2 Mean residual3 activation (Rust): {:.6}", mean_residual3);
            // // the mean of l_ln2
            // let mean_ln2 = l_ln2.iter().sum::<f32>() / l_ln2.len() as f32;
            // println!("2 Mean ln2 activation (Rust): {:.6}", mean_ln2);
            // // of l_fcproj
            // let mean_fcproj = l_fcproj.iter().sum::<f32>() / l_fcproj.len() as f32;
            // println!("2 Mean fcproj activation (Rust): {:.6}", mean_fcproj);
            residual_forward(
                l_residual3,
                &l_residual2,
                l_fcproj,
                b*t*c);
            // print mean of l_residual3
            // let mean_residual3 = l_residual3.iter().sum::<f32>() / l_residual3.len() as f32;
            // println!("Mean residual3 activation (Rust): {:.6}", mean_residual3);
        }
        // line 758 of C code
        let residual = &mut self.acts.residual3[(self.config.num_layers - 1)*self.batch_size * self.seq_len * self.config.channels..].to_vec();
        layernorm_forward(
            &mut self.acts.lnf,
            &mut self.acts.lnf_mean,
            &mut self.acts.lnf_rstd,
            residual,
            &self.params.lnfw,
            &self.params.lnfb,
            b,
            t,
            c);
        // print mean of self.acts.lnf
        // let mean_lnf = self.acts.lnf.iter().sum::<f32>() / self.acts.lnf.len() as f32;
        // println!("Mean lnf activation (Rust): {:.6}", mean_lnf);
        // let start = Instant::now();
        matmul_forward(&mut self.acts.logits,
            &mut self.acts.lnf,
            & self.params.wte,
            None,
            b,
            t,
            c,
            v);
        // print mean of self.acts.logits
        // let mean_logits = self.acts.logits.iter().sum::<f32>() / self.acts.logits.len() as f32;
        // println!("Mean logits activation (Rust): {:.6}", mean_logits);
        // let duration = start.elapsed();
        // println!("Function took: {:?}", duration);
        softmax_forward(&mut self.acts.probs,
            &self.acts.logits,
            b,
            t,
            v);
        // print mean of self.acts.probs
        // let mean_probs = self.acts.probs.iter().sum::<f32>() / self.acts.probs.len() as f32;
        // println!("Mean probs activation (Rust): {:.6}", mean_probs);
        // line 764
        if let Some(targets) = targets {
            crossentropy_forward(&mut self.acts.losses, &self.acts.probs, targets, b, t, v);
            let mut loss = 0.0;
            for i in 0..b*t {
                loss += self.acts.losses[i];
            }
            self.mean_loss = loss / (b * t) as f32;
        }else{
            self.mean_loss = -1.0;
        }
        Ok(())
    }
    pub fn zero_grad(&mut self) {
        // Using the fill method to set all elements to zero
        self.grads_memory.fill(0.0);
        // Reset the gradients for parameters
        self.grads.wte.fill(0.0);
        self.grads.wpe.fill(0.0);
        self.grads.ln1w.fill(0.0);
        self.grads.ln1b.fill(0.0);
        self.grads.qkvw.fill(0.0);
        self.grads.qkvb.fill(0.0);
        self.grads.attprojw.fill(0.0);
        self.grads.attprojb.fill(0.0);
        self.grads.ln2w.fill(0.0);
        self.grads.ln2b.fill(0.0);
        self.grads.fcw.fill(0.0);
        self.grads.fcb.fill(0.0);
        self.grads.fcprojw.fill(0.0);
        self.grads.fcprojb.fill(0.0);
        self.grads.lnfw.fill(0.0);
        self.grads.lnfb.fill(0.0);
        // Reset gradient activations
        self.grads_acts.encoded.fill(0.0);
        self.grads_acts.ln1.fill(0.0);
        self.grads_acts.ln1_mean.fill(0.0);
        self.grads_acts.ln1_rstd.fill(0.0);
        self.grads_acts.qkv.fill(0.0);
        self.grads_acts.atty.fill(0.0);
        self.grads_acts.preatt.fill(0.0);
        self.grads_acts.att.fill(0.0);
        self.grads_acts.attproj.fill(0.0);
        self.grads_acts.residual2.fill(0.0);
        self.grads_acts.ln2.fill(0.0);
        self.grads_acts.ln2_mean.fill(0.0);
        self.grads_acts.ln2_rstd.fill(0.0);
        self.grads_acts.fch.fill(0.0);
        self.grads_acts.fch_gelu.fill(0.0);
        self.grads_acts.fcproj.fill(0.0);
        self.grads_acts.residual3.fill(0.0);
        self.grads_acts.lnf.fill(0.0);
        self.grads_acts.lnf_mean.fill(0.0);
        self.grads_acts.lnf_rstd.fill(0.0);
        self.grads_acts.logits.fill(0.0);
        self.grads_acts.probs.fill(0.0);
        self.grads_acts.losses.fill(0.0);

        // Alternatively, using an iterator method (commented out since fill is preferable):
        // self.grads_memory.iter_mut().for_each(|g| *g = 0.0);
        // self.grads_acts_memory.iter_mut().for_each(|g| *g = 0.0);
        /* Medium explanation
        To implement the gpt2_zero_grad function in Rust, you will want to reset all gradients to zero.
        This is analogous to setting all elements in an array to zero in C, which is often done using memset. In Rust, you don't typically manipulate memory directly like this, but you would instead use Rust's iterator methods or direct indexing.

        Given your structure where you have grads_memory and grads_acts_memory as vectors of f32, you can iterate over these vectors and set each element to zero. Alternatively, you can use the fill method, which is more succinct and expressive.

        fill Method: The fill method sets all items in the slice to the specified value.
        Iterator Method: It's useful when you need to perform more complex operations during the reset process than just setting a value.
 */
    }
    /* BACKWARD */
    pub fn backward(&mut self) -> io::Result<()> {
        if self.mean_loss == -1.0 {
            return Err(io::Error::new(io::ErrorKind::Other, "Error: must forward with targets before backward"));
        }

        let b = self.batch_size;
        let t = self.seq_len;
        let v = self.config.vocab_size;
        let l = self.config.num_layers;
        let c = self.config.channels;
        let nh = self.config.num_heads;

        if self.grads_memory.is_empty() {
            self.grads_memory.resize(self.num_parameters, 0.0);
            self.allocate_grad_activation_tensors(b, t, l, nh, c, v);
            //self.zero_grad();
        }

        let dloss_mean = 1.0 / (b * t) as f32;
        self.grads_acts.losses.fill(dloss_mean);

        // Backprop from loss
        crossentropy_softmax_backward(
            &mut self.grads_acts.logits,
            &mut self.grads_acts.losses,
            &self.acts.probs,
            &self.targets,
            b,
            t,
            v,
        );
        // // print mean of dlogits
        // let mean_dlogits = self.grads_acts.logits.iter().sum::<f32>() / self.grads_acts.logits.len() as f32;
        // println!("Mean dlogits activation (Rust): {:.6}", mean_dlogits);

        // Backprop into lnf and wte
        {
            // For matmul_backward_blas, we need mut references to grads_acts.lnf and grads.wte.
            // grads.wte is separate from grads_acts, so it's safe.
            // grads_acts.logits is already mutable borrowed.

            // We'll clone grads_acts.lnf and grads_acts.logits into locals.
            let mut local_lnf = self.grads_acts.lnf.clone();
            let mut local_logits = self.grads_acts.logits.clone();

            matmul_backward_blas(
                &mut local_lnf,
                &mut self.grads.wte,
                None,
                &mut local_logits,
                &self.acts.lnf,
                &self.params.wte,
                b,
                t,
                c,
                v,
            );
            // // print mean of local_lnf
            // let mean_local_logits = local_logits.iter().sum::<f32>() / local_logits.len() as f32;
            // println!("Mean local_lnf activation (Rust): {:.6}", mean_local_logits);

            // Copy results back
            self.grads_acts.lnf.copy_from_slice(&local_lnf);
            self.grads_acts.logits.copy_from_slice(&local_logits);
        }

        // Backprop through final layernorm
        {
            let residual_start = (l - 1) * b * t * c;
            let residual_end = residual_start + b * t * c;

            // layernorm_backward modifies dresidual, dweight, dbias, dout
            // dresidual = grads_acts.residual3 slice
            // dout = grads_acts.lnf slice
            // dweight, dbias = grads.lnfw, grads.lnfb (no overlap)
            // inp = acts.residual3 (read-only), weight = params.lnfw (read-only)

            let residual = &self.acts.residual3[residual_start..residual_end];

            // Clone the slices we need to mutate
            let mut local_dresidual = self.grads_acts.residual3[residual_start..residual_end].to_vec();
            let mut local_dlnf = self.grads_acts.lnf.to_vec(); // because it's also modified as dout

            layernorm_backward(
                &mut local_dresidual,
                &mut self.grads.lnfw,
                &mut self.grads.lnfb,
                &mut local_dlnf,
                residual,
                &self.params.lnfw,
                &self.acts.lnf_mean,
                &self.acts.lnf_rstd,
                b,
                t,
                c,
            );
            // // print mean of local_dresidual
            // let mean_local_dlnf = local_dlnf.iter().sum::<f32>() / local_dlnf.len() as f32;
            // println!("Mean local_dlnf activation (Rust): {:.6}", mean_local_dlnf);

            // Copy updated values back
            self.grads_acts.residual3[residual_start..residual_end].copy_from_slice(&local_dresidual);
            self.grads_acts.lnf.copy_from_slice(&local_dlnf);
        }

        //let start_time = Instant::now();
        // Now handle each layer in reverse order
        for layer_idx in (0..l).rev() {
            let (residual_slice, dresidual_slice_range) = if layer_idx == 0 {
                // first layer: residual is acts.encoded
                (self.acts.encoded.as_slice(), 0..b*t*c)
            } else {
                let start = (layer_idx - 1) * b * t * c;
                let end = start + b * t * c;
                (&self.acts.residual3[start..end], start..end)
            };

            let start_bt_c = layer_idx * b * t * c;
            let end_bt_c = start_bt_c + b * t * c;

            //let start_residual_backward = Instant::now();
            // residual_backward on residual3
            {
                // We need dl_residual2, dl_fcproj, dl_residual3 from grads_acts overlapping
                let mut local_residual2 = self.grads_acts.residual2[start_bt_c..end_bt_c].to_vec();
                let mut local_fcproj = self.grads_acts.fcproj[start_bt_c..end_bt_c].to_vec();
                let mut local_residual3 = self.grads_acts.residual3[start_bt_c..end_bt_c].to_vec();

                residual_backward(&mut local_residual2, &mut local_fcproj, &mut local_residual3, b*t*c);
                // // print mean of local_residual2
                // let mean_local_residual3 = local_residual3.iter().sum::<f32>() / local_residual3.len() as f32;
                // println!("Mean local_residual3 activation (Rust): {:.6}", mean_local_residual3);

                // Copy back
                self.grads_acts.residual2[start_bt_c..end_bt_c].copy_from_slice(&local_residual2);
                self.grads_acts.fcproj[start_bt_c..end_bt_c].copy_from_slice(&local_fcproj);
                self.grads_acts.residual3[start_bt_c..end_bt_c].copy_from_slice(&local_residual3);
            }
            //let end_residual_backward = start_residual_backward.elapsed();
            //println!("Time taken for residual_backward: {:?}", end_residual_backward);

            // fcproj backward + gelu backward + fcw backward
            //let start_fcproj_backward = Instant::now();
            {
                let fch_range = layer_idx * b * t * 4 * c .. (layer_idx + 1)* b * t * 4 * c;

                let mut local_fcproj = self.grads_acts.fcproj[start_bt_c..end_bt_c].to_vec();
                let mut local_fch = self.grads_acts.fch[fch_range.clone()].to_vec();
                let mut local_fch_gelu = self.grads_acts.fch_gelu[fch_range.clone()].to_vec();
                let mut local_ln2 = self.grads_acts.ln2[start_bt_c..end_bt_c].to_vec();

                let l_fch_gelu = &self.acts.fch_gelu[fch_range.clone()];
                let l_ln2 = &self.acts.ln2[start_bt_c..end_bt_c];
                let l_fcprojw = &self.params.fcprojw[layer_idx * c * 4 * c..(layer_idx + 1)*c*4*c];
                let l_fcw = &self.params.fcw[layer_idx * 4 * c * c..(layer_idx + 1)*4*c*c];

                // fcproj backward
                {
                    let mut local_fcprojw = self.grads.fcprojw[layer_idx * c * 4 * c..(layer_idx + 1)*c*4*c].to_vec();
                    let mut local_fcprojb = self.grads.fcprojb[layer_idx * c..(layer_idx + 1)*c].to_vec();

                    matmul_backward_blas(
                        &mut local_fch_gelu,
                        &mut local_fcprojw,
                        Some(&mut local_fcprojb),
                        &mut local_fcproj,
                        l_fch_gelu,
                        l_fcprojw,
                        b, t, 4*c, c
                    );
                    // // print mean of local_fcprojw
                    // let mean_local_fcproj = local_fcproj.iter().sum::<f32>() / local_fcproj.len() as f32;
                    // println!("Mean local_fcproj activation (Rust): {:.6}", mean_local_fcproj);

                    // copy fcprojw, fcprojb back
                    self.grads.fcprojw[layer_idx * c * 4 * c..(layer_idx + 1)*c*4*c].copy_from_slice(&local_fcprojw);
                    self.grads.fcprojb[layer_idx * c..(layer_idx + 1)*c].copy_from_slice(&local_fcprojb);
                }

                // gelu backward
                {
                    let l_fch = &self.acts.fch[fch_range.clone()];
                    gelu_backward(&mut local_fch, l_fch, &mut local_fch_gelu, b*t*4*c);

                    // let mean_local_fch_gelu = local_fch_gelu.iter().sum::<f32>() / local_fch_gelu.len() as f32;
                    // println!("Mean local_fch_gelu activation (Rust): {:.6}", mean_local_fch_gelu);
                }

                // fcw backward
                {
                    let mut local_fcw = self.grads.fcw[layer_idx * 4 * c * c..(layer_idx+1)*4*c*c].to_vec();
                    let mut local_fcb = self.grads.fcb[layer_idx * 4 * c..(layer_idx + 1)*4*c].to_vec();

                    matmul_backward_blas(
                        &mut local_ln2,
                        &mut local_fcw,
                        Some(&mut local_fcb),
                        &mut local_fch,
                        l_ln2,
                        l_fcw,
                        b, t, c, 4*c
                    );
                    // let mean_local_fch = local_fch.iter().sum::<f32>() / local_fch.len() as f32;
                    // println!("Mean local_fch activation (Rust): {:.6}", mean_local_fch);

                    // copy fcw,fcb back
                    self.grads.fcw[layer_idx * 4 * c * c..(layer_idx+1)*4*c*c].copy_from_slice(&local_fcw);
                    self.grads.fcb[layer_idx * 4 * c..(layer_idx + 1)*4*c].copy_from_slice(&local_fcb);
                }

                // copy fcproj, fch, fch_gelu, ln2 back
                self.grads_acts.fcproj[start_bt_c..end_bt_c].copy_from_slice(&local_fcproj);
                self.grads_acts.fch[fch_range.clone()].copy_from_slice(&local_fch);
                self.grads_acts.fch_gelu[fch_range.clone()].copy_from_slice(&local_fch_gelu);
                self.grads_acts.ln2[start_bt_c..end_bt_c].copy_from_slice(&local_ln2);
            }
            //let end_fcproj_backward = start_fcproj_backward.elapsed();
            //println!("Time taken for fcproj_backward: {:?}", end_fcproj_backward);

            // ln2 backward
            //let start_ln2_backward = Instant::now();
            {
                // Similar pattern: clone what ln2 backward modifies
                let mut local_ln2 = self.grads_acts.ln2[start_bt_c..end_bt_c].to_vec();
                let mut local_residual2 = self.grads_acts.residual2[start_bt_c..end_bt_c].to_vec();

                let l_ln2w = &self.params.ln2w[layer_idx * c..(layer_idx + 1)*c];
                let mut local_ln2w = self.grads.ln2w[layer_idx * c..(layer_idx + 1)*c].to_vec();
                let mut local_ln2b = self.grads.ln2b[layer_idx * c..(layer_idx + 1)*c].to_vec();
                let l_residual2 = &self.acts.residual2[start_bt_c..end_bt_c];

                layernorm_backward(
                    &mut local_residual2,
                    &mut local_ln2w,
                    &mut local_ln2b,
                    &mut local_ln2,
                    l_residual2,
                    l_ln2w,
                    &self.acts.ln2_mean[layer_idx*b*t..(layer_idx*b*t+b*t)],
                    &self.acts.ln2_rstd[layer_idx*b*t..(layer_idx*b*t+b*t)],
                    b, t, c
                );

                // copy back
                self.grads_acts.ln2[start_bt_c..end_bt_c].copy_from_slice(&local_ln2);
                self.grads_acts.residual2[start_bt_c..end_bt_c].copy_from_slice(&local_residual2);
                self.grads.ln2w[layer_idx * c..(layer_idx + 1)*c].copy_from_slice(&local_ln2w);
                self.grads.ln2b[layer_idx * c..(layer_idx + 1)*c].copy_from_slice(&local_ln2b);
            }
            //let end_ln2_backward = start_ln2_backward.elapsed();
            //println!("Time taken for ln2_backward: {:?}", end_ln2_backward);

            // residual backward for residual2, attproj
            //let start_residual2_attproj = Instant::now();
            {
                let mut local_attproj = self.grads_acts.attproj[start_bt_c..end_bt_c].to_vec();
                let mut local_residual2 = self.grads_acts.residual2[start_bt_c..end_bt_c].to_vec();
                let dresidual_slice_start = dresidual_slice_range.start;
                let dresidual_slice_end = dresidual_slice_range.end;
                // let mut local_dresidual: Vec<f32> = self.grads_acts.encoded[dresidual_slice_start..dresidual_slice_end].to_vec();
                // // If layer_idx > 0, this should be residual3 slice instead of encoded. Adjust accordingly:
                // if layer_idx > 0 {
                //     local_dresidual = self.grads_acts.residual3[dresidual_slice_start..dresidual_slice_end].to_vec();
                // }
                let mut local_dresidual = if layer_idx == 0 {
                    // layer_idx == 0: use encoded
                    self.grads_acts.encoded[dresidual_slice_range.clone()].to_vec()
                } else {
                    // layer_idx > 0: use residual3
                    self.grads_acts.residual3[dresidual_slice_range.clone()].to_vec()
                };

                residual_backward(&mut local_dresidual, &mut local_attproj, &mut local_residual2, b*t*c);

                // copy back
                if layer_idx == 0 {
                    self.grads_acts.encoded[dresidual_slice_start..dresidual_slice_end].copy_from_slice(&local_dresidual);
                } else {
                    self.grads_acts.residual3[dresidual_slice_start..dresidual_slice_end].copy_from_slice(&local_dresidual);
                }
                self.grads_acts.attproj[start_bt_c..end_bt_c].copy_from_slice(&local_attproj);
                self.grads_acts.residual2[start_bt_c..end_bt_c].copy_from_slice(&local_residual2);
            }
            //let end_residual2_attproj = start_residual2_attproj.elapsed();
            //println!("Time taken for residual2_attproj: {:?}", end_residual2_attproj);

            // attproj backward
            //let start_attproj = Instant::now();
            {
                let mut local_attproj = self.grads_acts.attproj[start_bt_c..end_bt_c].to_vec();
                let mut local_atty = self.grads_acts.atty[start_bt_c..end_bt_c].to_vec();

                let l_attprojw = &self.params.attprojw[layer_idx * c * c..(layer_idx + 1)*c*c];
                let mut local_attprojw = self.grads.attprojw[layer_idx * c * c..(layer_idx + 1)*c*c].to_vec();
                let mut local_attprojb = self.grads.attprojb[layer_idx * c..(layer_idx + 1)*c].to_vec();
                let l_atty = &self.acts.atty[start_bt_c..end_bt_c];

                matmul_backward_blas(
                    &mut local_atty,
                    &mut local_attprojw,
                    Some(&mut local_attprojb),
                    &mut local_attproj,
                    l_atty,
                    l_attprojw,
                    b, t, c, c
                );

                self.grads_acts.attproj[start_bt_c..end_bt_c].copy_from_slice(&local_attproj);
                self.grads_acts.atty[start_bt_c..end_bt_c].copy_from_slice(&local_atty);
                self.grads.attprojw[layer_idx * c * c..(layer_idx+1)*c*c].copy_from_slice(&local_attprojw);
                self.grads.attprojb[layer_idx * c..(layer_idx+1)*c].copy_from_slice(&local_attprojb);
            }
            //let end_attproj = start_attproj.elapsed();
            //println!("Time taken for attproj_backward: {:?}", end_attproj);

            // attention backward
            //let start_attention_backward = Instant::now();
            {
                let qkv_range = layer_idx * b * t * 3 * c .. (layer_idx+1)*b*t*3*c;
                let att_range = layer_idx * b * nh * t * t .. (layer_idx+1)*b*nh*t*t;

                let mut local_qkv = self.grads_acts.qkv[qkv_range.clone()].to_vec();
                let mut local_preatt = self.grads_acts.preatt[att_range.clone()].to_vec();
                let mut local_att = self.grads_acts.att[att_range.clone()].to_vec();
                let mut local_atty = self.grads_acts.atty[start_bt_c..end_bt_c].to_vec();

                let l_qkv = &self.acts.qkv[qkv_range.clone()];
                let l_att = &self.acts.att[att_range.clone()];

                attention_backward(
                    &mut local_qkv,
                    &mut local_preatt,
                    &mut local_att,
                    &mut local_atty,
                    l_qkv,
                    l_att,
                    b, t, c, nh
                );

                self.grads_acts.qkv[qkv_range.clone()].copy_from_slice(&local_qkv);
                self.grads_acts.preatt[att_range.clone()].copy_from_slice(&local_preatt);
                self.grads_acts.att[att_range.clone()].copy_from_slice(&local_att);
                self.grads_acts.atty[start_bt_c..end_bt_c].copy_from_slice(&local_atty);
            }
            //let end_attention_backward = start_attention_backward.elapsed();
            //println!("Time taken for attention_backward: {:?}", end_attention_backward);


            // qkv backward
            {
                let qkv_range = layer_idx * b * t * 3 * c .. (layer_idx+1)*b*t*3*c;

                let mut local_ln1 = self.grads_acts.ln1[start_bt_c..end_bt_c].to_vec();
                let mut local_qkv = self.grads_acts.qkv[qkv_range.clone()].to_vec();

                let l_ln1 = &self.acts.ln1[start_bt_c..end_bt_c];
                let l_qkvw = &self.params.qkvw[layer_idx*3*c*c..(layer_idx+1)*3*c*c];
                let mut local_qkvw = self.grads.qkvw[layer_idx*3*c*c..(layer_idx+1)*3*c*c].to_vec();
                let mut local_qkvb = self.grads.qkvb[layer_idx*3*c..(layer_idx+1)*3*c].to_vec();

                matmul_backward_blas(
                    &mut local_ln1,
                    &mut local_qkvw,
                    Some(&mut local_qkvb),
                    &mut local_qkv,
                    l_ln1,
                    l_qkvw,
                    b, t, c, 3*c
                );

                // Copy back
                self.grads_acts.ln1[start_bt_c..end_bt_c].copy_from_slice(&local_ln1);
                self.grads_acts.qkv[qkv_range].copy_from_slice(&local_qkv);
                self.grads.qkvw[layer_idx*3*c*c..(layer_idx+1)*3*c*c].copy_from_slice(&local_qkvw);
                self.grads.qkvb[layer_idx*3*c..(layer_idx+1)*3*c].copy_from_slice(&local_qkvb);
            }

            // ln1 backward
            {
                let mut local_dresidual = if layer_idx == 0 {
                    self.grads_acts.encoded[dresidual_slice_range.clone()].to_vec()
                } else {
                    self.grads_acts.residual3[dresidual_slice_range.clone()].to_vec()
                };

                let mut local_ln1 = self.grads_acts.ln1[start_bt_c..end_bt_c].to_vec();

                let l_ln1w = &self.params.ln1w[layer_idx*c..(layer_idx+1)*c];
                let mut local_ln1w = self.grads.ln1w[layer_idx*c..(layer_idx+1)*c].to_vec();
                let mut local_ln1b = self.grads.ln1b[layer_idx*c..(layer_idx+1)*c].to_vec();

                layernorm_backward(
                    &mut local_dresidual,
                    &mut local_ln1w,
                    &mut local_ln1b,
                    &mut local_ln1,
                    residual_slice,
                    l_ln1w,
                    &self.acts.ln1_mean[layer_idx*b*t..(layer_idx*b*t+b*t)],
                    &self.acts.ln1_rstd[layer_idx*b*t..(layer_idx*b*t+b*t)],
                    b,t,c
                );

                // copy back
                if layer_idx == 0 {
                    self.grads_acts.encoded[dresidual_slice_range.clone()].copy_from_slice(&local_dresidual);
                } else {
                    self.grads_acts.residual3[dresidual_slice_range.clone()].copy_from_slice(&local_dresidual);
                }
                self.grads_acts.ln1[start_bt_c..end_bt_c].copy_from_slice(&local_ln1);
                self.grads.ln1w[layer_idx*c..(layer_idx+1)*c].copy_from_slice(&local_ln1w);
                self.grads.ln1b[layer_idx*c..(layer_idx+1)*c].copy_from_slice(&local_ln1b);
            }

        }
        //let end_time = start_time.elapsed();
        //println!("Time taken for backward pass: {:?}", end_time);
        //let start_time = Instant::now();
        // Finally, backprop into encoder embeddings
        {
            let local_encoded = self.grads_acts.encoded.clone();
            // wte, wpe are from self.grads, no overlap. Just call normally:
            encoder_backward(
                &mut self.grads.wte,
                &mut self.grads.wpe,
                &local_encoded,
                &self.inputs,
                b,
                t,
                c,
            );
            // If encoder_backward modifies encoded grads in place, copy back as needed.
            self.grads_acts.encoded.copy_from_slice(&local_encoded);

            // let local_encoded_mean = local_encoded.iter().sum::<f32>() / local_encoded.len() as f32;
            // println!("Mean local_encoded activation (Rust): {:.6}", local_encoded_mean);
        }
        //let end_time = start_time.elapsed();
        //println!("Time taken for encoder backward pass: {:?}", end_time);

        Ok(())
    }

    fn update_grads_memory(&mut self) {
        // Ensure `grads_memory` has been allocated and matches the total number of parameters
        if self.grads_memory.len() != self.num_parameters {
            panic!(
                "grads_memory size mismatch: expected {}, got {}",
                self.num_parameters,
                self.grads_memory.len()
            );
        }

        let mut offset = 0; // Tracks the current position in `grads_memory`
        // Helper closure to copy gradients from a specific parameter to `grads_memory`.
        let check_and_copy = | dest: &mut [f32],
                                   src: &[f32],
                                   param_name: &str,
                                   offset: &mut usize | {
            // Ensure there is enough space in `grads_memory`
            if *offset + src.len() > dest.len() {
                panic!(
                    "Attempt to copy gradients for {} exceeds grads_memory bounds: \
                     Offset {}, Source Length {}, grads_memory Length {}",
                    param_name,
                    *offset,
                    src.len(),
                    dest.len()
                );
            }

            for (i, &val) in src.iter().enumerate() {
                // Copy the gradient value into `grads_memory`
                dest[*offset + i] = val;
            }

            // Update the offset for the next parameter
            *offset += src.len();
        };

        // Macro to simplify copying gradients for each parameter
        macro_rules! copy_grad {
            ($param_grad:expr, $param_name:expr) => {
                check_and_copy(
                    &mut self.grads_memory,
                    $param_grad,
                    $param_name,
                    &mut offset,
                );
            };
        }

        // Copy gradients for each parameter in the same order as `params_memory`
        copy_grad!(&self.grads.wte, "wte");
        copy_grad!(&self.grads.wpe, "wpe");
        copy_grad!(&self.grads.ln1w, "ln1w");
        copy_grad!(&self.grads.ln1b, "ln1b");
        copy_grad!(&self.grads.qkvw, "qkvw");
        copy_grad!(&self.grads.qkvb, "qkvb");
        copy_grad!(&self.grads.attprojw, "attprojw");
        copy_grad!(&self.grads.attprojb, "attprojb");
        copy_grad!(&self.grads.ln2w, "ln2w");
        copy_grad!(&self.grads.ln2b, "ln2b");
        copy_grad!(&self.grads.fcw, "fcw");
        copy_grad!(&self.grads.fcb, "fcb");
        copy_grad!(&self.grads.fcprojw, "fcprojw");
        copy_grad!(&self.grads.fcprojb, "fcprojb");
        copy_grad!(&self.grads.lnfw, "lnfw");
        copy_grad!(&self.grads.lnfb, "lnfb");

    }



    pub fn update(&mut self, learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, t: usize) {
        // Lazily allocate m_memory and v_memory if they are empty
        if self.m_memory.is_empty() {
            self.m_memory = vec![0.0; self.num_parameters];
            self.v_memory = vec![0.0; self.num_parameters];
        }

        for i in 0..self.num_parameters {
            let param = self.params_memory[i];
            let grad = self.grads_memory[i];

            // Update the first moment estimate
            let m = beta1 * self.m_memory[i] + (1.0 - beta1) * grad;
            // Update the second moment estimate
            let v = beta2 * self.v_memory[i] + (1.0 - beta2) * grad * grad;
            // Bias-corrected moment estimates
            let m_hat = m / (1.0 - beta1.powi(t as i32));
            let v_hat = v / (1.0 - beta2.powi(t as i32));

            // Update the parameter using AdamW update rule
            self.params_memory[i] -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);

            // Update the moments
            self.m_memory[i] = m;
            self.v_memory[i] = v;
        }

        // After updating params_memory, update individual parameter slices
        // this can be skipped if we're introducing lifetime parameters
        self.update_param_slices();
    }

    fn update_param_slices(&mut self) {
        // Update individual parameter slices after params_memory changes
        let mut offset = 0;
        self.params.wte.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[0]]); offset += self.param_sizes[0];
        self.params.wpe.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[1]]); offset += self.param_sizes[1];
        self.params.ln1w.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[2]]); offset += self.param_sizes[2];
        self.params.ln1b.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[3]]); offset += self.param_sizes[3];
        self.params.qkvw.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[4]]); offset += self.param_sizes[4];
        self.params.qkvb.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[5]]); offset += self.param_sizes[5];
        self.params.attprojw.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[6]]); offset += self.param_sizes[6];
        self.params.attprojb.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[7]]); offset += self.param_sizes[7];
        self.params.ln2w.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[8]]); offset += self.param_sizes[8];
        self.params.ln2b.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[9]]); offset += self.param_sizes[9];
        self.params.fcw.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[10]]); offset += self.param_sizes[10];
        self.params.fcb.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[11]]); offset += self.param_sizes[11];
        self.params.fcprojw.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[12]]); offset += self.param_sizes[12];
        self.params.fcprojb.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[13]]); offset += self.param_sizes[13];
        self.params.lnfw.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[14]]); offset += self.param_sizes[14];
        self.params.lnfb.copy_from_slice(&self.params_memory[offset..offset + self.param_sizes[15]]);
    }

}
/* END OF GPT2 CONFIGURATION */

/* INFERENCE */
fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> String{
    tokenizer.decode(&tokens.to_vec(), true).unwrap()
}

fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545_F491_4F6C_DD1D)) as u32
}

fn random_f32(state: &mut u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16_777_216.0
}

fn sample_mult(probabilities: &[f32], coin: f32) -> usize {
    let mut cdf = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cdf += prob;
        if coin < cdf {
            return i;
        }
    }
    probabilities.len() - 1
}

/* END INFERENCE */

fn gpt2_build_from_checkpoint(model: &mut GPT2, checkpoint_path: &Path) -> io::Result<()> {
    // Open the model file
    let mut file = BufReader::new(File::open(checkpoint_path)?);

    // Read in the model header
    let mut model_header = [0i32; 256];
    for i in 0..256 {
        model_header[i] = file.read_i32::<LittleEndian>()?;
    }

    if model_header[0] != 20240326 {
        return Err(io::Error::new(io::ErrorKind::Other, "Bad magic model file"));
    }
    if model_header[1] != 1 {
        return Err(io::Error::new(io::ErrorKind::Other, "Bad version in model file"));
    }

    // Read in hyperparameters
    let (max_t, v, l, nh, c) = (
        model_header[2] as usize,
        model_header[3] as usize,
        model_header[4] as usize,
        model_header[5] as usize,
        model_header[6] as usize,
    );

    // Setting the hyperparameters
    model.config = GPT2Config {
        max_seq_len: max_t,
        vocab_size: v,
        num_layers: l,
        num_heads: nh,
        channels: c,
    };

    // Calculate and store parameter sizes
    model.param_sizes = vec![
        v * c, // wte
        max_t * c, // wpe
        l * c, // ln1w
        l * c, // ln1b
        l * (3 * c) * c,// qkvw
        l * (3 * c), // qkvb
        l * c * c, // attprojw
        l * c, // attprojb
        l * c, // ln2w
        l * c, // ln2b
        l * (4 * c) * c, // fcw
        l * (4 * c), // fcb
        l * c * (4 * c), // fcprojw
        l * c, // fcprojb
        c, // lnfw
        c, // lnfb
    ];

    let num_parameters: usize = model.param_sizes.iter().sum();
    println!{"Number of parameters: {}", num_parameters};
    model.num_parameters = num_parameters;

    // Allocate space for all parameters and read them in
    model.params_memory = vec![0.0; num_parameters];
    println!("params_memory size: {}", model.params_memory.len());
    for i in 0..num_parameters {
        model.params_memory[i] = file.read_f32::<LittleEndian>()?;
    }
    // littleendian: functionality for reading and writing numbers in either little-endian or big-endian byte order directly to and from byte arrays

    // read all teh input model params ugly implementation
    let mut offset = 0;
    model.params.wte = model.params_memory[offset..offset + model.param_sizes[0]].to_vec(); offset += model.param_sizes[0];
    model.params.wpe = model.params_memory[offset..offset + model.param_sizes[1]].to_vec(); offset += model.param_sizes[1];
    model.params.ln1w = model.params_memory[offset..offset + model.param_sizes[2]].to_vec(); offset += model.param_sizes[2];
    model.params.ln1b = model.params_memory[offset..offset + model.param_sizes[3]].to_vec(); offset += model.param_sizes[3];
    model.params.qkvw = model.params_memory[offset..offset + model.param_sizes[4]].to_vec(); offset += model.param_sizes[4];
    model.params.qkvb = model.params_memory[offset..offset + model.param_sizes[5]].to_vec(); offset += model.param_sizes[5];
    model.params.attprojw = model.params_memory[offset..offset + model.param_sizes[6]].to_vec(); offset += model.param_sizes[6];
    model.params.attprojb = model.params_memory[offset..offset + model.param_sizes[7]].to_vec(); offset += model.param_sizes[7];
    model.params.ln2w = model.params_memory[offset..offset + model.param_sizes[8]].to_vec(); offset += model.param_sizes[8];
    model.params.ln2b = model.params_memory[offset..offset + model.param_sizes[9]].to_vec(); offset += model.param_sizes[9];
    model.params.fcw = model.params_memory[offset..offset + model.param_sizes[10]].to_vec(); offset += model.param_sizes[10];
    model.params.fcb = model.params_memory[offset..offset + model.param_sizes[11]].to_vec(); offset += model.param_sizes[11];
    model.params.fcprojw = model.params_memory[offset..offset + model.param_sizes[12]].to_vec(); offset += model.param_sizes[12];
    model.params.fcprojb = model.params_memory[offset..offset + model.param_sizes[13]].to_vec(); offset += model.param_sizes[13];
    model.params.lnfw = model.params_memory[offset..offset + model.param_sizes[14]].to_vec(); offset += model.param_sizes[14];
    model.params.lnfb = model.params_memory[offset..offset + model.param_sizes[15]].to_vec(); offset += model.param_sizes[15];
    // Initialize other fields to defaults
    model.acts_memory = Vec::new();
    model.grads_memory = Vec::new();
    model.m_memory = Vec::new();
    model.v_memory = Vec::new();
    model.grads_acts_memory = Vec::new();
    model.inputs = Vec::new();
    model.targets = Vec::new();
    model.batch_size = 0;
    model.seq_len = 0;
    model.mean_loss = -1.0; // Indicate no loss calculated yet

    // Allocate grads to the same sizes
    model.grads.wte.resize(model.params.wte.len(), 0.0);
    model.grads.wpe.resize(model.params.wpe.len(), 0.0);
    model.grads.ln1w.resize(model.params.ln1w.len(), 0.0);
    model.grads.ln1b.resize(model.params.ln1b.len(), 0.0);
    model.grads.qkvw.resize(model.params.qkvw.len(), 0.0);
    model.grads.qkvb.resize(model.params.qkvb.len(), 0.0);
    model.grads.attprojw.resize(model.params.attprojw.len(), 0.0);
    model.grads.attprojb.resize(model.params.attprojb.len(), 0.0);
    model.grads.ln2w.resize(model.params.ln2w.len(), 0.0);
    model.grads.ln2b.resize(model.params.ln2b.len(), 0.0);
    model.grads.fcw.resize(model.params.fcw.len(), 0.0);
    model.grads.fcb.resize(model.params.fcb.len(), 0.0);
    model.grads.fcprojw.resize(model.params.fcprojw.len(), 0.0);
    model.grads.fcprojb.resize(model.params.fcprojb.len(), 0.0);
    model.grads.lnfw.resize(model.params.lnfw.len(), 0.0);
    model.grads.lnfb.resize(model.params.lnfb.len(), 0.0);


    Ok(())
}

fn print_model_summary(model: &GPT2) {
    println!("Model Configuration:");
    println!("Max Sequence Length: {}", model.config.max_seq_len);
    println!("Vocabulary Size: {}", model.config.vocab_size);
    println!("Number of Layers: {}", model.config.num_layers);
    println!("Number of Heads: {}", model.config.num_heads);
    println!("Channels: {}", model.config.channels);

    // Print first few elements of params_memory
    println!("First 10 elements of params_memory:");
    model.params_memory.iter().take(10).enumerate().for_each(|(index, &value)| {
        println!("params_memory[{}] = {}", index, value);
    });

    // Print parameter sizes
    println!("Parameter sizes:");
    model.param_sizes.iter().enumerate().for_each(|(index, &size)| {
        println!("param_sizes[{}] = {}", index, size);
    });

}
fn main() -> Result<(), Box<dyn Error>> {
    // Introduce and parse cli argument
    let config = Config::from_args();
    println!(
        "Starting training with learning rate: {}, batch size: {}, epochs: {}",
        config.learning_rate, config.batch_size, config.epochs
    );
    // Set up Rayon to use a specific number of threads, this must be either the same as openblas or 1
    rayon::ThreadPoolBuilder::new().num_threads(16).build_global().unwrap();

    let mut model = GPT2::new();
    let checkpoint_path = Path::new("/Users/stefano.bosisio/Documents/llm.rust/gpt2_124M.bin");
    let _ = gpt2_build_from_checkpoint(&mut model,  &checkpoint_path);
    print_model_summary(&model);

    // debugging
    //print_model_summary(&model);
    let tiny_shakespeare_train: &Path = Path::new("/Users/stefano.bosisio/Documents/llm.rust/data/tiny_shakespeare_train.bin");
    let tiny_shakespeare_val: &Path = Path::new("/Users/stefano.bosisio/Documents/llm.rust/data/tiny_shakespeare_val.bin");
    // initialise B & T
    let b: usize = 4;
    let t: usize = 64; // max 1024
    let val_num_batches = 10;
    // train loader
    let mut train_loader: DataLoader = DataLoader::new(tiny_shakespeare_train, b , t).unwrap();
    // debug print
    println!("Num batches: {}", train_loader.num_batches);
    // val loader
    let mut val_loader: DataLoader = DataLoader::new(tiny_shakespeare_val, b, t).unwrap();

    // training variables
    //let rng_state = 1337;
    const GEN_MAX_LENGTH: usize = 64; // move the const above
    let mut gen_tokens = [0; GEN_MAX_LENGTH];
    let mut rng_state = 1337;

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();

    // init of the model
    model.mean_loss = 0.0;
    let mut train_losses = Vec::new();

    for step in 0..20{
        println!("Step: {}", step);
        // Training step
        //train_loader.reset();
        train_loader.next_batch();
        // print last 10 eleems of the input
        //println!("Last 10 elements of the input: {:?}", &train_loader.inputs[train_loader.inputs.len()-10..]);
        let starter = Instant::now();
        model.forward(&train_loader.inputs, Some(&train_loader.targets), b, t);
        model.zero_grad();
        let end_time = starter.elapsed();
        println!("Time taken for forward pass: {:?}", end_time);
        println!("train loss: {}", model.mean_loss);
        train_losses.push(model.mean_loss);
        //println!("Backward");
        let starter = Instant::now();
        model.backward();
        let end_time = starter.elapsed();
        println!("Time taken for backward pass: {:?}", end_time);
        //let grad_norm: f32 = model.grads_memory.iter().map(|g| g.abs()).sum();
        //println!("Gradient norm: {:.6}", grad_norm);
        // **Critical Step: Update grads_memory**
        model.update_grads_memory();
        model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1);


        // estimate the validation loss
        if step % 20 == 0 {
            let mut val_loss = 0.0;
            val_loader.reset();
            for _ in 0..val_num_batches {
                val_loader.next_batch();
                // print first 10 elements of val_loader.inputs
                // println!("First 10 elements of the input: {:?}", &val_loader.inputs[..10]);
                model.forward(&val_loader.inputs, Some(&val_loader.targets), b, t);
                // println!("!!!! val loss: {}", model.mean_loss);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches as f32;
            println!("!!!! val loss: {}", val_loss);
        }

        // and do inference as well
        if step > 0 && step % 20 == 0 {
            println!("Inference");
            gen_tokens[0] = GPT2_EOT;
            for tt in 1 .. GEN_MAX_LENGTH {
                let gen_tokens_i32: Vec<i32> = gen_tokens[..tt].iter().map(|&x| x as i32).collect();
                model.forward(&gen_tokens_i32, None, 1, tt);

                let probs_start = (tt-1) * model.config.vocab_size;
                let probs = &model.acts.probs[probs_start..probs_start + model.config.vocab_size];

                //let normalized_probs = probs.iter().map(|&p| p / (probs.iter().sum())).collect::<Vec<f32>>();
                let coin = random_f32(&mut rng_state);
                let next_token = sample_mult(&probs, coin);

                gen_tokens[tt] = next_token as u32;
                // if next_token == 50257 {
                //     break;
                // }
            }
            println!("Generated text:");
            for ttt in 0..GEN_MAX_LENGTH {
                let token = gen_tokens[ttt];
                print!("{}, ", token);
                let dec_token = decode_tokens(&tokenizer, &[token]);
                println!("{}", dec_token);
                // for the moment for decoding we can use
                // python enc.decode([tokens])
            }
        }
    }

    let file = File::create("training_losses.txt")?;
    let mut writer = BufWriter::new(file);

    for loss in &train_losses {
        writeln!(writer, "{}", loss)?;
    }

    println!("Losses have been written to training_losses.txt");

    Ok(())
}
