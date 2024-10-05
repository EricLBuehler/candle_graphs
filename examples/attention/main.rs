//! https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
//! https://github.com/pytorch/pytorch/blob/c7b0d4b148cf2e4e68f14193549945e1639bff40/aten/src/ATen/cuda/CUDAGraph.cpp

use std::{ops::Mul, time::Instant};

use candle_nn::{linear_no_bias, Linear, VarBuilder, VarMap};
use cuda_graph::{Graph, GraphDumpFormat, GraphDumpVerbosity};

use candle_core::{DType, Device, Tensor};

const NUM_ATTN_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HIDDEN_SZ: usize = 4096;
const INTERMEDIATE_SZ: usize = 14336;
const HEAD_DIM: usize = HIDDEN_SZ / NUM_ATTN_HEADS;
const KV_GROUPS: usize = NUM_ATTN_HEADS / NUM_KV_HEADS;
const NUM_LAYERS: usize = 32;

const BENCH_N: usize = 100;

struct Model {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    gate: Linear,
    up: Linear,
    down: Linear,
}

fn repeat_kv(x: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

impl Model {
    fn new(vb: &VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            q_proj: linear_no_bias(HIDDEN_SZ, NUM_ATTN_HEADS * HEAD_DIM, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(HIDDEN_SZ, NUM_KV_HEADS * HEAD_DIM, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(HIDDEN_SZ, NUM_KV_HEADS * HEAD_DIM, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(NUM_ATTN_HEADS * HEAD_DIM, HIDDEN_SZ, vb.pp("o_proj"))?,
            gate: linear_no_bias(HIDDEN_SZ, INTERMEDIATE_SZ, vb.pp("gate"))?,
            up: linear_no_bias(HIDDEN_SZ, INTERMEDIATE_SZ, vb.pp("up"))?,
            down: linear_no_bias(INTERMEDIATE_SZ, HIDDEN_SZ, vb.pp("down"))?,
        })
    }
}

impl Model {
    fn forward(&mut self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // MLP
        let xs = {
            let lhs = xs.apply(&self.gate)?.silu()?;
            let rhs = xs.apply(&self.up)?;
            (lhs * rhs)?.apply(&self.down)?
        };

        // Attention
        let (bs, seq_len, _) = xs.dims3()?;
        let q = xs.apply(&self.q_proj)?;
        let k = xs.apply(&self.k_proj)?;
        let v = xs.apply(&self.v_proj)?;

        let q = q
            .reshape((bs, seq_len, NUM_ATTN_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((bs, seq_len, NUM_KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((bs, seq_len, NUM_KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        let k = repeat_kv(k.clone(), KV_GROUPS)?.contiguous()?;
        let v = repeat_kv(v.clone(), KV_GROUPS)?.contiguous()?;
        let mut att = q
            .contiguous()?
            .matmul(&k.t()?.contiguous()?)?
            .mul(1. / (HEAD_DIM as f64).sqrt())?;

        att = candle_nn::ops::softmax_last_dim(&att)?;
        att = att
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape((bs, seq_len, ()))?;

        att.apply(&self.o_proj)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);

    let mut model = Model::new(&vb)?;

    let test_x = Tensor::randn(0., 1., (1, 5, HIDDEN_SZ), &device)?.to_dtype(DType::BF16)?;
    let _ = model.forward(&test_x)?;

    let inputs = Tensor::ones((1, 1, HIDDEN_SZ), DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    let graph = Graph::new(
        || {
            let mut xs = inputs.clone();
            for _ in 0..NUM_LAYERS {
                xs = model.forward(&xs)?;
            }
            y = Some(xs);
            Ok(())
        },
        &device,
        [("inputs", inputs.clone())].into(),
    )?;

    graph.output_dot("out.png", GraphDumpFormat::Png, GraphDumpVerbosity::Verbose)?;

    let mut graph_outputs = Vec::new();
    let mut eager_outputs = Vec::new();

    let start = Instant::now();
    for i in 0..BENCH_N {
        let new = Tensor::full(i as f32, (1, 1, HIDDEN_SZ), &device)?.to_dtype(DType::BF16)?;
        graph.replay([("inputs", &new)].into())?;
        graph_outputs.push(y.as_ref().unwrap().copy()?);
    }
    let graph_duration = Instant::now().duration_since(start);

    let start = Instant::now();
    for i in 0..BENCH_N {
        let mut x = Tensor::full(i as f32, (1, 1, HIDDEN_SZ), &device)?.to_dtype(DType::BF16)?;
        for _ in 0..NUM_LAYERS {
            x = model.forward(&x)?;
        }
        eager_outputs.push(x);
    }
    let eager_duration = Instant::now().duration_since(start);
    println!(
        "Graph run took {}s",
        graph_duration.div_f32(BENCH_N as f32).as_secs_f32()
    );
    println!(
        "Eager run took {}s",
        eager_duration.div_f32(BENCH_N as f32).as_secs_f32()
    );
    println!(
        "Graph is {} faster",
        eager_duration.div_f32(BENCH_N as f32).as_secs_f32()
            / graph_duration.div_f32(BENCH_N as f32).as_secs_f32()
    );
    for (graph, eager) in graph_outputs.into_iter().zip(eager_outputs) {
        dbg!((graph.mean_all()?, eager.mean_all()?));
    }

    Ok(())
}
