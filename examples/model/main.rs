//! https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
//! https://github.com/pytorch/pytorch/blob/c7b0d4b148cf2e4e68f14193549945e1639bff40/aten/src/ATen/cuda/CUDAGraph.cpp

use std::time::Instant;

use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use cuda_graph::{Graph, GraphDumpFormat, GraphDumpVerbosity};

use candle_core::{DType, Device, Tensor};

const IN_DIM: usize = 8;
const HIDDEN_DIM: usize = 64;
const OUT_DIM: usize = 8;

// 2.5-3.5x speedup
// const N_HIDDEN: usize = 32;

// 1.5-2.5x speedup
const N_HIDDEN: usize = 12;

const BENCH_N: usize = 100;

struct Model {
    up: Linear,
    hidden: Vec<Linear>,
    down: Linear,
}

impl Model {
    fn new(vb: &VarBuilder) -> candle_core::Result<Self> {
        let mut hidden_layers = Vec::with_capacity(N_HIDDEN);
        for i in 0..N_HIDDEN {
            hidden_layers.push(linear(HIDDEN_DIM, HIDDEN_DIM, vb.pp("hidden").pp(i))?);
        }
        Ok(Self {
            up: linear(IN_DIM, HIDDEN_DIM, vb.pp("up"))?,
            hidden: hidden_layers,
            down: linear(HIDDEN_DIM, OUT_DIM, vb.pp("down"))?,
        })
    }
}

impl Module for Model {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut hidden_states = xs.apply(&self.up)?.relu()?;

        for layer in &self.hidden {
            hidden_states = hidden_states.apply(layer)?;
        }

        hidden_states.apply(&self.down)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);

    let model = Model::new(&vb)?;

    let x = Tensor::ones((1, IN_DIM), DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    let graph = Graph::new(
        || {
            let out_data = model.forward(&x)?;
            y = Some(out_data);
            Ok(())
        },
        &device,
        [("x", x.clone())].into(),
    )?;

    graph.output_dot("out.png", GraphDumpFormat::Png, GraphDumpVerbosity::Verbose)?;

    let start = Instant::now();
    for _ in 0..BENCH_N {
        let new = Tensor::randn(0., 1., (1, IN_DIM), &device)?.to_dtype(DType::BF16)?;
        graph.replay([("x", &new)].into())?;
    }
    let graph_duration = Instant::now().duration_since(start);

    let start = Instant::now();
    for _ in 0..BENCH_N {
        let x = Tensor::randn(0., 1., (1, IN_DIM), &device)?.to_dtype(DType::BF16)?;
        let _ = model.forward(&x)?;
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

    Ok(())
}
