//! https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
//! https://github.com/pytorch/pytorch/blob/c7b0d4b148cf2e4e68f14193549945e1639bff40/aten/src/ATen/cuda/CUDAGraph.cpp

use candle_nn::{linear_no_bias, Linear, Module, VarBuilder, VarMap};
use cuda_graph::{Graph, GraphDumpFormat, GraphDumpVerbosity};

use candle_core::{DType, Device, Tensor};

const IN_DIM: usize = 8;
const HIDDEN_DIM: usize = 64;
const OUT_DIM: usize = 8;

struct Model {
    ff1: Linear,
    ff2: Linear,
}

impl Model {
    fn new(vb: &VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            ff1: linear_no_bias(IN_DIM, HIDDEN_DIM, vb.pp("ff1"))?,
            ff2: linear_no_bias(HIDDEN_DIM, OUT_DIM, vb.pp("ff2"))?,
        })
    }
}

impl Module for Model {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.ff1)?.relu()?.apply(&self.ff2)
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

    let new = Tensor::randn(0., 1., (1, IN_DIM), &device)?.to_dtype(DType::BF16)?;
    println!("starting");
    graph.replay([("x", &new)].into())?;
    println!("ending");
    Ok(())
}
