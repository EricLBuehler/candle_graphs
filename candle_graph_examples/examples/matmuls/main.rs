//! https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
//! https://github.com/pytorch/pytorch/blob/c7b0d4b148cf2e4e68f14193549945e1639bff40/aten/src/ATen/cuda/CUDAGraph.cpp

use candle_graph::{Graph, GraphDumpFormat, GraphDumpVerbosity, NodeData};
use candle_graph_macro::GraphInputItem;

use std::{f64::consts::E, time::Instant};

use candle_core::{DType, Device, Tensor};

const N: usize = 1000;
const INNER_N: usize = 25;
const SHAPE: (usize, usize) = (32, 32);

#[derive(GraphInputItem)]
struct Inputs {
    x: Tensor,
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;

    let x = Tensor::ones(SHAPE, DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    let graph = Graph::new(
        |input| {
            let x = &input.x;
            let mut out_data = x.matmul(&x)?;
            for _ in 0..INNER_N {
                out_data = out_data.matmul(&x)?;
            }
            y = Some(out_data);
            Ok(())
        },
        &device,
        Inputs { x },
    )?;

    graph.output_dot("out.png", GraphDumpFormat::Png, GraphDumpVerbosity::Verbose)?;

    println!(
        "Kernel grid dims: {:?}",
        graph
            .nodes()?
            .iter()
            .filter_map(|node| match node.as_ref() {
                NodeData::Kernel { launch_params } => Some((
                    launch_params.grid_dim_x,
                    launch_params.grid_dim_y,
                    launch_params.grid_dim_z
                )),
                _ => None,
            })
            .collect::<Vec<_>>()
    );

    let start = Instant::now();
    for i in 1..=N {
        let new = Tensor::full(E.powi(i as i32), SHAPE, &device)?.to_dtype(DType::BF16)?;
        graph.replay(Inputs { x: new })?;
    }
    let graph_duration = Instant::now().duration_since(start);

    let start = Instant::now();
    for i in 1..=N {
        let x = Tensor::full(E.powi(i as i32), SHAPE, &device)?.to_dtype(DType::BF16)?;
        let mut out_data = x.matmul(&x)?;
        for _ in 0..INNER_N {
            out_data = out_data.matmul(&x)?;
        }
    }
    let eager_duration = Instant::now().duration_since(start);
    println!(
        "Graph run took {}s",
        graph_duration.div_f32(N as f32).as_secs_f32()
    );
    println!(
        "Eager run took {}s",
        eager_duration.div_f32(N as f32).as_secs_f32()
    );
    println!(
        "Graph is {} faster",
        eager_duration.div_f32(N as f32).as_secs_f32()
            / graph_duration.div_f32(N as f32).as_secs_f32()
    );

    Ok(())
}
