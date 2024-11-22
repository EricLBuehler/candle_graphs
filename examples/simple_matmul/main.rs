use candle_graph::{Graph, GraphDumpFormat, GraphDumpVerbosity};

use std::f64::consts::E;

use candle_core::{DType, Device, Tensor};

const SHAPE: (usize, usize) = (32, 32);

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;

    let x = Tensor::ones(SHAPE, DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    // Build the graph. The closure here is automatically traced to build the graph.
    let graph = Graph::new(
        || {
            let out_data = x.matmul(&x)?.log()?;
            y = Some(out_data);
            Ok(())
        },
        &device,
        [("x", x.clone())].into(),
    )?;

    graph.output_dot("out.png", GraphDumpFormat::Png, GraphDumpVerbosity::Verbose)?;

    // Replay the graph. This can be done any number of times.
    let new = Tensor::full(E, SHAPE, &device)?.to_dtype(DType::BF16)?;
    graph.replay([("x", &new)].into())?;

    Ok(())
}
