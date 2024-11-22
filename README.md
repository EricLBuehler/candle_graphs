# `candle_graph`

Easy-to-use CUDA graph API for Candle.

## Example
```rust
use candle_graph::{Graph, GraphDumpFormat, GraphDumpVerbosity, NodeData};

use std::{f64::consts::E, time::Instant};

use candle_core::{DType, Device, Tensor};

const N: usize = 1000;
const INNER_N: usize = 25;
const SHAPE: (usize, usize) = (32, 32);

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;

    let x = Tensor::ones(SHAPE, DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    let graph = Graph::new(
        || {
            let mut out_data = x.matmul(&x)?;
            y = Some(out_data);
            Ok(())
        },
        &device,
        [("x", x.clone())].into(),
    )?;

    graph.output_dot("out.png", GraphDumpFormat::Png, GraphDumpVerbosity::Verbose)?;

    for i in 1..=N {
        let new = Tensor::full(E.powi(i as i32), SHAPE, &device)?.to_dtype(DType::BF16)?;
        graph.replay([("x", &new)].into())?;
    }
    
    Ok(())
}
```
