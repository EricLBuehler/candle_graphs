# `candle_graph`

Easy-to-use CUDA graph API for Candle 🔥.

## Features 
- Simple, abstracted API
- Generate `.dot` graphs

## Example
```rust
use candle_graph::{Graph, GraphDumpFormat, GraphDumpVerbosity};

use std::f64::consts::E;

use candle_core::{DType, Device, Tensor};

const N: usize = 1000;
const SHAPE: (usize, usize) = (32, 32);

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;

    let x = Tensor::ones(SHAPE, DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

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

    for i in 1..=N {
        let new = Tensor::full(E.powi(i as i32), SHAPE, &device)?.to_dtype(DType::BF16)?;
        graph.replay([("x", &new)].into())?;
    }

    Ok(())
}
```

Generated `.dot` graph:

<img src="https://github.com/user-attachments/assets/b44887ea-b947-4c86-a9c0-1fc89d4c2495" width="400">
