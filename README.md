# `candle_graph`

Easy-to-use CUDA graph API for Candle 🔥.

## Features 
- Simple, abstracted API
- Generate `.dot` graphs

## Roadmap
- Support generating graphs for LLMs (🧪 Experimental example [here](https://github.com/EricLBuehler/candle_graphs/blob/master/examples/cache/main.rs)):
    - This will require KV cache support
    - `Cache`/`KvCache` now have an `append_at` that writes at a position read from a device
      tensor at kernel-launch time, so a graph captured around one decode step can be replayed
      at a different KV-cache position each time by updating that tensor between replays --
      see `examples/decode_replay`. This covers the *write* side (advancing the cache across
      replays); a full decode loop still needs the *read* side (attention over a growing
      context) to also take a device-resident length instead of a host `narrow`, which this
      does not add yet.

## Example
```rust
use candle_graph::{Graph, GraphDumpFormat, GraphDumpVerbosity};
use candle_graph_macro::GraphInputItem;

use std::f64::consts::E;

use candle_core::{DType, Device, Tensor};

const SHAPE: (usize, usize) = (32, 32);

#[derive(GraphInputItem)]
struct Inputs {
    x: Tensor,
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;

    let x = Tensor::ones(SHAPE, DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    // Build the graph. The closure here is automatically traced to build the graph.
    let graph = Graph::new(
        |inputs| {
            let x = &inputs.x;
            let out_data = x.matmul(&x)?.log()?;
            y = Some(out_data);
            Ok(())
        },
        &device,
        Inputs { x },
    )?;

    graph.output_dot("out.png", GraphDumpFormat::Png, GraphDumpVerbosity::Verbose)?;

    // Replay the graph. This can be done any number of times.
    let new = Tensor::full(E, SHAPE, &device)?.to_dtype(DType::BF16)?;
    graph.replay(Inputs { x: new })?;

    Ok(())
}
```

Generated `.dot` graph:

<img src="https://github.com/user-attachments/assets/b44887ea-b947-4c86-a9c0-1fc89d4c2495" width="400">
