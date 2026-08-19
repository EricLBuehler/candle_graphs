//! Demonstrates `Cache::append_at`: a graph captured once around a single
//! decode step, replayed at a *different* KV-cache position on every call by
//! updating a device-resident position tensor between replays -- instead of
//! `examples/cache`'s `append`, which bakes the write position into the graph
//! at capture time and can only ever write to that one slot on replay.
//!
//! Run with: `cargo run --example decode_replay`

use candle_graph::{cache::KvCache, Graph};
use candle_graph_macro::GraphInputItem;

use candle_core::{DType, Device, Tensor};

const N_HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const MAX_SEQ_LEN: usize = 8;
const N_STEPS: u32 = 5;

#[derive(GraphInputItem)]
struct Inputs {
    k: Tensor,
    v: Tensor,
    position: Tensor,
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;

    let mut cache = KvCache::new(2, MAX_SEQ_LEN);

    let k = Tensor::zeros((1, N_HEADS, 1, HEAD_DIM), DType::F32, &device)?;
    let v = Tensor::zeros((1, N_HEADS, 1, HEAD_DIM), DType::F32, &device)?;
    let position = Tensor::new(&[0u32], &device)?;

    // Allocate the cache before capturing: `append_at` would otherwise allocate on its first
    // call -- which happens inside the closure below -- and the allocation's zero-fill would be
    // captured into the graph, so every replay would wipe the cache before writing its slot.
    cache.reserve(&k, &v)?;

    // The capture records the cache's device addresses, not the cache itself, so the graph has
    // to hold its own reference to the backing buffers -- otherwise dropping `cache` while the
    // graph is still replayable would free the memory each replay keeps writing to. Here `cache`
    // outlives the graph anyway, but a decode loop that owns the graph and drops the cache
    // between sequences would not.
    let retained = [cache.k_cache(), cache.v_cache()]
        .iter()
        .filter_map(|c| c.all_data().clone())
        .collect();

    // Capture once: one decode step, writing at whatever `position` holds when
    // the graph runs -- not at a position fixed here during capture.
    let graph = Graph::new_retaining(
        |inputs: &Inputs| {
            cache.append_at(&inputs.k, &inputs.v, &inputs.position)?;
            Ok(())
        },
        &device,
        Inputs { k, v, position },
        retained,
    )?;

    // Replay it once per decode step, writing a distinguishable value each
    // time and advancing `position` -- no re-capture between steps.
    for step in 0..N_STEPS {
        let value = step as f32 + 1.;
        // `.contiguous()`: `Tensor::full` broadcasts a 1-element storage, and graph inputs
        // are copied in by a flat memcpy of that storage (see `copy_inplace`).
        let k = Tensor::full(value, (1, N_HEADS, 1, HEAD_DIM), &device)?.contiguous()?;
        let v = Tensor::full(-value, (1, N_HEADS, 1, HEAD_DIM), &device)?.contiguous()?;
        let position = Tensor::new(&[step], &device)?;
        graph.replay(Inputs { k, v, position })?;
    }

    // `current_data()`/`current_seq_len()` aren't tracked by `append_at` (see
    // its docs), so read the raw backing buffer and check each replay landed
    // in its own slot instead of all landing on slot 0.
    let k_all = cache
        .k_cache()
        .all_data()
        .as_ref()
        .expect("append_at should have allocated the backing buffer")
        .clone();
    for step in 0..N_STEPS as usize {
        let expected = step as f32 + 1.;
        // Every head and head_dim of this slot should hold this step's value.
        for got in k_all.narrow(2, step, 1)?.flatten_all()?.to_vec1::<f32>()? {
            assert_eq!(
                got, expected,
                "slot {step}: expected {expected} (this step's value), got {got} -- \
                 if this were `expected` for step 0 on every slot, the write position \
                 would still be pinned at capture time instead of tracking replays"
            );
        }
    }
    println!("{N_STEPS} decode steps, each replay landed in its own cache slot: ok");

    Ok(())
}
