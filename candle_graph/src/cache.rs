use candle_core::{Result, Tensor};

use crate::CudaTensorExtension;

#[derive(Debug, Clone)]
pub struct Cache {
    // all_data is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a KvCache that has been reseted (as in it will not share
    // its internal state with the cloned instance).
    all_data: Option<Tensor>,
    dim: usize,
    current_seq_len: usize,
    max_seq_len: usize,
    /// Set by [`reserve`](Cache::reserve), which is the caller declaring this cache is about to
    /// be captured into a graph. From then on the allocation may be baked into that graph as a
    /// raw pointer, so [`reset`](Cache::reset) must not free it.
    reserved: bool,
}

impl Cache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            reserved: false,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> &Option<Tensor> {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(d.narrow(self.dim, 0, self.current_seq_len)?),
        };
        Ok(data)
    }

    /// Rewinds to an empty cache.
    ///
    /// Frees the backing buffer, unless [`reserve`](Self::reserve) was called -- see there. Drop
    /// the `Cache` to release a reserved buffer.
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        if !self.reserved {
            self.all_data = None;
        }
    }

    /// Allocates the backing buffer now, if it isn't allocated yet, taking its dtype, device and
    /// non-`dim` sizes from `like` (`dim` itself is sized to `max_seq_len`).
    ///
    /// [`append`](Self::append) and [`append_at`](Self::append_at) do this lazily on their first
    /// call, which is a problem when that first call happens inside a graph capture: the
    /// allocation's zero-fill gets captured along with the write, so *every* replay re-zeroes the
    /// whole cache and only the most recently written slot survives. Call this before capturing.
    ///
    /// This also pins the buffer: [`reset`](Self::reset) stops freeing it, because a graph that
    /// captured a write to this cache keeps only the raw destination pointer and would go on
    /// writing to the freed address. A consequence is that a reserved cache keeps its shape
    /// across `reset`, so appending a differently shaped batch afterwards is an error rather
    /// than a silent reallocation.
    ///
    /// Dropping the `Cache` still frees the buffer, and `reset` cannot help with that. Hand
    /// [`all_data`](Self::all_data) to
    /// [`Graph::new_retaining`](crate::graph::Graph::new_retaining) so the graph holds its own
    /// reference and outliving the cache is safe.
    pub fn reserve(&mut self, like: &Tensor) -> Result<()> {
        self.alloc(like)?;
        // Calling this is the caller saying the buffer is about to be captured. A graph that
        // captures a write to this cache holds nothing but the raw destination pointer -- there
        // is no Rust code on the replay path to notice the tensor was dropped -- so from here on
        // `reset` only rewinds the position instead of freeing the allocation out from under it.
        self.reserved = true;
        Ok(())
    }

    /// The allocation itself, without the [`reserve`](Self::reserve) pin: `append`/`append_at`
    /// call this so that a cache only used outside a capture keeps `reset`'s freeing behaviour.
    fn alloc(&mut self, like: &Tensor) -> Result<()> {
        // Indexing `shape[self.dim]` below would panic on an out-of-range `dim` rather than
        // returning an error; `dim` validates it the way `append` already did before delegating
        // here. `append_at` reaches this without a check of its own.
        let _ = like.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = like.dims().to_vec();
            shape[self.dim] = self.max_seq_len;
            let ad = Tensor::zeros(shape, like.dtype(), like.device())?;
            self.all_data = Some(ad)
        };
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        self.alloc(src)?;
        let ad = self.all_data.as_mut().unwrap();
        if self.current_seq_len + seq_len > self.max_seq_len {
            candle_core::bail!(
                "kv-cache: above max-seq-len {}+{seq_len}>{}",
                self.current_seq_len,
                self.max_seq_len
            )
        }
        // To suport graph
        ad.slice_set_fingerprinted(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }

    /// Like [`append`](Self::append), except the write position on `dim` is read from
    /// `position` (a `U32` tensor holding one element) at kernel-launch time instead of from
    /// `self.current_seq_len`.
    ///
    /// `append` bakes `self.current_seq_len` into the write as a plain kernel-launch argument,
    /// so a graph that captures it can only ever write to that one position on every replay --
    /// there is no Rust code re-run on replay to advance it. Reading the position from a device
    /// tensor instead means only the *pointer* is captured; updating `position`'s contents
    /// between replays (e.g. via [`copy_inplace`](crate::graph::copy_inplace)) advances where
    /// the next replay writes.
    ///
    /// This does not read or advance `self.current_seq_len` -- there is no Rust-side call
    /// between replays to do that against. Tracking the decode position is the caller's
    /// responsibility. `current_data`/`current_seq_len` are not meaningful for a cache driven
    /// through `append_at`; read `all_data()` directly instead.
    ///
    /// A position past the end of the cache does not write out of bounds: the kernel drops the
    /// copy when `*position + src`'s length on `dim` exceeds `max_seq_len` (chosen in
    /// [`new`](Self::new)). That is a memory-safety backstop, not position tracking -- the
    /// dropped write is silent, so a caller that runs past `max_seq_len` loses the step rather
    /// than getting an error, and still has to keep the position in range itself.
    ///
    /// Call [`reserve`](Self::reserve) before capturing if this would otherwise be the first
    /// `append*` on the cache -- see its docs for why a capture-time allocation is a problem.
    pub fn append_at(&mut self, src: &Tensor, position: &Tensor) -> Result<()> {
        self.alloc(src)?;
        let ad = self.all_data.as_ref().unwrap();
        ad.slice_set_fingerprinted_at(src, self.dim, 0, position)
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    k: Cache,
    v: Cache,
}

impl KvCache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = Cache::new(dim, max_seq_len);
        let v = Cache::new(dim, max_seq_len);
        Self { k, v }
    }

    pub fn k_cache(&self) -> &Cache {
        &self.k
    }

    pub fn v_cache(&self) -> &Cache {
        &self.v
    }

    pub fn k_cache_mut(&mut self) -> &mut Cache {
        &mut self.k
    }

    pub fn v_cache_mut(&mut self) -> &mut Cache {
        &mut self.v
    }

    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k.current_data()
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v.current_data()
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let out_k = self.k.current_data()?;
        let out_v = self.v.current_data()?;
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }

    /// Allocates both backing buffers now -- see [`Cache::reserve`], including why this also
    /// stops [`reset`](Self::reset) from freeing them. Call this before capturing a graph that
    /// appends to this cache, so the allocation isn't captured into it.
    pub fn reserve(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        self.k.reserve(k)?;
        self.v.reserve(v)?;
        Ok(())
    }

    /// Like [`append`](Self::append), except the write position is read from `position` at
    /// kernel-launch time -- see [`Cache::append_at`], including what happens when `position`
    /// runs past `max_seq_len`. This is what makes a graph capturing a decode step replayable
    /// at a different position each time, without re-capturing.
    ///
    /// Returns nothing: unlike `append`, there is no meaningful `current_data()` to hand back
    /// (see `append_at`'s docs on why `current_seq_len` isn't tracked here). Reading the
    /// attention context back out for a growing sequence across replays -- as opposed to just
    /// writing the new step in -- needs the read side (e.g. `narrow`) to also take a
    /// device-resident length, which this PR does not add; see the `decode_replay` example for
    /// where that would plug in.
    pub fn append_at(&mut self, k: &Tensor, v: &Tensor, position: &Tensor) -> Result<()> {
        self.k.append_at(k, position)?;
        self.v.append_at(v, position)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use candle_core::{DType, Device, Result, Tensor};

    use super::Cache;

    /// A graph that captured a write to this cache keeps only the raw destination pointer, so
    /// `reset` freeing the buffer would leave every later replay writing to a freed address.
    /// `reserve` is the caller declaring that capture is about to happen, so from then on
    /// `reset` rewinds without deallocating -- observable here as the written slot surviving.
    #[test]
    fn reset_keeps_a_reserved_buffer() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;
        let zeros = Tensor::zeros((1, 1, 1, 2), DType::F32, &device)?;
        let ones = Tensor::ones((1, 1, 1, 2), DType::F32, &device)?;

        let mut reserved = Cache::new(2, 4);
        reserved.reserve(&zeros)?;
        reserved.append(&ones)?;
        reserved.reset();

        assert_eq!(reserved.current_seq_len(), 0);
        let data = reserved
            .all_data()
            .as_ref()
            .expect("reset freed a reserved buffer");
        let slot0 = data.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(slot0, [1., 1.], "reset reallocated a reserved buffer");

        // A cache that was never reserved keeps the original behaviour: nothing captured it,
        // so reset is free to hand the memory back.
        let mut plain = Cache::new(2, 4);
        plain.append(&ones)?;
        assert!(plain.all_data().is_some());
        plain.reset();
        assert!(plain.all_data().is_none());

        Ok(())
    }
}
