"""
The same leaky integrate-and-fire brain as flysim.py, stepped on a GPU.

WHY A SECOND SIMULATOR
flysim.FlyBrain.run is the ground truth and stays so. It gathers the outgoing
synapses of the neurons that fired, so one step costs what the fired cells
cost, and one run of one brain fits a laptop. What it cannot do is run many
brains at once: the plume experiment and the calibration sweeps want tens of
seeds per condition, and each is a separate pass over the same 10.2 million
synapses. Here the synaptic input of a step is one sparse-dense product,
v += W @ (fired * gain), with B independent runs as the B columns of a dense
(n, B) state, so the weights are read once per step for all of them. That is
where the speed-up is; a single run is only a little faster than numpy.

WHAT IS IDENTICAL, IN ORDER, EVERY STEP (flysim.FlyBrain.run, line by line)
leak toward rest; external Poisson kicks to thresh + 1 mV; v[refr > 0] = reset;
fired = (v >= thresh) & (refr <= 0); refr[fired] = refr_steps; v[fired] =
reset; synaptic input; per-population spike counts; refr -= 1. The returned
dict has the same keys with numpy arrays, the '_state' dict has the same three
parts (v float32, refr int32, rng as a numpy bit_generator state), so a state
made by either simulator continues on the other.

RANDOM NUMBERS
The uniform draws that decide the external kicks come from numpy Generators on
the host, one per column, exactly the calls flysim makes (rng.random(k) once
per step), and are copied to the device each step. torch's own generators were
the obvious choice and were not taken: their streams differ from numpy's, so a
state could not cross between the two simulators and no run could be checked
spike for spike against the ground truth. The host draws cost a few hundred
microseconds per step for a batch and overlap with the device's work. The
draws can also be injected (`random_source`), which is how the tests hand both
simulators the same numbers.

PRECISION
float32 membrane and weights, like the CPU. The CPU sums synaptic input in
float64 (np.bincount) and casts; cuSPARSE sums in float32 in its own order, so
sums can differ in the last bit. On the real graph that changes a threshold
crossing once in a very long while and two runs then diverge as any two seeds
do; the tests therefore compare fired sets on graphs whose weights are exact
in float32, and rates within noise on the real brain.

LEARNED WEIGHTS
mushroom.MushroomBody.apply() writes into fb.wdata in place. Here wdata is a
read-only window on the brain's own weight buffer that counts the writes numpy
routes through Python - item assignment, in-place operators, ufuncs with out=,
ufunc.at, fill and put, on the window or a view of it - and every other numpy
path (np.copyto, .flat[k] = v, np.putmask, np.place, sort, assignment through
np.asarray or a plain view) is refused by numpy's own read-only error. run()
pushes to the device when the count moved, and an untracked write cannot leave
the device stale because it cannot happen. The raw path is still there for a
caller who needs the buffer itself: fb.wdata.base, or a plain view whose
flags.writeable the caller sets True (torch.from_numpy(fb.wdata) is such a
path too); those writes are invisible, and push_weights() is their other half.
The wiring can change as well: when indptr, indices or the weight count are
replaced - lesion.lesioned does that to a copy.copy of a brain - the device
operator is rebuilt on the next run, and the copy owns its weights and device
buffers, so neither brain's push can clobber the other's. indptr and indices
are read-only views for the same reason wdata is a read-only window: the
operator is built from them and only their setters can see a change, so an
in-place write there (a rewiring that keeps the count) would step the device
on the old targets while numpy on the same object took the new ones. The
writeable arrays stay on fb.W, scipy's own, which is the raw path.
"""
import warnings
import weakref

import numpy as np
import scipy.sparse as sp
import torch

from flysim import BUILD, FlyBrain, Params

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")

# The spike log is staged on the device in chunks of about this many bytes
# (n * B bools per step) and copied out chunk by chunk; the tests shrink it
# to cross chunk boundaries on a small graph.
LOG_BUFFER_BYTES = 64 << 20


class _TrackedWeights(np.ndarray):
    """
    fb.wdata: a read-only window on the brain's weight buffer that counts the
    writes it lets through.

    The learning circuit writes weights with fb.wdata[pos] = values, and a
    device copy that went stale silently would be the worst kind of bug. The
    write paths numpy routes through Python are intercepted here (__setitem__,
    in-place operators and ufuncs with an out= - the same thing to numpy -,
    ufunc.at, fill and put): each unlocks the window for that one write and
    tells the brain afterwards. Every other path (np.copyto, .flat, putmask,
    place, sort, a plain view or np.asarray of the window) never reaches
    Python, so numpy refuses it on the read-only flag instead: loud, never
    stale. The buffer beneath is the brain's own writeable array, which is
    what lets a view unlock itself. Views of the window carry its owner; a
    fancy-index result or a .copy() is new memory, not the brain's weights,
    and comes back as a plain ndarray so writes to it are nobody's business.
    """

    _owner = None

    def __array_finalize__(self, obj):
        owner = getattr(obj, "_owner", None)
        brain = owner() if owner is not None else None
        buf = getattr(brain, "_wdata_raw", None)
        # a window is only a window when it looks at the brain's buffer; a
        # fancy-index result or a copy is new memory, whatever numpy typed it
        self._owner = owner if buf is not None and np.may_share_memory(self, buf) else None

    def __getitem__(self, key):
        out = super().__getitem__(key)
        if isinstance(out, _TrackedWeights) and out._owner is None:
            return out.view(np.ndarray)          # a copy, not a window
        return out

    def copy(self, order="C"):
        return self.view(np.ndarray).copy(order=order)

    def _unlock(self):
        """Writeable for the one write about to happen; returns the flag to restore."""
        was = bool(self.flags.writeable)
        if not was:
            self.flags.writeable = True          # allowed: the brain's buffer beneath is writeable
        return was

    def _relock(self, was):
        if not was:
            self.flags.writeable = False
        owner = self._owner() if self._owner is not None else None
        if owner is not None:
            owner._wdata_version += 1

    def __setitem__(self, key, value):
        was = self._unlock()
        try:
            super().__setitem__(key, value)
        finally:
            self._relock(was)

    def fill(self, value):
        was = self._unlock()
        try:
            super().fill(value)
        finally:
            self._relock(was)

    def put(self, indices, values, mode="raise"):
        was = self._unlock()
        try:
            super().put(indices, values, mode=mode)
        finally:
            self._relock(was)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        outs = kwargs.get("out", ())
        written = [o for o in outs if isinstance(o, _TrackedWeights)]
        if method == "at" and isinstance(inputs[0], _TrackedWeights):
            written.append(inputs[0])
        flags = [(o, o._unlock()) for o in written]
        try:
            # plain views, taken after the unlock so the ones written to are writeable
            plain = tuple(x.view(np.ndarray) if isinstance(x, _TrackedWeights) else x
                          for x in inputs)
            if outs:
                kwargs["out"] = tuple(o.view(np.ndarray) if isinstance(o, _TrackedWeights) else o
                                      for o in outs)
            return getattr(ufunc, method)(*plain, **kwargs)
        finally:
            for o, was in reversed(flags):
                o._relock(was)


def _drive_arrays(drive, dt):
    """
    The CPU's drive parsing, verbatim: index array and per-step hit probability.

    Kept as one function so the two simulators cannot drift on the one place
    a caller's mistake is caught (the ValueError on mismatched lengths).
    """
    if drive:
        idx_parts, p_parts = [], []
        for k, r in drive.items():
            ii = np.asarray(k, dtype=np.int64)
            rr = np.asarray(r, dtype=np.float32)
            if rr.ndim == 0:
                rr = np.full(len(ii), float(rr), dtype=np.float32)
            elif len(rr) != len(ii):
                raise ValueError(
                    f"drive rates ({len(rr)}) do not match neurons ({len(ii)})")
            idx_parts.append(ii)
            p_parts.append(np.clip(rr * dt / 1000.0, 0.0, 1.0))
        return np.concatenate(idx_parts), np.concatenate(p_parts).astype(np.float32)
    return np.array([], dtype=np.int64), np.array([], dtype=np.float32)


class _Column:
    """One run of a batch: its drive, its random source and its starting state."""

    def __init__(self, n, drive, dt, seed, state, random_source):
        self.ext_idx, self.ext_p = _drive_arrays(drive, dt)
        if state is None:
            self.v = None
            self.refr = None
        else:
            self.v = np.array(state["v"], dtype=np.float32, copy=True)
            self.refr = np.array(state["refr"], dtype=np.int32, copy=True)
            if self.v.shape != (n,) or self.refr.shape != (n,):
                raise ValueError(
                    f"state is for {self.v.shape[0]} neurons, this brain has {n}")
        # The generator is made exactly as flysim makes it, so the same seed
        # (or the same carried state) gives the same draws on both simulators.
        if state is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = state["rng"]
        if random_source is None:
            self.draw = self.rng.random
        elif isinstance(random_source, np.random.Generator):
            # the Generator is carried in the returned state, so it cannot be
            # combined with a state that carries its own: that run would
            # continue the membranes but not the stream, and neither
            # simulator could reproduce it
            if state is not None:
                raise ValueError("a state carries its own generator; to continue its membranes "
                                 "with other draws pass a plain callable (the Generator's .random)")
            self.rng = random_source
            self.draw = random_source.random
        elif callable(random_source):
            self.draw = random_source
        else:
            raise TypeError("random_source must be None, a numpy Generator or a callable "
                            f"k -> k uniform draws, not {type(random_source).__name__}")

    def mask(self):
        """
        Which driven entries are hit this step: one draw per entry, as the CPU
        does, and no call at all for a column without drive, as the CPU does
        not - its generator must end the run in the seeded state.
        """
        if not len(self.ext_idx):
            return self.ext_p > 0.0
        return np.asarray(self.draw(len(self.ext_idx))) < self.ext_p


class FlyBrainGPU(FlyBrain):
    """
    flysim.FlyBrain with run() on a torch device and run_batch() for many seeds.

    It is a FlyBrain: the annotations, where(), the CSC arrays and wdata are
    the parent's, so roam, backroom and plume read the same attributes, and
    FlyBrain.run(gpu_brain, ...) is still the CPU ground truth on the same
    weights - which is how the real-brain test compares the two on one load.
    """

    _wdata_version = 0
    _pushed_version = -1
    _operator_stale = True
    _perm = None
    _wdata_dev = None
    _M = None

    def __init__(self, graph_path=BUILD / "graph.npz", p=Params(), device="cuda"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' asked for but torch.cuda.is_available() is False; "
                               "pass device='cpu' to step the same model on the CPU with torch")
        super().__init__(graph_path, p)
        self._build_operator()

    # ---- weights and wiring ----------------------------------------------

    @property
    def wdata(self):
        return self._wdata

    @wdata.setter
    def wdata(self, value):
        # the brain's own buffer: a copy, so no caller keeps a writeable
        # handle on it by accident, and the window over it is read-only
        raw = np.array(value, dtype=np.float32, order="C", copy=True)
        arr = raw.view(_TrackedWeights)
        arr._owner = weakref.ref(self)
        arr.flags.writeable = False
        self._wdata_raw, self._wdata = raw, arr
        self._wdata_version += 1
        if self._perm is None or self._perm.numel() != len(raw):
            self._operator_stale = True

    @staticmethod
    def _read_only_view(value):
        """
        The CSC index arrays as the brain keeps them: a view numpy refuses to
        write through. The device operator is built from these arrays and
        only their setters mark it stale, so an in-place write (the targets
        of a column permuted, say) would leave the device on the old wiring
        with nothing to say so; the view makes it a ValueError instead. The
        caller's array is untouched, and fb.W still holds scipy's writeable
        ones.
        """
        view = np.asarray(value).view()
        view.flags.writeable = False
        return view

    @property
    def indptr(self):
        return self._indptr

    @indptr.setter
    def indptr(self, value):
        self._indptr = self._read_only_view(value)
        self._operator_stale = True

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = self._read_only_view(value)
        self._operator_stale = True

    def __copy__(self):
        """
        copy.copy(fb), the first line of lesion.lesioned: a brain sharing the
        annotations and, until they are replaced, the CSC index arrays, with
        its own weights and its own device state. Shared device buffers would
        let one brain's push clobber the other's; the operator is built on
        the copy's first run, on whatever wiring it has by then.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._wdata_version, new._pushed_version = 0, -1
        new._perm = new._crow = new._col = new._wdata_dev = new._M = None
        new.wdata = self._wdata_raw
        new._operator_stale = True
        return new

    def _build_operator(self):
        """
        The CSC arrays as a CSR operator on the device.

        Column j of the CPU's CSC holds the targets of presynaptic j, so the
        matrix M with M[indices[k], j] = wdata[k] for k in column j is exactly
        the operator in v += M @ x. cuSPARSE wants it row-compressed; the
        conversion is done with scipy on a matrix whose data is the position
        of each synapse, which yields the permutation that maps wdata's order
        onto the CSR order, so a weight push is one gather. Built at load and
        again on the next run after indptr, indices or the weight count change.
        """
        n, nnz = self.n, len(self._wdata_raw)
        indptr, indices = np.asarray(self._indptr), np.asarray(self._indices)
        if len(indptr) != n + 1 or len(indices) != nnz or (nnz and int(indptr[-1]) != nnz):
            raise ValueError(
                f"indptr ({len(indptr)} entries, last {int(indptr[-1]) if len(indptr) else 'none'}), "
                f"indices ({len(indices)}) and wdata ({nnz}) do not describe one CSC matrix of {n} "
                "neurons: replace all three together, as lesion.lesioned does")
        pos = sp.csc_matrix((np.arange(nnz, dtype=np.int64), indices, indptr), shape=(n, n)).tocsr()
        dev = self.device
        self._perm = torch.from_numpy(pos.data.astype(np.int64)).to(dev)
        self._crow = torch.from_numpy(pos.indptr.astype(np.int32)).to(dev)
        self._col = torch.from_numpy(pos.indices.astype(np.int32)).to(dev)
        self._wdata_dev = torch.empty(nnz, dtype=torch.float32, device=dev)
        self._M = None
        self._operator_stale = False
        self.push_weights()

    def push_weights(self):
        """
        Make the device hold exactly what wdata holds: the operator is rebuilt
        first when the wiring changed, else the weights are copied over.
        run() calls this itself when it saw a write; a caller who wrote
        through the raw buffer calls it.
        """
        if self._operator_stale:
            self._build_operator()
            return
        if self._wdata_dev.numel() != len(self._wdata_raw):
            raise ValueError(f"wdata has {len(self._wdata_raw)} synapses but the device operator was "
                             f"built for {self._wdata_dev.numel()}: indptr, indices and wdata are "
                             "replaced together, and the next run rebuilds")
        self._wdata_dev.copy_(torch.from_numpy(self._wdata_raw))
        values = self._wdata_dev[self._perm]
        with warnings.catch_warnings():
            # torch's "sparse CSR is in beta" notice, once per construction otherwise
            warnings.simplefilter("ignore", UserWarning)
            self._M = torch.sparse_csr_tensor(self._crow, self._col, values,
                                              size=(self.n, self.n), check_invariants=False)
        self._pushed_version = self._wdata_version

    def weights_are_current(self):
        """True when the device holds exactly what wdata holds, on the current wiring (a test's check, not a fast one)."""
        if self._operator_stale or self._wdata_dev is None or self._wdata_dev.numel() != len(self._wdata_raw):
            return False
        return bool(np.array_equal(self._wdata_dev.cpu().numpy(), self._wdata_raw))

    def _operator(self):
        if self._operator_stale or self._pushed_version != self._wdata_version:
            self.push_weights()
        return self._M

    # ---- simulation ------------------------------------------------------

    def run(self, drive, steps, gains=None, record=None, seed=0, spike_log=False,
            state=None, random_source=None):
        """
        FlyBrain.run, same arguments, same returned dict, stepped on the device.

        random_source : None, or a callable k -> k uniform draws in [0, 1)
                        used instead of the seeded generator, or a numpy
                        Generator, which is then the one carried in the
                        returned '_state'. With a callable the returned
                        '_state' carries the untouched generator for `seed`
                        (or the state's own, untouched). A Generator together
                        with a state is refused: the state carries its own.
        """
        return self.run_batch([drive], steps, gains=gains, record=record, seeds=[seed],
                              spike_log=spike_log,
                              states=None if state is None else [state],
                              random_sources=None if random_source is None else [random_source])[0]

    def run_batch(self, drives, steps, gains=None, record=None, seeds=None, spike_log=False,
                  states=None, random_sources=None):
        """
        B independent runs at once; returns a list of B dicts, each what run() returns.

        drives         : a list of B drive dicts, or one dict for every column
        seeds          : a list of B seeds, one int (seed + column), or None (the column)
        states         : None or a list of B '_state' dicts (None entries start at rest)
        random_sources : None or a list of B random sources (see run)

        Column b equals run(drives[b], ..., seed=seeds[b], state=states[b]) spike
        for spike: every column draws from its own generator, and the columns
        never mix except through the one product that reads the weights.
        """
        p, n, dev = self.p, self.n, self.device
        if isinstance(drives, dict) or drives is None:
            B = (len(seeds) if isinstance(seeds, (list, tuple, np.ndarray))
                 else len(states) if states is not None else 1)
            drives = [drives] * B
        B = len(drives)
        if seeds is None:
            seeds = list(range(B))
        elif isinstance(seeds, (int, np.integer)):
            seeds = [int(seeds) + b for b in range(B)]
        if states is None:
            states = [None] * B
        if random_sources is None:
            random_sources = [None] * B
        if not (len(seeds) == len(states) == len(random_sources) == B):
            raise ValueError("drives, seeds, states and random_sources must have one entry per column")
        cols = [_Column(n, d, p.dt, s, st, rs)
                for d, s, st, rs in zip(drives, seeds, states, random_sources)]

        # membrane and refractory state, (n, B), one column per run
        v = torch.full((n, B), p.v_rest, dtype=torch.float32, device=dev)
        refr = torch.zeros((n, B), dtype=torch.int32, device=dev)
        for b, c in enumerate(cols):
            if c.v is not None:
                v[:, b] = torch.from_numpy(c.v).to(dev)
                refr[:, b] = torch.from_numpy(c.refr).to(dev)

        if gains is None:
            gain = None
        else:
            gain = torch.from_numpy(np.asarray(gains)[self.type_code].astype(np.float32)).to(dev)[:, None]

        # every driven (neuron, column) pair, flattened, so one scatter kicks
        # every column; a neuron driven twice in one column needs the OR of
        # its hits, which the accumulate path provides
        E_each = [len(c.ext_idx) for c in cols]
        E = sum(E_each)
        if E:
            rows_np = np.concatenate([c.ext_idx for c in cols])
            cols_np = np.concatenate([np.full(k, b, dtype=np.int64) for b, k in enumerate(E_each)])
            rows = torch.from_numpy(rows_np).to(dev)
            colt = torch.from_numpy(cols_np).to(dev)
            dup = len(np.unique(rows_np * B + cols_np)) != E
            hit_buf = torch.zeros((n, B), dtype=torch.int32, device=dev) if dup else None
            pin = dev.type == "cuda"
        kick = p.v_thresh + 1.0

        record = record or {}
        rec_names = list(record)
        rec_sel = [np.asarray(record[k], dtype=np.int64).ravel() for k in rec_names]
        rec_off = np.cumsum([0] + [len(s) for s in rec_sel])
        rec_idx = torch.from_numpy(np.concatenate(rec_sel) if rec_sel else np.array([], dtype=np.int64)).to(dev)
        rec_counts = torch.zeros((len(rec_idx), B), dtype=torch.int64, device=dev)
        total = torch.zeros(B, dtype=torch.int64, device=dev)
        ever = torch.zeros((n, B), dtype=torch.bool, device=dev)

        if spike_log:
            log_chunk = max(1, min(int(steps), LOG_BUFFER_BYTES // max(1, n * B)))
            log_buf = torch.empty((log_chunk, n, B), dtype=torch.bool, device=dev)
            log_host, li = [], 0

        M = self._operator()
        thresh, rest, reset, decay = p.v_thresh, p.v_rest, p.v_reset, float(self.decay)
        refr_steps = self.refr_steps

        for _ in range(int(steps)):
            # leak toward rest, in the CPU's operation order so the bits agree
            v.sub_(rest).mul_(decay).add_(rest)

            # external Poisson drive, drawn on the host, kicked on the device
            if E:
                mask_np = np.concatenate([c.mask() for c in cols]) if B > 1 else cols[0].mask()
                mask = torch.from_numpy(mask_np)
                if pin:
                    mask = mask.pin_memory().to(dev, non_blocking=True)
                else:
                    mask = mask.to(dev)
                if dup:
                    hit_buf.zero_()
                    hit_buf.index_put_((rows, colt), mask.to(torch.int32), accumulate=True)
                    v.masked_fill_(hit_buf > 0, kick)
                else:
                    v.index_put_((rows, colt), torch.where(mask, kick, v[rows, colt]))

            v.masked_fill_(refr > 0, reset)
            fired = (v >= thresh) & (refr <= 0)

            if spike_log:
                log_buf[li].copy_(fired)
                li += 1
                if li == log_chunk:
                    # copy=True: on the CPU device .cpu() would hand back the
                    # live buffer itself, and the next chunk would overwrite
                    # this one's rows; on CUDA it is the one device-to-host copy
                    log_host.append(log_buf.to("cpu", copy=True).numpy())
                    li = 0

            refr.masked_fill_(fired, refr_steps)
            v.masked_fill_(fired, reset)

            # every outgoing synapse of every fired neuron, all columns at once
            x = fired.to(torch.float32)
            if gain is not None:
                x.mul_(gain)
            v.add_(torch.sparse.mm(M, x))

            total.add_(fired.sum(0))
            ever.logical_or_(fired)
            if len(rec_idx):
                rec_counts.add_(fired.index_select(0, rec_idx).to(torch.int64))

            refr.sub_(1)

        if spike_log and li:
            log_host.append(log_buf[:li].to("cpu", copy=True).numpy())

        # back to the host: (B, n) so each column is one contiguous array
        v_host = v.t().contiguous().cpu().numpy()
        refr_host = refr.t().contiguous().cpu().numpy()
        ever_host = ever.t().contiguous().cpu().numpy()
        rec_host = rec_counts.cpu().numpy()
        total_host = total.cpu().numpy()

        secs = steps * p.dt / 1000.0
        outs = []
        for b, c in enumerate(cols):
            out = {k: rec_host[rec_off[i]:rec_off[i + 1], b] / secs for i, k in enumerate(rec_names)}
            spikes = int(total_host[b])
            out["_total_hz"] = spikes / secs / n
            out["_spikes_per_sec"] = spikes / secs
            out["_fired"] = np.flatnonzero(ever_host[b])
            vb = np.ascontiguousarray(v_host[b])
            out["_mean_mv"] = float(vb.mean())
            out["_state"] = {"v": vb, "refr": np.ascontiguousarray(refr_host[b]),
                             "rng": c.rng.bit_generator.state}
            if spike_log:
                out["_spikes"] = [np.flatnonzero(chunk[i, :, b]).astype(np.int32)
                                  for chunk in log_host for i in range(len(chunk))]
            outs.append(out)
        return outs


def open(device="cuda", graph_path=BUILD / "graph.npz", p=None):
    """
    The brain on the device asked for: 'cuda' or 'cpu' give a FlyBrainGPU
    (torch on that device), 'numpy' gives the original flysim.FlyBrain. One
    place for a caller to choose, so flysim.py itself stays as it is.
    """
    p = Params() if p is None else p
    if device == "numpy":
        return FlyBrain(graph_path, p)
    return FlyBrainGPU(graph_path, p, device=device)
