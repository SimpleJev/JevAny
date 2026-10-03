"""CUDA graphs for row-mode inference on one GPU.

Row mode runs every question as an independent causal row (state + branch tokens). At batch size one the eager
forward is launch-bound: a 4B hybrid backbone issues thousands of small kernels per request, and the GPU waits on
Python between them. ``RowGraphs`` captures one graph per padded length bucket, all sharing one memory pool, and
replays the smallest bucket that fits each row.

Single-question requests (one row) replay a graph; other requests, rows longer than the largest bucket, and
gradient-enabled calls run the eager path. Rows are right-padded with the pad token and run without an attention
mask. Every layer used by row-mode backbones is causal (full or sliding-window attention, gated delta rule, short
convolution), so tokens placed after the last real token cannot change a real token's hidden state. Pad positions
continue the row's positions, keeping position ids increasing.

Results are not bit-identical to the eager path: padded shapes select different kernels. Replays reuse static
buffers, so calls must not overlap; ``DecisionRuntime`` already serializes inference under its lock.
"""
import time

import torch

# Padded lengths; bucket spacing bounds the padding overhead at 25-50% of the row.
LENGTHS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096,
           5120, 6144, 8192, 10240, 12288, 16384)


class RowGraphs:
    """Captured row forwards for one DecisionModel. Construct, then ``capture()``; the model consults
    ``forward_rows_batch`` before its eager path and ``stats`` reports lengths, capture time and call counts."""

    def __init__(self, model, lengths=LENGTHS, max_tokens=None):
        if model.branch_mode != "rows":
            raise ValueError("CUDA graphs cover row-mode backbones; this model runs packed branches")
        if model.special_embeddings:
            raise ValueError("CUDA graphs do not support adapters with trainable token embeddings")
        devices = {parameter.device for parameter in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("CUDA graphs need the whole model on one CUDA device")
        self.model, self.device = model, next(iter(devices))
        self.window = model.inference_capabilities.context_window
        limit = min(value for value in (self.window, max_tokens) if value is not None) \
            if self.window is not None or max_tokens is not None else None
        self.lengths = tuple(n for n in sorted(set(lengths)) if limit is None or n <= limit)
        if not self.lengths:
            raise ValueError("no CUDA-graph length fits the backbone's context window")
        self.graphs = {}
        # Updated in place, so describe() and benchmark reports read live counts.
        self.stats = {"buckets": list(self.lengths), "capture_seconds": None, "graph_calls": 0, "eager_calls": 0}

    def _forward(self, ids, pos):
        return self.model.lm(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state

    @torch.no_grad()   # not inference_mode: the static inputs are refilled in place on every replay
    def capture(self):
        start = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()
        current = torch.cuda.current_stream(self.device)
        for n in self.lengths:
            ids = torch.full((1, n), self.model.pad_id, dtype=torch.long, device=self.device)
            pos = torch.arange(n, device=self.device).unsqueeze(0)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                for _ in range(2):   # initialise lazily loaded kernels and allocator state outside the capture
                    self._forward(ids, pos)
            current.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self._forward(ids, pos)
            self.graphs[n] = (graph, ids, pos, out)
        torch.cuda.synchronize(self.device)
        self.stats["capture_seconds"] = round(time.perf_counter() - start, 2)
        return self

    def _bucket(self, ids, pos):
        length = len(ids)
        for n in self.lengths:
            if n >= length:
                # Absolute position tables (GPT-2) must also cover the padded tail.
                if self.window is not None and pos[-1] + n - length >= self.window:
                    return None
                return n
        return None

    def _hidden(self, ids, pos, n):
        graph, static_ids, static_pos, out = self.graphs[n]
        length = len(ids)
        static_ids.fill_(self.model.pad_id)
        static_ids[0, :length] = torch.tensor(ids, device=self.device)
        static_pos[0, :length] = torch.tensor(pos, device=self.device)
        static_pos[0, length:] = torch.arange(pos[-1] + 1, pos[-1] + 1 + n - length, device=self.device)
        graph.replay()
        # Readout immediately selects its token vectors and computes fresh logits
        # before another replay can overwrite this view; no full-row FP32 copy.
        return out[0, :length]

    def forward_rows_batch(self, encs):
        """Logits like DecisionModel.forward_rows_batch for a single-question request, or None (run eagerly)."""
        from .model import rows_of
        if len(encs) == 1:
            state, state_pos, rows = rows_of(encs[0])
            if len(rows) == 1:
                row = rows[0]
                ids, pos = state + row["ids"], state_pos + row["pos"]
                n = self._bucket(ids, pos)
                if n is not None:
                    self.stats["graph_calls"] += 1
                    decide, options = len(state) + row["decide"], [len(state) + o for o in row["opts"]]
                    return [[self.model._question_readout(self._hidden(ids, pos, n), decide, options)]]
        self.stats["eager_calls"] += 1
        return None
