# Efficient LLM Serving at Scale with Unified Caching (vLLM + LMCache on AMD)

Companion notebook: **`Efficient LLM Serving at Scale with Unified Caching_revised.ipynb`**.

## Objective

Show how **LMCache** adds a CPU-DRAM KV-cache tier under vLLM so that, when a long-context
working set overflows the GPU KV pool, evicted KV is **reloaded from host memory instead of
recomputed** — cutting **time-to-first-token (TTFT)**. We compare two servers on the same
overflow workload:

| Server | Warm-pass TTFT |
|---|---|
| vLLM prefix cache only | stays high — working set exceeds the GPU pool, recomputed every repeat |
| vLLM prefix cache **+ LMCache** | **drops** — evicted KV reloaded from CPU DRAM |

Model: `google/gemma-4-31B-it` (a sliding-window **hybrid** model). Hardware: one AMD Instinct
GPU (MI300X/MI325X = `gfx942`, MI350X/MI355X = `gfx950`). Runtime: `vllm/vllm-openai-rocm:v0.23.0`.

## Prerequisites (kernel = a ROCm PyTorch + vLLM environment)

A notebook cell runs *inside* the Jupyter kernel — the ROCm + vLLM environment must exist before
you run any cell. A stock host Python won't work. Easiest path (see notebook §0 for the bare-metal
alternative):

```bash
# on the host — the image entrypoint is `vllm serve`, so override it
docker run -it --rm --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -v /path/to/your/models:/models \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.23.0

# inside the container
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Open the printed `http://…:8888/?token=…` URL (or point your notebook client at it as a remote
kernel). You also need the model on disk at `/models/gemma-4-31B-it` — it is Apache-2.0 and
**ungated**, so `hf download google/gemma-4-31B-it --local-dir /models/gemma-4-31B-it` needs no token.

## Install LMCache — from source, with HIP kernels

**On ROCm, `pip install lmcache` is not enough.** The PyPI wheel is CUDA-only: its native
transfer kernels (`c_ops`) require `libcudart` and silently fall back to a slow Python
KV-transfer path (~700 MB/s), which makes CPU offload/reload *slower* than recompute. Build from
source so `BUILD_WITH_HIP=1` compiles native HIP `c_ops` and `setup.py` auto-selects `cupy-rocm`:

```bash
export PATH=/opt/rocm/bin:$PATH
git clone --depth 1 https://github.com/LMCache/LMCache.git && cd LMCache
pip install -r requirements/build.txt
PYTORCH_ROCM_ARCH=gfx950 CXX=hipcc BUILD_WITH_HIP=1 \
  pip install --no-build-isolation .          # gfx942 for MI300X/MI325X
pip install "grpcio==1.78.0"                  # source build can bump it; keep vLLM's pin
```

Verify: `python -c "from lmcache import c_ops; import cupy; ..."` must load `c_ops` natively
(no `libcudart` warning) and report `is_hip = True`.

## Server configuration

Two servers, same model / GPU budget; only the CPU cache tier differs.

- **Baseline (prefix cache only):**
  `vllm serve … --max-model-len auto --gpu-memory-utilization 0.4 --enable-prefix-caching`
- **+ LMCache:** a separate `lmcache server --l1-size-gb 400 …` (CPU DRAM), plus
  `--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"lmcache.mp.port":5555,"lmcache.mp.mq_timeout":900}}'`.

**Pool sizing — the important part:** do **not** pin the KV pool with `--kv-cache-memory-bytes`
or a small `--max-model-len`. On a hybrid model that shrinks the sliding-window pool ~5×. Keep
`--max-model-len auto` and make GPU memory the binding constraint with a lower
`--gpu-memory-utilization`, then size the workload to overflow it.

## Benchmark

`lmcache bench engine` runs the `long-doc-qa` workload. The notebook:

- makes it **decode-bound and deterministic** with `--ignore-eos --ldqa-max-output-length 2048`
  (a prefill-bound run hides the decode/TTFT gap),
- reads the **actual GPU KV pool** (GiB + tokens) from vLLM's own startup log and sizes the
  working set to `OVERFLOW×` that pool — correct for hybrid attention, portable across GPUs,
- runs twice: **PASS 1 primes**, **PASS 2 measures** steady state.

Prefix-cache-only ⇒ PASS 2 ≈ PASS 1 (recompute). + LMCache ⇒ PASS 2 drops (reload from CPU).
Proof: the `lmcache server` log shows `Stored` (cold) then `Retrieved` (warm).

## Key metrics

Mean / P90 / P99 TTFT, decode (output) throughput, prefill (input) throughput, and cache-hit
rate — the last from vLLM `:8000/metrics` (`external_prefix_cache_*` for LMCache; the LMCache
Prometheus endpoint is often disabled in MP builds).

## AMD / ROCm gotchas

- **Build LMCache from source with `BUILD_WITH_HIP=1`** — the PyPI wheel is CUDA-only and falls
  back to a slow Python transfer path (LMCache ends up slower than recompute).
- **Don't pin the KV pool** on hybrid models (`--kv-cache-memory-bytes` / small
  `--max-model-len`) — it shrinks the sliding-window pool ~5×. Lower `--gpu-memory-utilization`.
- **Use `LMCacheMPConnector`** (separate `lmcache server`) on ROCm; the in-engine connector
  faults under concurrency. Raise `lmcache.mp.mq_timeout` for large CPU pools.
- **`PYTHONHASHSEED` is not needed** — MP mode hashes chunks with **blake3** (deterministic
  content hash). It only matters if you force `--hash-algorithm builtin`.
- **Tear down by PID**, not `pkill -f` (which self-matches and orphans EngineCore/Worker
  processes that keep holding GPU memory); confirm the GPU returns to ~0 MiB.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cell 1 verify shows a `libcudart` warning / no `c_ops` | you installed the PyPI wheel — build from source with `BUILD_WITH_HIP=1`. |
| Server won't start: "KV cache … larger than available" | pool pinned too small — use `--max-model-len auto`, lower util, don't pin bytes. |
| Warm pass doesn't drop / low hit rate | working set doesn't overflow, or run is prefill-bound — raise `OVERFLOW`, keep `--ignore-eos` / large output. |
| Connector aborts "register_kv_caches within 300s" | large pinned CPU pool — raise `lmcache.mp.mq_timeout` (≥ 900). |
| `DeviceIPCWrapper` / register hang after a restart | a stale old `lmcache server` held the port — kill it by PID and confirm the port is free. |
| LMCache slower than prefix cache | working set already fits GPU HBM (nothing to reload), or transfer is on the slow CUDA-wheel path. |

> **Status:** install, native transfer, and server wiring are verified on ROCm (MI350X). The
> exact overflow operating point (util, `OVERFLOW`, `--ldqa-*`) is calibrated from the validated
> hybrid-benchmarking methodology and may need one tuning pass on your hardware.
