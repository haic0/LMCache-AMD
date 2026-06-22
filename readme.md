## Objective
The main goal is to evaluate when how to leverage LMcache for the KV cache management, to improve LLM serving performance, especially **time-to-first-token (TTFT)**, for long-context and multi-turn workloads.
## What This Tutorial Covers
- Set up a ROCm/vLLM environment for AMD GPUs
- Build LMCache from source with HIP support
- Serve `Qwen/Qwen3.6-35B-A3B` using vLLM
- Compare three cache configurations
- Run Long Document QA and Multi-round QA benchmarks
- Analyze TTFT, latency, throughput, and cache behavior
## Experiment Configurations
### Experiment A: No Prefix Cache
Baseline configuration.
vLLM recomputes the prompt prefill for every request.
### Experiment B: vLLM HBM Prefix Cache
Uses vLLM's GPU-resident prefix cache.
This is usually effective when the KV working set fits in GPU HBM.
### Experiment C: LMCache CPU DRAM Tier
Uses LMCache with `LMCacheConnectorV1` to offload and reuse KV cache through CPU memory.
This is useful when long-context workloads create KV cache pressure beyond available GPU HBM.
## Benchmark Workloads
### Long Document QA Benchmark
This benchmark sends repeated long-document prompts through the OpenAI-compatible endpoint.
It measures whether repeated document prefixes can reuse KV cache and reduce TTFT.
### Multi-round QA Benchmark
This benchmark simulates multiple users having stateful conversations.
Each later round includes previous conversation history, so the prompt grows over time and tests cache reuse in realistic multi-turn chat workloads.
## Important Configuration Notes
- Set `PYTHONHASHSEED=0` for stable cache keys.
- Use `--enable-prefix-caching` when testing LMCache.
- Avoid `LMCACHE_SAVE_DECODE_CACHE=true`, because it can distort benchmark results.
- Use `--language-model-only` for text-only workloads to reduce GPU memory usage.
- Tune `--max-model-len`, `--tensor-parallel-size`, `--gpu-memory-utilization`, and `LMCACHE_MAX_LOCAL_CPU_SIZE` based on available HBM and CPU DRAM.
## Key Metrics
The notebook compares the following metrics across configurations:
- Average TTFT
- P50 TTFT
- P95 TTFT
- Maximum TTFT
- Input token throughput
- Output token throughput
- LMCache hit behavior

## Recommended Experiment Flow
1. Run Experiment A: No Prefix Cache.
2. Start the server and run the smoke test.
3. Run either the Long Document QA benchmark or the Multi-round QA benchmark.
4. Repeat the same benchmark with Experiment B: vLLM HBM Prefix Cache.
5. Repeat the same benchmark with Experiment C: LMCache CPU DRAM Tier.
6. Compare TTFT, throughput, and cache-hit behavior across all three results.
## Troubleshooting Tips
If the server does not become ready:
- Confirm AMD GPUs are visible with `amd-smi` or `rocm-smi`.
- Reduce `--max-model-len`, `--gpu-memory-utilization`, or tensor parallel size.
- Confirm Hugging Face credentials are available inside the container.
If LMCache shows zero hit tokens:
- Ensure `PYTHONHASHSEED=0` is set everywhere.
- Confirm `--enable-prefix-caching` is enabled.
- Use the exact same model string across requests.
- Confirm repeated prompts share byte-identical prefixes.
If LMCache is slower than vLLM prefix cache:
- The active KV working set may already fit in GPU HBM.
- Increase document length, number of users, number of documents, or benchmark duration to create more HBM pressure.
