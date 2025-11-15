# KV3D Prefix Caching Experiments

This document describes how to run experiments with prefix caching support using mooncake trace files.

## Quick Start

### Basic Command
```bash
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file ./traces/mooncake/conversation_trace.jsonl --ttft-slo 30000 --tpot-slo 30000
```

### Using a Smaller Trace (First N Requests)

To limit the trace to the first N requests, use the `head` command to create a smaller trace file:

```bash
# Method 1: Create a temporary file (works in all shells)
head -100 ./traces/mooncake/conversation_trace.jsonl > /tmp/trace_100.jsonl
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file /tmp/trace_100.jsonl --ttft-slo 30000 --tpot-slo 30000

# Method 2: One-liner with temporary file
head -50 ./traces/mooncake/conversation_trace.jsonl > /tmp/trace_50.jsonl && \
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file /tmp/trace_50.jsonl --ttft-slo 30000 --tpot-slo 30000

# Method 3: Quick testing with first 10 requests
head -10 ./traces/mooncake/conversation_trace.jsonl > /tmp/trace_10.jsonl
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file /tmp/trace_10.jsonl --ttft-slo 30000 --tpot-slo 30000
```

**Examples for different sizes:**
- First 10 requests: `head -10` (for quick testing/debugging)
- First 50 requests: `head -50` (for small experiments)
- First 100 requests: `head -100` (for medium experiments)
- First 1000 requests: `head -1000` (for larger experiments)

### Available Mooncake Trace Files
- `./traces/mooncake/conversation_trace.jsonl`
- `./traces/mooncake/toolagent_trace.jsonl`
- `./traces/mooncake/synthetic_trace.jsonl`
- `./traces/mooncake/mooncake_trace.jsonl`

### Supported Models
- `qwen2.5-32b` - Qwen2.5 32B model
- `llama3-70b` - Llama3 70B model
- Other models from `main.py` SHORTCUT list

## Common Options

### Show All Execution Plans
```bash
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file ./traces/mooncake/conversation_trace.jsonl --all --ttft-slo 30000 --tpot-slo 30000
```

### With Quantization (FP8)
```bash
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file ./traces/mooncake/conversation_trace.jsonl --kv-dtype float8 --weight-dtype float8 --activation-dtype half --ttft-slo 30000 --tpot-slo 30000
```

### With Energy Consumption (Frequency)
```bash
# Energy consumption requires non-zero frequency (810 or 1980 MHz)
# Default frequency=0 will show 0.00 KJ energy consumption
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file ./traces/mooncake/conversation_trace.jsonl --ttft-slo 30000 --tpot-slo 30000 --frequency 1980
```

### With Max Batch Size (Throughput Optimization)
```bash
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file ./traces/mooncake/conversation_trace.jsonl --ttft-slo 30000 --tpot-slo 30000 --max-batch-size 10
```

**Note**: `--ttft-slo 30000` and `--tpot-slo 30000` (30 seconds) are set to maximize throughput by not constraining latency. Adjust these values if you need to optimize for latency instead.

### Quick Testing with Small Trace
```bash
# Test with first 20 requests (useful for debugging)
head -20 ./traces/mooncake/conversation_trace.jsonl > /tmp/trace_20.jsonl
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file /tmp/trace_20.jsonl --ttft-slo 30000 --tpot-slo 30000
```

## Debug Logging

APEX+ supports detailed debug logging for time and energy calculations. This is useful for debugging performance issues and understanding where time/energy is being consumed.

### Enable Debug Logging

Set the `APEX_DEBUG_LOG` environment variable to `1` to enable logging:

```bash
# Enable debug logging (logs to apex_debug.log by default)
export APEX_DEBUG_LOG=1
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file ./traces/mooncake/conversation_trace.jsonl --ttft-slo 30000 --tpot-slo 30000

# Or specify a custom log file
export APEX_DEBUG_LOG=1
export APEX_DEBUG_LOG_FILE=my_debug.log
head -10 ./traces/mooncake/conversation_trace.jsonl > /tmp/trace_10.jsonl
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file /tmp/trace_10.jsonl --ttft-slo 30000 --tpot-slo 30000 --frequency 1980
```

### What Gets Logged

When debug logging is enabled, the following information is logged to the file:
- **GEMM operations**: Time and energy for matrix multiplications (interpolation and extrapolation)
- **Task-level execution**: Time and energy for each task type (MHAHead, SwiGLUFilter, etc.)
- **Cell-level execution**: Accumulated time and energy per cell
- **Stage-level execution**: Time and energy scaling with number of blocks
- **Sub-simulation iterations**: Time and energy increments per iteration
- **Model replica execution**: Time and energy accumulation across replicas
- **Alerts**: Warnings when negative time or energy values are detected

### Environment Variables

- `APEX_DEBUG_LOG`: Set to `1` to enable debug logging, `0` or unset to disable (default: disabled)
- `APEX_DEBUG_LOG_FILE`: Path to log file (default: `apex_debug.log` in current directory)

**Note**: Debug logging can generate large log files, especially for long traces. Use with caution on large experiments.

## Trace Format

Mooncake trace files use the following JSON format:
```json
{"timestamp": 0, "input_length": 6758, "output_length": 500, "hash_ids": [0, 1, 2, 3, ...]}
```

- `timestamp`: Arrival time in milliseconds
- `input_length`: Number of input tokens
- `output_length`: Number of output tokens to generate
- `hash_ids`: List of cached block IDs (each block = 256 tokens)

Prefix caching is automatically enabled when `hash_ids` are present in the trace file.

