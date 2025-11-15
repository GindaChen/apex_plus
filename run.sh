#!/bin/bash

# APEX+ Experiment Runner Script
# This script provides a convenient way to run APEX+ experiments with common configurations

set -e  # Exit on error

# Default values
MODEL="qwen2.5-32b"
NUM_GPUS_PER_NODE=2
TRACE_FILE="./traces/mooncake/conversation_trace.jsonl"
TTFT_SLO=30000
TPOT_SLO=30000
FREQUENCY=0
MAX_BATCH_SIZE=""
KV_DTYPE=""
WEIGHT_DTYPE=""
ACTIVATION_DTYPE=""
ALL_PLANS=false
DEBUG_LOG=false
DEBUG_LOG_FILE="apex_debug.log"
NUM_REQUESTS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run APEX+ experiments with prefix caching support.

OPTIONS:
    -m, --model MODEL              Model to use (default: qwen2.5-32b)
    -g, --gpus-per-node NUM         Number of GPUs per node (default: 2)
    -t, --trace-file FILE           Path to trace file (default: ./traces/mooncake/conversation_trace.jsonl)
    -n, --num-requests NUM          Limit trace to first N requests (creates temp file)
    --ttft-slo MS                   TTFT SLO in milliseconds (default: 30000)
    --tpot-slo MS                   TPOT SLO in milliseconds (default: 30000)
    -f, --frequency FREQ            GPU frequency in MHz (default: 0, use 810 or 1980 for energy)
    -b, --max-batch-size NUM        Maximum batch size
    --kv-dtype DTYPE                KV cache dtype (e.g., float8)
    --weight-dtype DTYPE             Weight dtype (e.g., float8)
    --activation-dtype DTYPE        Activation dtype (e.g., half)
    -a, --all                       Show all execution plans
    -d, --debug                     Enable debug logging
    --debug-log-file FILE            Debug log file path (default: apex_debug.log)
    -h, --help                      Show this help message

EXAMPLES:
    # Basic run
    $0

    # Run with first 100 requests
    $0 -n 100

    # Run with energy consumption
    $0 -f 810

    # Run with FP8 quantization
    $0 --kv-dtype float8 --weight-dtype float8 --activation-dtype half

    # Run with debug logging
    $0 -d

    # Run with custom trace file
    $0 -t ./traces/mooncake/toolagent_trace.jsonl

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -g|--gpus-per-node)
            NUM_GPUS_PER_NODE="$2"
            shift 2
            ;;
        -t|--trace-file)
            TRACE_FILE="$2"
            shift 2
            ;;
        -n|--num-requests)
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --ttft-slo)
            TTFT_SLO="$2"
            shift 2
            ;;
        --tpot-slo)
            TPOT_SLO="$2"
            shift 2
            ;;
        -f|--frequency)
            FREQUENCY="$2"
            shift 2
            ;;
        -b|--max-batch-size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --kv-dtype)
            KV_DTYPE="$2"
            shift 2
            ;;
        --weight-dtype)
            WEIGHT_DTYPE="$2"
            shift 2
            ;;
        --activation-dtype)
            ACTIVATION_DTYPE="$2"
            shift 2
            ;;
        -a|--all)
            ALL_PLANS=true
            shift
            ;;
        -d|--debug)
            DEBUG_LOG=true
            shift
            ;;
        --debug-log-file)
            DEBUG_LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Build command
CMD="python main.py --model $MODEL --num-gpus-per-node $NUM_GPUS_PER_NODE --trace-file"

# Handle trace file limiting
if [[ -n "$NUM_REQUESTS" ]]; then
    TEMP_TRACE="/tmp/trace_${NUM_REQUESTS}.jsonl"
    echo -e "${YELLOW}Limiting trace to first $NUM_REQUESTS requests...${NC}"
    head -n "$NUM_REQUESTS" "$TRACE_FILE" > "$TEMP_TRACE"
    CMD="$CMD $TEMP_TRACE"
    echo -e "${GREEN}Created temporary trace: $TEMP_TRACE${NC}"
else
    CMD="$CMD $TRACE_FILE"
fi

# Add SLOs
CMD="$CMD --ttft-slo $TTFT_SLO --tpot-slo $TPOT_SLO"

# Add frequency if specified
if [[ "$FREQUENCY" != "0" ]]; then
    CMD="$CMD --frequency $FREQUENCY"
fi

# Add max batch size if specified
if [[ -n "$MAX_BATCH_SIZE" ]]; then
    CMD="$CMD --max-batch-size $MAX_BATCH_SIZE"
fi

# Add quantization options if specified
if [[ -n "$KV_DTYPE" ]]; then
    CMD="$CMD --kv-dtype $KV_DTYPE"
fi
if [[ -n "$WEIGHT_DTYPE" ]]; then
    CMD="$CMD --weight-dtype $WEIGHT_DTYPE"
fi
if [[ -n "$ACTIVATION_DTYPE" ]]; then
    CMD="$CMD --activation-dtype $ACTIVATION_DTYPE"
fi

# Add --all flag if requested
if [[ "$ALL_PLANS" == true ]]; then
    CMD="$CMD --all"
fi

# Set up debug logging if enabled
if [[ "$DEBUG_LOG" == true ]]; then
    export APEX_DEBUG_LOG=1
    export APEX_DEBUG_LOG_FILE="$DEBUG_LOG_FILE"
    echo -e "${GREEN}Debug logging enabled. Log file: $DEBUG_LOG_FILE${NC}"
else
    unset APEX_DEBUG_LOG
    unset APEX_DEBUG_LOG_FILE
fi

# Print configuration
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}APEX+ Experiment Configuration${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Model: $MODEL"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Trace file: $TRACE_FILE"
if [[ -n "$NUM_REQUESTS" ]]; then
    echo "Limited to: $NUM_REQUESTS requests"
fi
echo "TTFT SLO: $TTFT_SLO ms"
echo "TPOT SLO: $TPOT_SLO ms"
if [[ "$FREQUENCY" != "0" ]]; then
    echo "Frequency: $FREQUENCY MHz"
fi
if [[ -n "$MAX_BATCH_SIZE" ]]; then
    echo "Max batch size: $MAX_BATCH_SIZE"
fi
if [[ -n "$KV_DTYPE" ]]; then
    echo "KV dtype: $KV_DTYPE"
fi
if [[ -n "$WEIGHT_DTYPE" ]]; then
    echo "Weight dtype: $WEIGHT_DTYPE"
fi
if [[ -n "$ACTIVATION_DTYPE" ]]; then
    echo "Activation dtype: $ACTIVATION_DTYPE"
fi
if [[ "$ALL_PLANS" == true ]]; then
    echo "Show all plans: Yes"
fi
if [[ "$DEBUG_LOG" == true ]]; then
    echo "Debug logging: Enabled ($DEBUG_LOG_FILE)"
fi
echo -e "${GREEN}========================================${NC}"
echo ""

# Run the command
echo -e "${YELLOW}Running command:${NC}"
echo "$CMD"
echo ""
eval $CMD

# Cleanup temporary trace file if created
if [[ -n "$NUM_REQUESTS" && -f "$TEMP_TRACE" ]]; then
    echo ""
    echo -e "${YELLOW}Cleaning up temporary trace file...${NC}"
    rm -f "$TEMP_TRACE"
fi

echo ""
echo -e "${GREEN}Experiment completed!${NC}"

