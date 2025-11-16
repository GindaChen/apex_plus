

head -n1000 /Users/mike/Project/GitHub/apex_plus/traces/mooncake/conversation_trace.jsonl > /tmp/trace.jsonl
python main.py --model qwen2.5-32b --num-gpus-per-node 1 --trace-file /tmp/trace.jsonl --ttft-slo 30000 --tpot-slo 30000 --frequency 1980 --enable-pd-disaggregation



head -n1000 /Users/mike/Project/GitHub/apex_plus/traces/mooncake/conversation_trace.jsonl > /tmp/trace.jsonl
python main.py --model qwen2.5-32b --num-gpus-per-node 2 --trace-file /tmp/trace.jsonl --ttft-slo 30000 --tpot-slo 30000 --frequency 1980 --enable-pd-disaggregation



head -n1000 /Users/mike/Project/GitHub/apex_plus/traces/mooncake/conversation_trace.jsonl > /tmp/trace.jsonl
python main.py --model qwen2.5-32b --num-gpus-per-node 4 --trace-file /tmp/trace.jsonl --ttft-slo 30000 --tpot-slo 30000 --frequency 1980 --enable-pd-disaggregation


head -n1000 /Users/mike/Project/GitHub/apex_plus/traces/mooncake/conversation_trace.jsonl > /tmp/trace.jsonl
python main.py --model qwen2.5-32b --num-gpus-per-node 8 --trace-file /tmp/trace.jsonl --ttft-slo 30000 --tpot-slo 30000 --frequency 1980 --enable-pd-disaggregation