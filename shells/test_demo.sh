#!/bin/bash

# PathSearch Demo Testing Script
# This script runs retrieval testing on the demo dataset

# Parse arguments
DEVICE=${DEVICE:-cpu}
MODEL_PATH=${MODEL_PATH:-""}

echo "=========================================="
echo "PathSearch Demo Retrieval Test"
echo "=========================================="
echo "Device: $DEVICE"
if [ -n "$MODEL_PATH" ]; then
    echo "Model: $MODEL_PATH"
else
    echo "Model: Random weights (for testing only)"
fi
echo ""

# Run the test
python src/test/test_demo.py \
    --device $DEVICE \
    --data_dir ./demo_dataset \
    --model_path "$MODEL_PATH" \
    --output ./demo_retrieval_results.csv

echo ""
echo "=========================================="
echo "Test completed!"
echo "Results saved to: demo_retrieval_results.csv"
echo "=========================================="
