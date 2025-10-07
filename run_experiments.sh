#! /usr/bin/bash
source ~/.bashrc
conda activate sensei_proc

# Check if mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <mode>"
    echo "Example: $0 argessays"
    exit 1
fi

MODE=$1

grid_search() {
    local mode=$1
    local count=0
    for learning_rate in 0.001 0.0001 0.00001; do
        for weight_decay in 0.01 0.1; do
            python train.py --epochs 8 --mode "$mode" \
                --learning_rate "$learning_rate" \
                --weight_decay "$weight_decay"
            count=$((count + 1))
            echo "Run $count completed for mode $mode with lr=$learning_rate and wd=$weight_decay"
        done
    done
}

# Run grid search for the specified mode
grid_search "$MODE"
