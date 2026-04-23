#!/bin/bash
# Auto-generated training script for exp{{EXP_ID}}
# {{EXP_DESC}}
set -euo pipefail

export EXPERIMENT_NAME="exp{{EXP_ID}}"
mkdir -p logs/${EXPERIMENT_NAME}
LOG_FILE="logs/${EXPERIMENT_NAME}/train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting exp{{EXP_ID}}: {{EXP_DESC}}" | tee $LOG_FILE

python3 train.py \
  --lr {{LEARNING_RATE}} \
  --batch-size {{BATCH_SIZE}} \
  --entropy-coef {{ENTROPY_COEF}} \
  --experiment-name $EXPERIMENT_NAME \
  2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
[ $EXIT_CODE -eq 0 ] && echo "✅ Done" || echo "❌ Failed (code=$EXIT_CODE)"
exit $EXIT_CODE
