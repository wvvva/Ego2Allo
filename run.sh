#!/bin/bash

# python run_3DSRBench_GRPO.py --model-name sft_4_4
# python run_3DSRBench_GRPO.py --model-name sft_4_8
# python run_3DSRBench_GRPO.py --model-name sft_8_8
python run_3DSRBench_GRPO.py --model-name sft_8_16
scancel -u ydinga

# FILE_TO_RUN="python run_SAT.py"

# PID=24987

# # eval "$FILE_TO_RUN" --start_index 1200 --end_index 1250
# # eval "$FILE_TO_RUN" --start_index 1250 --end_index 1300
# # eval "$FILE_TO_RUN" --start_index 1300 --end_index 1350
# # eval "$FILE_TO_RUN" --start_index 1350 --end_index 1400
# # eval "$FILE_TO_RUN" --start_index 1450 --end_index 1500
# eval "$FILE_TO_RUN" --start_index 1500 --end_index 1550
# eval "$FILE_TO_RUN" --start_index 1550 --end_index 1600
# eval "$FILE_TO_RUN" --start_index 1600 --end_index 1650
# eval "$FILE_TO_RUN" --start_index 1650 --end_index 1700
# eval "$FILE_TO_RUN" --start_index 1700 --end_index 1750


# while true; do
# if ps -p $PID > /dev/null 2>&1; then
#     echo "$(date '+%Y-%m-%d %H:%M:%S') - Process $PID is still running."
#     sleep 30
# else
#     scancel -u ydinga
#     break
# fi
# done