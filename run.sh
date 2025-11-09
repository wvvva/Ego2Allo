#!/bin/bash
FILE_TO_RUN="python run_SAT.py"

PID=11865

eval "$FILE_TO_RUN" --start_index 238 --end_index 300
eval "$FILE_TO_RUN" --start_index 300 --end_index 350
eval "$FILE_TO_RUN" --start_index 350 --end_index 400
eval "$FILE_TO_RUN" --start_index 400 --end_index 450
eval "$FILE_TO_RUN" --start_index 450 --end_index 500
eval "$FILE_TO_RUN" --start_index 500 --end_index 550
eval "$FILE_TO_RUN" --start_index 550 --end_index 600
eval "$FILE_TO_RUN" --start_index 600 --end_index 650
eval "$FILE_TO_RUN" --start_index 650 --end_index 700


while true; do
if ps -p $PID > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Process $PID is still running."
else
    scancel -u ydinga
    break
fi