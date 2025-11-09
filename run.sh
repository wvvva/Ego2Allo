#!/bin/bash
FILE_TO_RUN="python run_SAT.py"

PID=11973

eval "$FILE_TO_RUN" --start_index 1150 --end_index 1200
eval "$FILE_TO_RUN" --start_index 1200 --end_index 1250
eval "$FILE_TO_RUN" --start_index 1250 --end_index 1300
eval "$FILE_TO_RUN" --start_index 1300 --end_index 1350
eval "$FILE_TO_RUN" --start_index 1350 --end_index 1400
eval "$FILE_TO_RUN" --start_index 1400 --end_index 1450
eval "$FILE_TO_RUN" --start_index 1450 --end_index 1500



while true; do
if ps -p $PID > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Process $PID is still running."
    sleep 30
else
    scancel -u ydinga
    break
fi
done