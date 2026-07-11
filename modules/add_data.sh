#!/bin/sh

cd "$(dirname "$0")"

python3.11 data.py

if [ $(ls ../data/backup | wc -l) -gt 10 ]; then
    echo "More than 10 backup files found. Deleting the oldest one."
    ls -t ../data/backup | tail -n 1 | xargs -I {} rm ../data/backup/{}
fi