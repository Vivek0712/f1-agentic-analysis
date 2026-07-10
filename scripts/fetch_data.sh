#!/usr/bin/env bash
# Fetch the Ergast-schema historical dataset (CSV mirror).
# Source schema: Ergast Motor Racing Database (CC BY-NC-SA style terms apply
# to upstream data; this project uses it for non-commercial research/demo).
set -euo pipefail
mkdir -p data/raw && cd data/raw
BASE="https://raw.githubusercontent.com/LucaA777/F1-Data-Analysis/main"
for f in lap_times pit_stops results races drivers constructors circuits qualifying status sprint_results; do
  echo "fetching $f.csv"
  curl -sfO "$BASE/$f.csv"
done
wc -l lap_times.csv pit_stops.csv
