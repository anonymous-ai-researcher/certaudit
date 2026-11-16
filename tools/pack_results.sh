#!/usr/bin/env bash
# Pack everything needed for the analysis (no model weights). Usage: bash tools/pack_results.sh NAME
set -uo pipefail
name=${1:?usage: bash tools/pack_results.sh NAME}
out=certaudit_results_${name}_$(date +%Y%m%d_%H%M).tar.gz
find runs \( -name train.json -o -name val.json -o -name 'raw_*.pkl' -o -name noise.json -o -name sweep.json -o -name recall.json -o -name fig1.json -o -name export.json \
     -o -name job.log -o -name selection.json -o -name selfcheck.json -o -name retrain.json \) -print > .pack_list
ls data/*.stats.json 2>/dev/null >> .pack_list
tar -czf "$out" -T .pack_list results jobs logs env_report.txt 2>/dev/null
rm -f .pack_list
echo "wrote $out ($(du -h "$out" | cut -f1))"
