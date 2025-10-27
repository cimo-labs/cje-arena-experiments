#!/bin/bash

for est in dr-cpo tmle mrdr stacked-dr; do
  echo "============================================"
  echo "$est:"
  echo "============================================"
  jq -r "select(.spec.estimator == \"$est\" and .spec.sample_size == 250 and .spec.oracle_coverage == 0.25) | [(.spec.extra.use_weight_calibration // false), (.spec.extra.use_covariates // false), .rmse_vs_oracle] | @tsv" results/all_experiments.jsonl | awk -v est="$est" '{
    key = $1 "_" $2
    sum[key] += $3
    count[key]++
  } END {
    print ""
    if (count["false_false"] > 0 || count["false_true"] > 0) {
      print "WITHOUT weight calibration:"
      if (count["false_false"] > 0) {
        print "  WITHOUT cov: mean =", sum["false_false"]/count["false_false"]
      }
      if (count["false_true"] > 0) {
        print "  WITH cov:    mean =", sum["false_true"]/count["false_true"]
        if (count["false_false"] > 0) {
          pct = ((sum["false_true"]/count["false_true"]) - (sum["false_false"]/count["false_false"])) / (sum["false_false"]/count["false_false"]) * 100
          print "  Change: " pct "%"
        }
      }
    }
    print ""
    if (count["true_false"] > 0 || count["true_true"] > 0) {
      print "WITH weight calibration:"
      if (count["true_false"] > 0) {
        print "  WITHOUT cov: mean =", sum["true_false"]/count["true_false"]
      }
      if (count["true_true"] > 0) {
        print "  WITH cov:    mean =", sum["true_true"]/count["true_true"]
        if (count["true_false"] > 0) {
          pct = ((sum["true_true"]/count["true_true"]) - (sum["true_false"]/count["true_false"])) / (sum["true_false"]/count["true_false"]) * 100
          print "  Change: " pct "%"
        }
      }
    }
  }'
  echo ""
done
