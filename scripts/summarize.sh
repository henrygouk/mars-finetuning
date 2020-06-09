#!/bin/bash

# $1 = results file
# $2 = network architecture

cat $1 | grep "$2" | grep "inf,inf" | tsv-summarize -d, --mean 9 --stdev 9 | awk -F , '{printf("%.2f$\\pm$%.2f\n", $1*100, $2*100)}'
cat $1 | grep "$2" | grep "delta" | tsv-summarize -d, --mean 9 --stdev 9 | awk -F , '{printf("%.2f$\\pm$%.2f\n", $1*100, $2*100)}'
cat $1 | grep "$2" | grep "penalty,frob" | tsv-summarize -d, --mean 9 --stdev 9 | awk -F , '{printf("%.2f$\\pm$%.2f\n", $1*100, $2*100)}'
cat $1 | grep "$2" | grep "penalty,inf" | tsv-summarize -d, --mean 9 --stdev 9 | awk -F , '{printf("%.2f$\\pm$%.2f\n", $1*100, $2*100)}'
cat $1 | grep "$2" | grep "constraint,frob" | tsv-summarize -d, --mean 9 --stdev 9 | awk -F , '{printf("%.2f$\\pm$%.2f\n", $1*100, $2*100)}'
cat $1 | grep "$2" | grep "constraint,inf" | grep -v "inf,inf" | tsv-summarize -d, --mean 9 --stdev 9 | awk -F , '{printf("%.2f$\\pm$%.2f\n", $1*100, $2*100)}'

