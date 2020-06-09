#!/bin/bash

set -o xtrace

mul() {
    echo $(echo $1 $2 | awk '{printf "%f", $1*$2}')
}

run_exp() {
    echo -n "$6,"
    python3 finetune.py --dataset=/raid/anon/vgg-pets --network=$1 --quiet --test --reg-norm=$2 --reg-method=$3 --reg-extractor=$(mul $4 $6) --reg-classifier=$(mul $5 $6)
}

for i in 0.0001 0.001 0.01 0.1 10.0 100.0 1000.0 10000.0
do
    run_exp "resnet101" "inf-op" "constraint" "13.952100641753956" "17.090633659963352" "$i"
    run_exp "resnet101" "frob" "constraint" "1.1961134581246993" "1.7253622107522244" "$i"

    run_exp "enb0" "inf-op" "constraint" "5.951676915312538" "11.952030079391935" "$i"
    run_exp "enb0" "frob" "constraint" "1.7304358752274287" "1.9959420134152572" "$i"
done
