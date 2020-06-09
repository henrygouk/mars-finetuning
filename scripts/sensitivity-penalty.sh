#!/bin/bash

set -o xtrace

mul() {
    echo $(echo $1 $2 | awk '{printf "%f", $1*$2}')
}

run_exp() {
    echo -n "$6,"
    python3 finetune.py --dataset=/raid/anon/vgg-pets --network=$1 --quiet --test --reg-norm=$2 --reg-method=$3 --reg-extractor=$(mul $4 $6) --reg-classifier=$(mul $5 $6) --delta-cache=dc/${1}-pets.npy
}

for i in 0.0001 0.001 0.01 0.1 10.0 100.0 1000.0 10000.0
do
    run_exp "resnet101" "inf-op" "penalty" "0.00021916093034877063" "0.0011648743930556243" "$i"
    run_exp "resnet101" "frob" "penalty" "0.03994148696774106" "0.09534244732508584" "$i"
    run_exp "resnet101" "inf-op" "delta" "0.00017987266196887002" "0.16281288722701628" "$i"

    run_exp "enb0" "inf-op" "penalty" "8.360897036453021e-05" "0.2636018626012623" "$i"
    run_exp "enb0" "frob" "penalty" "0.00034208425578714574" "0.19482385270664973" "$i"
    run_exp "enb0" "inf-op" "delta" "0.15480606092251736" "0.17474053423403835" "$i"
done
