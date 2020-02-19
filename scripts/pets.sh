#!/bin/bash

set -o xtrace

for i in 1 2 3 4 5
do
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=resnet101 --quiet --test
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=resnet101 --quiet --test --reg-method=delta --delta-cache=dc/resnet101-pets.npy --reg-extractor=0.00017987266196887002 --reg-classifier=0.16281288722701628
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=2.615742426422671 --reg-classifier=9.008723864410545
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.00021916093034877063 --reg-classifier=0.0011648743930556243
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=resnet101 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=1.1961134581246993 --reg-classifier=1.7253622107522244
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=resnet101 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.03994148696774106 --reg-classifier=0.09534244732508584

    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=enb0 --quiet --test
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=enb0 --quiet --test --reg-method=delta --delta-cache=dc/enb0-pets.npy --reg-extractor=0.15480606092251736 --reg-classifier=0.17474053423403835
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=5.951676915312538 --reg-classifier=11.952030079391935
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=8.360897036453021e-05 --reg-classifier=0.2636018626012623
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=enb0 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=1.7304358752274287 --reg-classifier=1.9959420134152572
    python3 finetune.py --dataset=/raid/anon/vgg-pets/ --network=enb0 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.00034208425578714574 --reg-classifier=0.19482385270664973
done
