#!/bin/bash

set -o xtrace

for i in 1 2 3 4 5
do
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=resnet101 --quiet --test
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=resnet101 --quiet --test --reg-method=delta --delta-cache=dc/resnet101-flowers.npy --reg-classifier=0.26426670241812295 --reg-extractor=0.05908174367339489
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=6.581151746007226 --reg-classifier=7.614292570518677
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.00021916093034877063 --reg-classifier=0.0011648743930556243
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=resnet101 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=5.271757318304804 --reg-classifier=3.0129629926206167
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=resnet101 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.03994148696774106 --reg-classifier=0.09534244732508584

    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=enb0 --quiet --test
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=enb0 --quiet --test --reg-method=delta --delta-cache=dc/enb0-flowers.npy --reg-classifier=0.04151159356207949 --reg-extractor=0.00047024619624219747
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=10.250269668487821 --reg-classifier=8.167263052525952
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.00037650422211560217 --reg-classifier=0.0015179709568761022
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=enb0 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=8.78574321567812 --reg-classifier=1.8054416382394405
    python3 finetune.py --dataset=/raid/anon/vgg-flowers --network=enb0 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.0010454505046214517 --reg-classifier=0.3077015638067559
done

