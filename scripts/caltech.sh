#!/bin/bash

set -o xtrace

for i in 1 2 3 4 5
do
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=resnet101 --quiet --test
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=resnet101 --quiet --test --reg-method=delta --delta-cache=dc/resnet101-caltech.npy --reg-extractor=0.02806252814025854 --reg-classifier=0.06722156323081158
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=18.067335785022625 --reg-classifier=31.01600540213599
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.00018095892661754381 --reg-classifier=0.057724667510961565
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=resnet101 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=4.299619976472963 --reg-classifier=1.781445684054469
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=resnet101 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.012832519899543154 --reg-classifier=0.09987031012039471

    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=enb0 --quiet --test
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=enb0 --quiet --test --reg-method=delta --delta-cache=dc/enb0-caltech.npy --reg-extractor=0.0009089464426302129 --reg-classifier=0.17315132721127716
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=3.480587069042308 --reg-classifier=27.790473530507324
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.00013232687033245253 --reg-classifier=0.020054744259953957
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=enb0 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=3.1790857927349845 --reg-classifier=1.9635483235436386
    python3 finetune.py --dataset=/raid/anon/caltech-256 --network=enb0 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.0010034127490568371 --reg-classifier=0.04313921988538812
done

