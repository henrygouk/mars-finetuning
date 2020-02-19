#!/bin/bash

set -o xtrace

for i in 1 2 3 4 5
do
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=resnet101 --quiet --test
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=resnet101 --quiet --test --reg-method=delta --delta-cache=dc/resnet101-butterfly.npy --reg-extractor=0.0009052115005055005 --reg-classifier=0.2362590182097744
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=5.639192186024681 --reg-classifier=9.613474460081614
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.12286664439121874 --reg-classifier=0.1186676186210077
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=resnet101 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=6.173430095799361 --reg-classifier=3.1067217250234584
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=resnet101 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.005613815517799307 --reg-classifier=0.29914544366140783

    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=enb0 --quiet --test
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=enb0 --quiet --test --reg-method=delta --delta-cache=dc/enb0-butterfly.npy --reg-extractor=0.0014523650332113945 --reg-classifier=0.3603203950519164
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=4.857979308556528 --reg-classifier=8.6679322121532
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=8.733621925976164e-05 --reg-classifier=0.0062381684731317244
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=enb0 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=4.36355581453845 --reg-classifier=1.7295210923183646
    python3 finetune.py --dataset=/raid/anon/butterfly-200 --network=enb0 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.0010198338700776709 --reg-classifier=0.3581989424760648
done

