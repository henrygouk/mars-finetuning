#!/bin/bash

set -o xtrace

for i in 1 2 3 4 5
do
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=resnet101 --quiet --test
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=resnet101 --quiet --test --reg-method=delta --delta-cache=dc/resnet101-aircraft.npy --reg-extractor=0.03187606634855057 --reg-classifier=0.14765567167862487
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=10.360496492780777 --reg-classifier=3.259388007391971
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.003853539013486976 --reg-classifier=0.07476345145522699
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=resnet101 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=2.236423682164394 --reg-classifier=1.8631994367794358
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=resnet101 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.014600076196758066 --reg-classifier=0.23391497598093233

    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=enb0 --quiet --test
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=enb0 --quiet --test --reg-method=delta --delta-cache=dc/enb0-aircraft.npy --reg-extractor=5.9997718356372584e-05 --reg-classifier=0.2769387669648185
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=4.711232636099617 --reg-classifier=11.693961839062071
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.00021863955200940454 --reg-classifier=0.2701099148142051
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=enb0 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=5.522346634044927 --reg-classifier=1.7009094549652324
    python3 finetune.py --dataset=/raid/anon/vgg-aircraft --network=enb0 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=9.594723930304776e-05 --reg-classifier=0.1456088766309684
done

