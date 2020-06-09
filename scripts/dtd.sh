#!/bin/bash

set -o xtrace

for i in 1 2 3 4 5
do
    python3 finetune.py --dataset=/raid/anon/dtd --network=resnet101 --quiet --test
    python3 finetune.py --dataset=/raid/anon/dtd --network=resnet101 --quiet --test --reg-method=delta --delta-cache=dc/resnet101-dtd.npy --reg-extractor=0.17261854852586161 --reg-classifier=0.1696369662557768
    python3 finetune.py --dataset=/raid/anon/dtd --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=3.959406653965832 --reg-classifier=23.508422525954273
    python3 finetune.py --dataset=/raid/anon/dtd --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.006168197617135301 --reg-classifier=0.04066234358056213
    python3 finetune.py --dataset=/raid/anon/dtd --network=resnet101 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=2.405874675106675 --reg-classifier=1.7153536730573504
    python3 finetune.py --dataset=/raid/anon/dtd --network=resnet101 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.03089663957640984 --reg-classifier=0.33454240183014805

    python3 finetune.py --dataset=/raid/anon/dtd --network=enb0 --quiet --test
    python3 finetune.py --dataset=/raid/anon/dtd --network=enb0 --quiet --test --reg-method=delta --delta-cache=dc/enb0-dtd.npy --reg-extractor=0.016873427192034163 --reg-classifier=0.2057422958342999
    python3 finetune.py --dataset=/raid/anon/dtd --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=9.939410179437791 --reg-classifier=12.862498557488857
    python3 finetune.py --dataset=/raid/anon/dtd --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.003127544533413377 --reg-classifier=0.03651331099480047
    python3 finetune.py --dataset=/raid/anon/dtd --network=enb0 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=1.907721285824912 --reg-classifier=1.6577318747516427
    python3 finetune.py --dataset=/raid/anon/dtd --network=enb0 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.0011188215779291424 --reg-classifier=0.1996312074617487
done

