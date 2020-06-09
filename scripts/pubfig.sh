#!/bin/bash

set -o xtrace

for i in 1 2 3 4 5
do
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=resnet101 --quiet --test
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=resnet101 --quiet --test --reg-method=delta --delta-cache=dc/resnet101-pubfig.npy --reg-extractor=0.006829603469076509 --reg-classifier=0.06420513234115749
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=9.369018136370515 --reg-classifier=7.998947195275408
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=resnet101 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.017902251575106182 --reg-classifier=0.05644511190541913
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=resnet101 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=12.178268046576283 --reg-classifier=4.037971869872896
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=resnet101 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.00047442412966256425 --reg-classifier=0.10531517750039265

    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=enb0 --quiet --test
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=enb0 --quiet --test --reg-method=delta --delta-cache=dc/enb0-pubfig.npy --reg-extractor=0.008952195157278565 --reg-classifier=0.21815841400749678
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=constraint --reg-extractor=10.769842053506178 --reg-classifier=15.30267014764564
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=enb0 --quiet --test --reg-norm=inf-op --reg-method=penalty --reg-extractor=0.0033300896619318492 --reg-classifier=0.2421975573380983
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=enb0 --quiet --test --reg-norm=frob --reg-method=constraint --reg-extractor=1.9297929364190836 --reg-classifier=2.01799550466595
    python3 finetune.py --dataset=/raid/anon/pubfig-83 --network=enb0 --quiet --test --reg-norm=frob --reg-method=penalty --reg-extractor=0.003948139880248755 --reg-classifier=0.03922946356768655
done

