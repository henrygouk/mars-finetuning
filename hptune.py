from argparse import ArgumentParser
from hyperopt import fmin, hp, tpe, Trials
import subprocess as sp

parser = ArgumentParser()
parser.add_argument("--network", action="store")
parser.add_argument("--dataset", action="store")
parser.add_argument("--reg-norm", dest="reg_norm", action="store", default="inf-op")
parser.add_argument("--reg-method", dest="reg_method", action="store", default="constraint")
parser.add_argument("--delta-cache", dest="delta_cache", action="store", default="")
parser.add_argument("--max-evals", dest="max_evals", action="store", default="20")
args = parser.parse_args()

def objective(params):
    cmd = ["python3", "finetune.py", "--network=" + args.network, "--dataset=" + args.dataset, "--quiet"]
    cmd.append("--reg-method=" + args.reg_method)
    cmd.append("--reg-norm=" + args.reg_norm)
    cmd.append("--reg-classifier=" + str(params["classifier"]))
    cmd.append("--reg-extractor=" + str(params["extractor"]))
    cmd.append("--delta-cache=" + args.delta_cache)
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, universal_newlines=True)
    loss = 0.0

    for line in proc.stdout:
        fields = line.split(',')
        loss = -float(fields[8])

    print(str(loss) + ": " + str(params))
    return loss

space = {}

if args.reg_method == "constraint":
    space["classifier"] = hp.loguniform("classifier", 0.5, 3.5)
    space["extractor"] = hp.loguniform("extractor", 0.5, 3.5)
else:
    space["classifier"] = hp.loguniform("classifier", -10.0, -1.0)
    space["extractor"] = hp.loguniform("extractor", -10.0, -1.0)

trials = Trials()
best_params = fmin(objective, space=space, algo=tpe.suggest, max_evals=int(args.max_evals), trials=trials)

print(best_params)
