import sys
import numpy as np
from collections import defaultdict
import ast
import argparse
import json
from sklearn import metrics as skmetrics


parser = argparse.ArgumentParser(description='average results of multiple runs and for quality dimensions')
parser.add_argument('-evaldir', type=str, nargs='?', default="../keras-src/predictions/",
        help='dir with results that need be averaged')
parser.add_argument('-metrics', type=str, nargs='?', default="keras-src/util/smatch-metrics.json",
        help='metrics, see default')
parser.add_argument('-idxs', type=str, nargs='+', default="1",
        help='indexes of result files')

args = parser.parse_args()

def readf(p):
    with open(p,"r") as f:
        return f.read()

def map2classes(ls):
    new = []
    for subl in ls:
        o = []
        for num in subl:
            if num < 0.25:
                o.append(0)
            elif num < 0.5:
                o.append(1)
            elif num < 0.75:
                o.append(2)
            elif num < 0.95:
                o.append(3)
            else:
                o.append(4)
        new.append(o)
    return new

def extract(string):
    y = string.split("testpreds ")[1].split("testtargets")[0]
    y = ast.literal_eval(y)
    y = map2classes(y)
    y=np.array(y)
    
    p = string.split("testtargets ")[1]
    p = ast.literal_eval(p)
    p = map2classes(p)
    p = np.array(p)

    return y, p


def evaluate(metrics="../keras-src/util/smatch-metrics.json"
        , idxs=[0]
        , directory=""):
    ls = []
    for j,i in enumerate(idxs):
        string = readf(directory+"/"+str(i)+".testpreds")
        y,p = extract(string)
        ls.append( (y,p) )
    with open(metrics,"r") as f:
        metrics = json.load(f)["main"]
    for i,metric in enumerate(metrics):
        print("### classification according to metric ", metric," ###")
        macro_f1s = []
        kappa2s = []
        accuracys = []
        for (y,p) in ls:
            macro_f1 = skmetrics.f1_score(y[:,i], p[:,i], average="macro")
            accuracy = skmetrics.accuracy_score(y[:,i], p[:,i])
            kappa2 = skmetrics.cohen_kappa_score(y[:,i], p[:,i], weights="quadratic")
            macro_f1s.append(macro_f1)
            accuracys.append(accuracy)
            kappa2s.append(kappa2)
        print("----------------------------")
        print("Main result")
        print("macro f1", np.mean(macro_f1s), np.std(macro_f1s))
        print("accuracy", np.mean(accuracys), np.std(accuracys))
        print("kappa2", np.mean(kappa2s), np.std(kappa2s))
        print("----------------------------\n\n")
        
        from sklearn.dummy import DummyClassifier
        clf = DummyClassifier(strategy="stratified")
        rand = clf.fit(np.random.rand(len(y[:,i]),5),y[:,i]).predict(np.random.rand(len(y[:,i]),5))
        
        macro_f1s = [skmetrics.f1_score(y[:,i], rand, average="macro")]
        accuracys = [skmetrics.accuracy_score(y[:,i], rand)]
        kappa2s = [skmetrics.cohen_kappa_score(y[:,i], rand, weights="quadratic")]
        print("----------------------------")
        print("(stratified)  Random baseline result")
        print("macro f1", np.mean(macro_f1s), np.std(macro_f1s))
        print("accuracy", np.mean(accuracys), np.std(accuracys))
        print("kappa2", np.mean(kappa2s), np.std(kappa2s))
        print("----------------------------\n\n")

        
        clf = DummyClassifier(strategy="most_frequent")
        mfs = clf.fit(np.random.rand(len(y[:,i]),5),y[:,i]).predict(np.random.rand(len(y[:,i]),5))
        macro_f1s = [skmetrics.f1_score(y[:,i], mfs, average="macro")]
        accuracys = [skmetrics.accuracy_score(y[:,i], mfs)]
        kappa2s = [skmetrics.cohen_kappa_score(y[:,i], mfs, weights="quadratic")]
        
        print("----------------------------")
        print("majority baseline result")
        print("macro f1", np.mean(macro_f1s), np.std(macro_f1s))
        print("accuracy", np.mean(accuracys), np.std(accuracys))
        print("kappa2", np.mean(kappa2s), np.std(kappa2s))
        print("----------------------------\n\n\n\n")

        



evaluate(metrics=args.metrics,idxs=args.idxs, directory=args.evaldir)
