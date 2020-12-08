import sys
import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='average results of multiple runs and for quality dimensions')
parser.add_argument('-evaldir', type=str, nargs='?', default="../keras-src/predictions/",
        help='dir with results that need be averaged')
parser.add_argument('-prf', type=str, nargs='?', default="F1", choices = ["F1","P","R"],
        help='precision recall or F1')
parser.add_argument('-idxs', type=str, nargs='+', default="1",
        help='indexes of result files')

args = parser.parse_args()

def readf(p):
    with open(p,"r") as f:
        return f.read()

def extract(string):
    dic_pr = {}
    dic_mse = {}
    for line in string.split("\n"):
        if not line:
            continue
        #0 Smatch -F1 0.01647724820395294 (0.6874436432207517, 0.0)
        line = line.replace("-","")
        line = line.replace("amed Ent", "amedEnt")
        line = line.replace("No WSD", "NoWSD")
        ls = line.split()
        task = ls[1]
        prf = ls[2]
        mse = float(ls[3])
        pear = float(ls[4].replace("(","").split(",")[0])
        if task not in dic_pr:
            dic_pr[task] = {}
            dic_mse[task] = {}
        dic_pr[task][prf] = pear
        dic_mse[task][prf] = mse
    return dic_pr,dic_mse

def format_vals(dics,opt="F1",mse=False):
    dicnew = {}
    for task in dics[0]:
        scores=[]
        for dic in dics:
            #print(dic[task])
            scores.append(dic[task][opt])
        if mse:
            scores=np.sqrt(np.array(scores))
        mean,std = np.mean(scores), np.std(scores)
        mean = str(round(mean,3))
        std=str(round(std,2))
        if len(mean) != 5:
            mean=mean+"0"
        if len(std) != 4:
            std=std+"0"
        dicnew[task] = str(mean)+ "$^{\pm "+str(std)+"}$"
    return dicnew


def formatt(dics_sys1,opt="F1",mse=False,tasks=True):

    dic1= format_vals(dics_sys1,opt=opt,mse=mse)
    strings=[]
    for task in dic1:
        if not tasks:
            strings.append(dic1[task]+"\\\\")
        else:
            strings.append(task +" & "+dic1[task]+"\\\\")

    return "\n".join(strings)


def get_pearsonr_table_main(opt="F1",idxs=[0], directory=""):
    dics=[]
    for j,i in enumerate(idxs):
        string = readf(directory+"/"+str(i)+".eval")
        dic,_ = extract(string)
        dics.append(dic)
    string = formatt(dics,opt=opt,mse=False)
    return string

def get_mse_table_main(opt="F1",idxs=[0], directory=""):
    dics=[]
    for j,i in enumerate(idxs):
        string = readf(directory+"/"+str(i)+".eval")

        _,dic = extract(string)
        dics.append(dic)
    string = formatt(dics,opt=opt,mse=True)
    return string


print("pearsonr:")
print(get_pearsonr_table_main(opt=args.prf,idxs=args.idxs, directory=args.evaldir))
print("\nRMS error:")
print(get_mse_table_main(opt=args.prf, idxs=args.idxs, directory=args.evaldir))
