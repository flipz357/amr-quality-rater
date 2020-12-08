import json
import logging
import spacy
import my_tokenizers
import dependency_tree_handling as dep_helpers
import amr_graph_handling as amr_helpers

logger = logging.getLogger("dh_helper")

def writef(string, path):
    with open(path, "w") as f:
        f.write(string)


def readj(path):
    with open(path,"r") as f:
        dat = json.load(f)
    return dat


def load_dat(path_train
        , path_dev
        , path_test
        , share_vocab=True
        , amr_shape=(30,30)
        , dep_shape=(30,30)
        , wikiopt="keep"
        , senseopt="keep"
        , reentrancyopt="rvn"
        , use_dependency=True):
    
    #load data 
    tr=readj(path_train)
    de=readj(path_dev)
    te=readj(path_test)
    
    tr_keys = list(sorted(list(tr.keys())))
    de_keys = list(sorted(list(de.keys())))
    te_keys = list(sorted(list(te.keys())))

    #loading spacy
    nlp=spacy.load("en_core_web_sm")
    dep_helpers.modify_nlp(nlp)

    #project AMR to 2d image
    amr_helpers.add_2d_strings(tr, wikiopt=wikiopt
            , senseopt=senseopt, reentrancyopt=reentrancyopt)
    amr_helpers.add_2d_strings(de, wikiopt=wikiopt
            , senseopt=senseopt, reentrancyopt=reentrancyopt)
    amr_helpers.add_2d_strings(te, wikiopt=wikiopt
            , senseopt=senseopt, reentrancyopt=reentrancyopt)
    logger.info("First training example AMR: {}".format(
        [tr[key]["amr2d"] for key in tr_keys][0]))
    
    #project DEP to 2d image
    dep_helpers.add_2d_strings_dependency(tr, nlp, use_dependency=use_dependency)
    dep_helpers.add_2d_strings_dependency(de, nlp, use_dependency=use_dependency)
    dep_helpers.add_2d_strings_dependency(te, nlp, use_dependency=use_dependency) 
    logger.info("First training example DEP: {}".format(
        [tr[key]["dep2d"] for key in tr_keys][0]))
    

    #tokenize and map to indeces 
    tok = my_tokenizers.Tokenizer2D()
    tok.fit([tr[key]["amr2d"]+tr[key]["dep2d"] for key in tr_keys])
    tr_tok_amr = tok.apply([tr[key]["amr2d"] for key in tr_keys], out_dim=amr_shape)
    de_tok_amr = tok.apply([de[key]["amr2d"] for key in de_keys], out_dim=amr_shape)
    te_tok_amr = tok.apply([te[key]["amr2d"] for key in te_keys], out_dim=amr_shape)
    
    tr_tok_dep = tok.apply([tr[key]["dep2d"] for key in tr_keys], out_dim=dep_shape)
    de_tok_dep = tok.apply([de[key]["dep2d"] for key in de_keys], out_dim=dep_shape)
    te_tok_dep = tok.apply([te[key]["dep2d"] for key in te_keys], out_dim=dep_shape)
    
    #create data dict
    data_dict = {
              "train_toks_amr": tr_tok_amr
            , "dev_toks_amr": de_tok_amr
            , "test_toks_amr": te_tok_amr
            , "train_toks_dep": tr_tok_dep
            , "dev_toks_dep": de_tok_dep
            , "test_toks_dep": te_tok_dep
            }

    # and add target scores
    tr_eval = get_eval(tr, tr_keys)
    de_eval = get_eval(de, de_keys)
    te_eval = get_eval(te, te_keys)
    logger.info("First training example TARGET Smatch F1: {}".format(
        tr_eval["Smatch -F1"][0]))
    
    data_dict.update({"train_" + k: tr_eval[k] for k in tr_eval})
    data_dict.update({"dev_" + k: de_eval[k] for k in de_eval})
    data_dict.update({"test_" + k: te_eval[k] for k in te_eval})
    
    return data_dict, tok


def load_dat_no_target(filepath
        , tokenizer
        , share_vocab=True
        , amr_shape=(30,30)
        , dep_shape=(30,30)
        , wikiopt="keep"
        , senseopt="keep"
        , reentrancyopt="rvn"
        , use_dependency=True): 

    """ this fucntion is only needed for application 
    of the system to raw AMR parsed text 
    """


    te = {}
    with open(filepath, encoding="utf-8") as f:
        amrs = [l for l in f.read().split("\n\n") if l]
    for i, amr in enumerate(amrs):
        te["x_"+str(i)] = {
                "amr":"\n".join([l for l in amr.split("\n") 
                    if l and not l.startswith("#")])
                , "snt":amr.split("::snt ")[1].split("\n")[0]}
    nlp=spacy.load("en_core_web_sm")
    amr_helpers.add_2d_strings(te, wikiopt=wikiopt
            , senseopt=senseopt, reentrancyopt="rvn")
    dep_helpers.add_2d_strings_dependency(te, nlp, use_dependency=use_dependency)
    te_keys = list(sorted(list(te.keys()), key=lambda x:int(x.split("_")[1])))
    if share_vocab:
        tok = tokenizer
        te_tok_amr = tok.apply([te[key]["amr2d"] for key in te_keys], out_dim=amr_shape)
        te_tok_dep = tok.apply([te[key]["dep2d"] for key in te_keys], out_dim=dep_shape)
    data_dict = {
              "test_toks_amr": te_tok_amr
            , "test_toks_dep": te_tok_dep
            }
    return data_dict, tok


def get_eval(dat, keys):

    """ extract eval from data dict

    Args:
        dat (dict): our input data dict
        keys: keys of the data examples for which we collect the target scores
    Returns:
        dict: data dict with evaluation scores for examples
    """


    dic={}
    for i, key in enumerate(keys):
        ev = dat[key]["eval"]
        for m in ev:
            if i == 0:
                dic[m+"-P"] = []
                dic[m+"-R"] = []
                dic[m+"-F1"] = []
            dic[m+"-P"].append(ev[m][0])
            dic[m+"-R"].append(ev[m][1])
            dic[m+"-F1"].append(ev[m][2])
    return dic
                
