import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s")
from keras.preprocessing.image import ImageDataGenerator
from model import ModelBuilder
import data_helpers as dh
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
import argparse
import pickle
import json
from keras.models import load_model
import time

logger = logging.getLogger("main")

def build_arg_parser():
    parser = argparse.ArgumentParser(description='amr accuracy prediction 2.0')


    parser.add_argument('-tokenizer_path', type=str, nargs='?', default='my-tokenizer-should-be-here',
                                help='path to tokenizer')
    parser.add_argument('-file_path', type=str, nargs='?', default='my-amr-file-should-be-here',
                                help='path to amr file')
    parser.add_argument('-model_path', type=str, nargs='?', default='my-model-should-be-here',
                                help='path to model file')

    parser.add_argument('-dep_h', type=int, nargs='?', default=40,
                                help='integer dependency height')

    parser.add_argument('-dep_w', type=int, nargs='?', default=15,
                                help='integer dependency width, if height = 0 only feed sentence, no dependency tree')

    parser.add_argument('-amr_h', type=int, nargs='?', default=40,
                                help='integer amr height')

    parser.add_argument('-amr_w', type=int, nargs='?', default=15,
                                help='integer amr height')

    parser.add_argument('-target_metrics', type=str, nargs='?',default='ready_data/allmetrics-basic.json')

    parser.add_argument('-wikiopt', type=str, nargs='?',default='keep')

    parser.add_argument('-senseopt', type=str, nargs='?',default='keep')

    parser.add_argument('-reentrancyopt', type=str, nargs='?',default='rvn')

    parser.add_argument('--write_preds', dest='write_preds', action='store_true')

    parser.add_argument('--only_sent_not_dep', dest='only_sent_not_dep', action='store_true')

    return parser


def maybe_preprocess_data(args):
    
    with open(args.tokenizer_path,"rb") as f:
        tok = pickle.load(f)
    logging.info("data not preprocessed...starting preprocessing...")
    data_dict, tok = dh.load_dat_no_target(args.file_path
            , tok
            , amr_shape=(args.amr_h,args.amr_w)
            , dep_shape=(args.dep_h,args.dep_w)
            , wikiopt = args.wikiopt
            , senseopt = args.senseopt
            , use_dependency = not args.only_sent_not_dep
            )
    
    return data_dict, tok


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    print(args)

    #run_uri = "runid-{}_deph-{}_depw-{}_amrh-{}_amrw-{}.txt".format(args.runidx, args.dep_h,args.dep_w,args.amr_h,args.amr_w) 

    logger.info("checking if data is already preprocessed")
    data_dict, tok = maybe_preprocess_data(args)
    
    logger.info("data loaded")
    print(type(data_dict))
    testlen = len(data_dict["test_toks_amr"])
    
    with open( args.target_metrics,"r") as f:
        target_metrics = json.load(f)
    main_metrics = target_metrics["main"]

    
    amr_pixels = args.amr_h*args.amr_w
    dep_pixels = args.dep_h*args.dep_w

    for key in data_dict:
        if "tok" in key and "amr" in key:
            elm = np.array(data_dict[key])
            data_dict[key]=elm.reshape(elm.shape[0],amr_pixels)
        elif "tok" in key and "dep" in key:
            elm = np.array(data_dict[key])
            data_dict[key]=elm.reshape(elm.shape[0],dep_pixels)
    

    logger.info("building model...")
    model = ModelBuilder().build_amr_quality_rater(len(tok.vocab), amr_shape=(args.amr_h,args.amr_w), dep_shape=(args.dep_h,args.dep_w),n_main_output_neurons=len(target_metrics["main"]))
    model.compile(loss='mean_squared_error',
        optimizer='adam')
    model=load_model(args.model_path)

    te_in_amr = data_dict["test_toks_amr"].reshape((testlen,amr_pixels))
    te_in_dep = data_dict["test_toks_dep"].reshape((testlen,dep_pixels))
    preds = model.predict([te_in_amr,te_in_dep])

    out = []
    for j in range(len(te_in_amr)):
        string=""
        for i in range(len(main_metrics)):
            #print(i)
            string+=main_metrics[i]+":"+str(preds[j,i])+"\t"
        string =string[:-1]
        out.append(string)
    with open(args.file_path+".estimatedquality","w") as f:
        f.write("\n".join(out))
    logging.info("program ran successfully, exiting now...")

