import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s")
from keras.preprocessing.image import ImageDataGenerator
from model import ModelBuilder
import data_helpers as dh
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
import argparse
import os
import pickle
import json
from keras.models import load_model
from keras.optimizers import Adam
import time

logger = logging.getLogger("main")

def build_arg_parser():
    
    parser = argparse.ArgumentParser(
            description='amr accuracy prediction with CNN')

    parser.add_argument('-runidx'
            , type=int
            , nargs='?'
            , default=0, help='integer dependency height')

    parser.add_argument('-prepro_data_path'
            , type=str
            , nargs='?'
            , default="preprocessed_data/data.json"
            , help='save preprocessed data under this path')

    parser.add_argument('-tokenizer_path'
            , type=str
            , nargs='?'
            , default="preprocessed_data/tokenizer.pkl"
            , help='save tokenizer under this path')
     
    parser.add_argument('-dep_h'
            , type=int
            , nargs='?'
            , default=40
            , help='integer dependency height')

    parser.add_argument('-dep_w'
            , type=int
            , nargs='?'
            , default=15
            , help='integer dependency width')

    parser.add_argument('-amr_h'
            , type=int
            , nargs='?'
            , default=40
            , help='integer amr height')

    parser.add_argument('-amr_w'
            , type=int
            , nargs='?'
            , default=15
            , help='integer amr height')

    parser.add_argument('-epochs'
            , type=int
            , nargs='?'
            , default=10
            , help='max num epochs')

    parser.add_argument('-target_metrics'
            , type=str
            , nargs='?'
            , default='util/allmetrics.json')

    parser.add_argument('-save_model_dir'
            , type=str
            , nargs='?'
            , default='saved_models/')
     
    parser.add_argument('-test_result_dir'
            , type=str
            , nargs='?'
            , default='predictions/')
    
    return parser


def get_preprocessed_data(args):

    with open(args.tokenizer_path, "rb") as f:
        tok = pickle.load(f)
    
    with open(args.prepro_data_path, "r") as f:
        data_dict=json.load(f)
    
    return data_dict, tok

def prepare_target(args, data_dict):
    
    with open( args.target_metrics,"r") as f:
        target_metrics = json.load(f)
    
    main_metrics = target_metrics["main"]
    
    #prep train targets
    tr_y = []
    for i in range(trainlen):
        y=[]
        for m in main_metrics:
            y.append(data_dict["train_"+m][i])
        tr_y.append(y)
    
    #prep dev targets
    de_y = []
    for i in range(devlen):
        y=[]
        for m in main_metrics:
            y.append(data_dict["dev_"+m][i])
        de_y.append(y)
    
    #prep test targets
    te_y = []
    for i in range(testlen):
        y=[]
        for m in main_metrics:
            y.append(data_dict["test_"+m][i])
        te_y.append(y)

    tr_y = np.array(tr_y)
    de_y = np.array(de_y)
    te_y = np.array(te_y)

    return tr_y, de_y, te_y, target_metrics, main_metrics


def get_data_generator(option=0):
    
    #this is default (no manipulation), all other gens are very experimental
    if option == 0:
        datagen = ImageDataGenerator()

    elif option == 1:
        #rotate
        datagen = ImageDataGenerator(
            rotation_range=25
                )
    elif option == 2:
        #flipping
        datagen = ImageDataGenerator(
            vertical_flip=True,
            horizontal_flip=True
            )
        
    elif option == 3:
        #image stuff
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    return datagen

if __name__ == "__main__":
    
    parser = build_arg_parser()
    args = parser.parse_args()
    
    print(args)

    run_uri = "runid-{}_deph-{}_depw-{}_amrh-{}_amrw-{}.txt".format(args.runidx, args.dep_h,args.dep_w,args.amr_h,args.amr_w) 

    data_dict, tok = get_preprocessed_data(args)
    
    logger.info("data loaded")

    trainlen = len(data_dict["train_toks_amr"])
    devlen = len(data_dict["dev_toks_amr"])
    testlen = len(data_dict["test_toks_amr"])

    tr_y, de_y, te_y, target_metrics, main_metrics = prepare_target(args,data_dict)
    tr_y = np.array(tr_y)
    de_y = np.array(de_y)
    te_y = np.array(te_y)
    
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
    
    model = ModelBuilder().build_amr_quality_rater(
            len(tok.vocab)
            , amr_shape=(args.amr_h,args.amr_w)
            , dep_shape=(args.dep_h,args.dep_w)
            , n_main_output_neurons=len(target_metrics["main"]))
    
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07) 
    model.compile(loss='mean_squared_error',
        optimizer=optim)


    def generator_n_item(datalist, y, gen, batch_size=64):
        gens = [gen.flow(data, y, batch_size=batch_size, seed=1) for data in datalist]
        while True:
            collector=[]
            for i in range(len(datalist)):
                collector.append(gens[i].next())
            yield [elm[0] for elm in collector], collector[0][1]
    
    logger.info("building data generator")
    datagen = get_data_generator(option=0)

    best=0.0

    #shape data (#examples, amrHeight, amrWidth) --> (#examples, amrHeight*amrWidth)
    tr_in_amr = data_dict["train_toks_amr"].reshape((trainlen,amr_pixels))
    tr_in_dep = data_dict["train_toks_dep"].reshape((trainlen,dep_pixels))
    
    de_in_amr = data_dict["dev_toks_amr"].reshape((devlen,amr_pixels))
    de_in_dep = data_dict["dev_toks_dep"].reshape((devlen,dep_pixels))
    
    te_in_amr = data_dict["test_toks_amr"].reshape((testlen,amr_pixels))
    te_in_dep = data_dict["test_toks_dep"].reshape((testlen,dep_pixels))
    
    #needed for image data generator expand last dim
    tr_in_amr_4d = np.expand_dims(
            tr_in_amr.reshape((trainlen,args.amr_h, args.amr_w)), -1)
    tr_in_dep_4d = np.expand_dims(
            tr_in_dep.reshape((trainlen,args.dep_h, args.dep_w)), -1)


    for e in range(args.epochs):
        logging.info("starting epoch {}".format(e))
        
        batches = 0 
         
        start = time.time() 

        running_loss = 0.0
        # loop over minibatches and fit model
        for x_batch, y_batch in generator_n_item([tr_in_amr_4d
            ,tr_in_dep_4d]
            ,tr_y,datagen,batch_size=64):
            
            tr_in_amr = x_batch[0].reshape((x_batch[0].shape[0], amr_pixels))
            tr_in_dep = x_batch[1].reshape((x_batch[0].shape[0], dep_pixels))
            
            loss = model.train_on_batch([tr_in_amr, tr_in_dep], y_batch)

            running_loss += loss
            
            if batches >= trainlen / 64:
                break
            
            if batches % 200 == 199:
                preds = model.predict([de_in_amr, de_in_dep])
                psr = pearsonr(preds.flatten(), de_y.flatten())[0]
                logging.info("epoch {}; examples {}; loss {}; Pearsonr {}".format(
                    e + 1, (batches + 1)*64, running_loss / 200, psr))
                if psr > best:
                    logging.info("new validation high: +{}\
                            ... saving model...".format(psr - best))
                    best = psr
                    model.save(args.save_model_dir+"/{}.h5".format(run_uri))
                running_loss = 0.0
            batches += 1
        end = time.time()
        #logger.info("TIME epoch:".format(end - start))
        
        
    model=load_model(args.save_model_dir+"/{}.h5".format(run_uri))
    
    logging.info("training finished.... predicting test with best on dev")
    preds = model.predict([te_in_amr,te_in_dep])
    
    logging.info("writing predictions and evaluation to {}".format(
        args.test_result_dir))

    if args.test_result_dir:
        txt = "testpreds " + str(preds.tolist())
        txt += "testtargets " + str(te_y.tolist())
        dh.writef(txt, args.test_result_dir+"/"+str(args.runidx)+".testpreds")
        
        txt = ""
        for i in range(len(main_metrics)):
            txt += "{} {} {} {}".format(i
                    , main_metrics[i]
                    , mse(te_y[:,i] , preds[:,i])
                    ,pearsonr(te_y[:,i],preds[:,i])) + "\n"
        
        dh.writef(txt, args.test_result_dir+"/"+str(args.runidx)+".eval")

    logging.info("program ran successfully, exiting now...")
