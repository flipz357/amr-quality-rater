import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s")
from model import CNNNet
import data_helpers as dh
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
import argparse
import pickle
import json
import time
import torch

logger = logging.getLogger("main")

def build_arg_parser():
    parser = argparse.ArgumentParser(
            description='amr accuracy prediction with CNN')

    parser.add_argument('-runidx'
            , type=int
            , nargs='?'
            , default=0
            , help='integer dependency height')

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


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    print(args)

    run_uri = "runid-{}_deph-{}_depw-{}_amrh-{}_amrw-{}.txt".format(args.runidx
            , args.dep_h
            , args.dep_w
            , args.amr_h
            , args.amr_w) 

    data_dict, tok = get_preprocessed_data(args)
    
    logger.info("data loaded")

    trainlen = len(data_dict["train_toks_amr"])
    devlen = len(data_dict["dev_toks_amr"])
    testlen = len(data_dict["test_toks_amr"])

    tr_y, de_y, te_y, target_metrics, main_metrics = prepare_target(args,data_dict)
    tr_y = np.array(tr_y)
    de_y = np.array(de_y)
    te_y = np.array(te_y)
    
    amr_pixels = args.amr_h * args.amr_w
    dep_pixels = args.dep_h * args.dep_w

    for key in data_dict:
        if "tok" in key and "amr" in key:
            elm = np.array(data_dict[key])
            data_dict[key]=elm.reshape(elm.shape[0], amr_pixels)
        elif "tok" in key and "dep" in key:
            elm = np.array(data_dict[key])
            data_dict[key]=elm.reshape(elm.shape[0], dep_pixels)
    
    #shape data to prepare for embedding lookup
    de_in_amr = data_dict["dev_toks_amr"].reshape((devlen, amr_pixels))
    de_in_dep = data_dict["dev_toks_dep"].reshape((devlen, dep_pixels))
    tr_in_amr = data_dict["train_toks_amr"].reshape((trainlen, amr_pixels))
    tr_in_dep = data_dict["train_toks_dep"].reshape((trainlen, dep_pixels))
    te_in_amr = data_dict["test_toks_amr"].reshape((testlen, amr_pixels))
    te_in_dep = data_dict["test_toks_dep"].reshape((testlen, dep_pixels))

    de_in_amr = torch.from_numpy(de_in_amr)
    de_in_dep = torch.from_numpy(de_in_dep)
    tr_in_amr = torch.from_numpy(tr_in_amr)
    tr_in_dep = torch.from_numpy(tr_in_dep)
    te_in_amr = torch.from_numpy(te_in_amr)
    te_in_dep = torch.from_numpy(te_in_dep)
   
    tr_y = torch.from_numpy(tr_y).float()
    de_y = torch.from_numpy(de_y).float()
    te_y = torch.from_numpy(te_y).float()

    net = CNNNet(vocab_count=len(tok.vocab))

    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(tr_in_amr, tr_in_dep, tr_y)
    
    loader = DataLoader(ds,batch_size=64, shuffle=True)
    
    
    ds_dev = TensorDataset(de_in_amr, de_in_dep, de_y)
    devloader = DataLoader(ds_dev,batch_size=64, shuffle=False)
    
    ds_test = TensorDataset(te_in_amr, te_in_dep, te_y)
    testloader = DataLoader(ds_test,batch_size=64, shuffle=False)

    from torch import optim, nn

    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)

    bestscore = 0.0
    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            amr, dep, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(amr,dep)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches, 300 * 64  examples
                
                preds = []
                for datadev in devloader:
                    damr, ddep, dlabels = datadev
                    pred = net(damr, ddep)
                    pred = pred.data.numpy()
                    preds.append(pred)
                preds = np.concatenate(preds, axis=0)
                
                psr = pearsonr(preds.flatten(), de_y.data.numpy().flatten())[0]
                logging.info("epoch {}; examples {}; loss {}; Pearsonr {}".format(
                                epoch + 1, (i + 1)*64, running_loss / 200, psr))
                if psr > bestscore:
                    logging.info("new validation high: +{}\
                            ... saving model...".format(psr - bestscore))
                    bestscore=psr
                    torch.save(net.state_dict(), args.save_model_dir+"/{}.pt".format(run_uri))
                running_loss = 0.0
    print('Finished Training') 
    
    
    preds = []
    
    net = CNNNet(vocab_count=len(tok.vocab))
    net.load_state_dict(torch.load(args.save_model_dir+"/{}.pt".format(run_uri)))
    net.eval()
     
    
    for datatest in testloader:
        tamr, tdep, _ = datatest
        pred = net(tamr,tdep)
        pred = pred.data.numpy()
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    te_y = te_y.data.numpy()
    logging.info("writing predictions and evaluation to {}".format(args.test_result_dir))
    if args.test_result_dir:
        txt = "testpreds " + str(preds.tolist())
        txt += "testtargets " + str(te_y.tolist())
        dh.writef(txt, args.test_result_dir+"/"+str(args.runidx)+".testpreds")
        
        txt = ""
        for i in range(len(main_metrics)):
            txt += "{} {} {} {}".format(i
                    , main_metrics[i]
                    , mse(te_y[:,i], preds[:,i])
                    , pearsonr(te_y[:,i],preds[:,i])) 
            txt += "\n"
        dh.writef(txt, args.test_result_dir+"/"+str(args.runidx)+".eval")

    logging.info("program ran successfully, exiting now...")
