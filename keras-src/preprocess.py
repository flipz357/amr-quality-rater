import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s")
import data_helpers as dh
import argparse
import os
import pickle
import json

logger = logging.getLogger("preprocess")

def build_arg_parser():
    parser = argparse.ArgumentParser(
            description='amr accuracy prediction with CNN')

    parser.add_argument('-train_json_path'
            , type=str
            , nargs='?'
            , default='../data/amr-quality/AMR_ACCURACIES_v1.2/train.json')
    
    parser.add_argument('-dev_json_path'
            , type=str
            , nargs='?'
            , default='../data/amr-quality/AMR_ACCURACIES_v1.2/dev.json')

    parser.add_argument('-test_json_path'
            , type=str
            , nargs='?'
            , default='../data/amr-quality/AMR_ACCURACIES_v1.2/test.json')
    
    parser.add_argument('-save_prepro_data_path'
            , type=str
            , nargs='?'
            , default="preprocessed_data/data.json"
            , help='save preprocessed data under this path')
    
    parser.add_argument('-save_tokenizer_path'
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
            , help='integer dependency width, if height = 0\
                    only feed sentence, no dependency tree')

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


    parser.add_argument('-wikiopt'
            , type=str
            , nargs='?'
            , default='keep')

    parser.add_argument('-senseopt'
            , type=str
            , nargs='?'
            , default='keep')

    parser.add_argument('-reentrancyopt'
            , type=str
            , nargs='?'
            , default='rvn')

    parser.add_argument('--only_sent_not_dep'
            , dest='only_sent_not_dep'
            , action='store_true')

    return parser


def maybe_preprocess_data(args):
    
    data_dict, tok = None, None
    if os.path.isfile(args.save_prepro_data_path):
        logging.info("found data that is already preprocessed... \
                skipping preprocessing")
    else:
        logging.info("data not preprocessed...starting preprocessing...")
        data_dict, tok = dh.load_dat(
                  path_train = args.train_json_path
                , path_dev = args.dev_json_path
                , path_test = args.test_json_path
                , amr_shape=(args.amr_h, args.amr_w)
                , dep_shape=(args.dep_h, args.dep_w)
                , wikiopt = args.wikiopt
                , senseopt = args.senseopt
                , use_dependency = not args.only_sent_not_dep
                )

        with open(args.save_tokenizer_path, "wb") as f:
            pickle.dump(tok, f)
        with open(args.save_prepro_data_path, "w") as f:
            f.write(json.dumps(data_dict)) 
    return data_dict, tok

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)
    logger.info("checking if data is already preprocessed")
    _,_ = maybe_preprocess_data(args)

