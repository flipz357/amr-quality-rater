#!/bin/bash

DEPH=40
AMRH=40
DEPW=15
AMRW=15


RUNS=10

cd pytorch-src

python -u preprocess.py \
    -train_json_path ../data/amr-quality-data-v1/train.json \
    -dev_json_path ../data/amr-quality-data-v1/dev.json \
    -test_json_path ../data/amr-quality-data-v1/test.json \
    -save_prepro_data_path preprocessed_data/data_dict.json \
    -save_tokenizer_path preprocessed_data/tokenizer.pkl \
    -dep_h $DEPH \
    -dep_w $DEPW \
    -amr_h $AMRH \
    -amr_w $AMRW \
    -wikiopt keep \
    -senseopt keep \
    -reentrancyopt rvn


# IF YOU WANT PYTORCH ON GPU REPLACE main.py with main-gpu.py
for i in `eval echo {1..$RUNS}`
do
python -u main.py \
        -runidx $i \
        -prepro_data_path preprocessed_data/data_dict.json \
        -tokenizer_path preprocessed_data/tokenizer.pkl \
        -target_metrics util/smatch-metrics.json \
        -epochs 5 \
        -dep_h $DEPH \
        -dep_w $DEPW \
        -amr_h $AMRH \
        -amr_w $AMRW \
        -save_model_dir saved_models/ \
        -test_result_dir predictions/ 
done

cd ..

echo "5-way graph classification result"
python analysis/result_classif.py -idxs `eval echo {1..$RUNS}` -evaldir pytorch-src/predictions/

echo "##################################"
echo "##################################"

echo "estimated F1 correlation and RMSE"
python analysis/result.py -idxs `eval echo {1..$RUNS}` -prf F1 -evaldir pytorch-src/predictions/

echo "estimated Precision correlation and RMSE"
python analysis/result.py -idxs `eval echo {1..$RUNS}` -prf P -evaldir pytorch-src/predictions/

echo "estimated Recall correlation and RMSE"
python analysis/result.py -idxs `eval echo {1..$RUNS}` -prf R -evaldir pytorch-src/predictions/


