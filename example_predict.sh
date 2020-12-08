#!/bin/bash

DEPH=40
AMRH=40
DEPW=15
AMRW=15

AMRFILEPATH=data/toy/toy.txt

cd keras-src

python -u main_only_infer.py \
    -tokenizer_path preprocessed_data/tokenizer.pkl \
    -model_path saved_models/runid-1_deph-$DEPH\_depw-$DEPW\_amrh-$AMRH\_amrw-$AMRW.txt.h5\
    -file_path ../$AMRFILEPATH \
    -target_metrics util/smatch-metrics.json \
	-dep_h $DEPH \
    -dep_w $DEPW \
    -amr_h $AMRH \
    -amr_w $AMRW \
    -wikiopt keep \
    -senseopt keep \
    -reentrancyopt rvn
cd ..


