# Data set information

To download the data sets please go to the parent directory and run `./download_data.sh`

# AmrQuality-1

This data set was created in the following steps:

1. parse sentences from LDC2015 with CAMR, JAMR and GPLA parser (see paper for references to parsers)
    * yields tuples `(sent, parse1, ..., parsen, goldparse)`
2. compute Smatch F1, Precision, Recall and Damonte 2017's subtask metrics (e.g., coreference F1) on the parser outputs
    * yields tuples like `(sent, score(parse1, goldparse), .... score(parsen, goldparse), 1.0)`
3. shuffle full data set on sentence id level (to balance the scores)
4. split in train/dev/test

#### potential experimental tasks

* given sent and a parse, predict quality scores. 
* rank parse*r*s or parses 

# AmrQuality-2

This data set was created in the following steps:

1. parse sentences from AMR3 with JAMR, GPLA and GSII (Cai and Lam 2020) parser (the latter parser is used in a recategorization variant and non-recat variant)
    * yields tuples `(sent, parse1, ..., parsen, goldparse)`
2. compute Smatch F1, Precision, Recall and missing correct/triples (e.g., coreference F1) on the parser outputs
    * yields tuples like `(sent, score(parse1, goldparse), .... score(parsen, goldparse), 1.0)`
3. shuffle full data set on sentence id level (to balance the scores)
4. split in train/dev/test
5. add negative samples to train, dev and test
    * negative samples are sentences that occur with a randomly selected goldparse, which means that a graph of high structural quality (gold graph) can occur with very low score
    * if you want to delete them just remove the corresponding keys from the json files `negGoldSample\_index`
 
#### potential experimental tasks

* given sent and a parse, predict quality scores. 
* given sent and a parse, predict false/correct triples, generate missing triples.
* rank parse*r*s or parses 

# Notes

* different parsing system use different variable names, so make sure to standardize the names or remove variables (like we do in this work)

