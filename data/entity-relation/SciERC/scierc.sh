#!/bin/bash

mkdir -p 01-change-fields # dir of the transferred version
mkdir -p 02-matrix # dir of the processed version

python ../transfer.py transfer 00-raw/train.json 01-change-fields/train.json [COMPARE,CONJUNCTION]
python ../transfer.py transfer 00-raw/dev.json 01-change-fields/dev.json [COMPARE,CONJUNCTION]
python ../transfer.py transfer 00-raw/test.json 01-change-fields/test.json [COMPARE,CONJUNCTION]

python ../process.py process 01-change-fields/train.json ent_rel_file.json 02-matrix/train.json allenai/scibert_scivocab_uncased 200 True
python ../process.py process 01-change-fields/dev.json ent_rel_file.json 02-matrix/dev.json allenai/scibert_scivocab_uncased 200 True
python ../process.py process 01-change-fields/test.json ent_rel_file.json 02-matrix/test.json allenai/scibert_scivocab_uncased 200 True