#!/bin/bash

mkdir -p fold1/01
mkdir -p fold1/02  # dir of the processed version

python ../transfer.py transfer fold1/00/train.json fold1/01/train.json [PER-SOC]
python ../transfer.py transfer fold1/00/dev.json fold1/01/dev.json [PER-SOC]
python ../transfer.py transfer fold1/00/test.json fold1/01/test.json [PER-SOC]

python ../process.py process fold1/01/train.json fold1/ent_rel_file.json fold1/02/train.json bert-base-uncased 200
python ../process.py process fold1/01/dev.json fold1/ent_rel_file.json fold1/02/dev.json bert-base-uncased 200
python ../process.py process fold1/01/test.json fold1/ent_rel_file.json fold1/02/test.json bert-base-uncased 200

#rm -rf 01/fold1