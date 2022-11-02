#!/bin/bash

mkdir -p 01  # dir of the transferred version
mkdir -p 02  # dir of the processed version

python ../transfer.py transfer 00/train.json 01/train.json [PER-SOC]
python ../transfer.py transfer 00/dev.json 01/dev.json [PER-SOC]
python ../transfer.py transfer 00/test.json 01/test.json [PER-SOC]

python ../process.py process 01/train.json ent_rel_file.json 02/train.json bert-base-uncased 200
python ../process.py process 01/dev.json ent_rel_file.json 02/dev.json bert-base-uncased 200
python ../process.py process 01/test.json ent_rel_file.json 02/test.json bert-base-uncased 200

#rm -rf 01