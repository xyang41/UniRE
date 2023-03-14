#!/bin/bash

#mkdir -p 01-change-fields  # dir of the transferred version
#mkdir -p 02-matrix  # dir of the processed version

python ../process.py process 01-change-fields/train.json ent_rel_file.json 02-matrix/train.json /mnt/data1/public/pretrain/bert-base-uncased
python ../process.py process 01-change-fields/dev.json ent_rel_file.json 02-matrix/dev.json /mnt/data1/public/pretrain/bert-base-uncased
python ../process.py process 01-change-fields/test.json ent_rel_file.json 02-matrix/test.json /mnt/data1/public/pretrain/bert-base-uncased

#rm -rf 01-change-fields
