#!/usr/bin/bash
# sh examples/run_MVCL_DAF_MIntRec.sh
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for method in 'mvcl_daf' 
    do
        for text_backbone in 'bert-large-uncased'
        do
            python run.py \
            --dataset 'MIntRec' \
            --logger_name ${method} \
            --method ${method} \
            --tune \
            --train \
            --save_results \
            --seed $seed \
            --gpu_id '0' \
            --text_backbone $text_backbone \
            --config_file_name MVCL_DAF_MIntRec \
            --results_file_name "MVCL_DAF_MIntRec_maxdeep5.csv" 
        done
    done
done


