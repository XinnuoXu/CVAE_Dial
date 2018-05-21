srun --mem 400G --gres=gpu --partition=amd-longq --nodes 1 sh train.sh
#srun --mem 20G --partition intel-longq --gres=gpu --nodelist=dgx01 sh train.sh
