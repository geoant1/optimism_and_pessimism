#!bin/bash

for i in {0..39}
do
    cat fit_params_all.sh > "fit_params_all_${i}.sh"
    orig='srun python ./main6.py*'
    repl="srun python ./main6.py ${i}"
    sed -i -e "s~$orig~$repl~" "fit_params_all_${i}.sh"

    orig='#SBATCH -J fit_all*'
    repl="#SBATCH -J fit_${i}"
    sed -i -e "s~$orig~$repl~" "fit_params_all_${i}.sh"
    if [ ! -d "/u/gantonov/out/save_params_${i}" ]; then
        mkdir "/u/gantonov/out/save_params_${i}"
    fi
    sbatch "fit_params_all_${i}.sh"
done
