PWD=`realpath "$0"`
PWD=`dirname $PWD`

# train NPE models
#for CONFIG in $PWD/npe-config/*.yaml; do
#  snakemake --jobs 30 --configfile $CONFIG \
#    --snakefile $PWD/../../workflow/training_workflow.smk
#done

# coverage experiment and figures
snakemake --jobs 10 --configfile $PWD/experiment-config.yaml \
  --snakefile $PWD/experiment.smk -R calculate_mse

