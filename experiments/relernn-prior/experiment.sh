PWD=`realpath "$0"`
PWD=`dirname $PWD`

# train NPE models
#for CONFIG in $PWD/npe-config/*.yaml; do
for CONFIG in $PWD/npe-config/*_100000.yaml; do
  snakemake --configfile $CONFIG \
    --jobs 30 \
    --snakefile $PWD/../../workflow/training_workflow.smk
done

## get coverage on a "true" model
#python3 predict_on_true.py

## make plot
#python3 plot_coverage_comparison.py

