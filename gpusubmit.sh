qsub -N elcom_nas -q qgpulong -l "select=1:ncpus=2" -l "walltime=24:00:00"  run$1.sh
