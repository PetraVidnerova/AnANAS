qsub -N elcom_cpu -q qgpulong -l "select=1:ncpus=2" -l "walltime=24:00:00"  run_cpu.sh
