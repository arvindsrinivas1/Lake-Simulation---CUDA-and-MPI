#!/bin/bash
cat lake_f_cpu_1.dat >> lake_f_cpu_0.dat
cat lake_f_cpu_2.dat >> lake_f_cpu_0.dat
cat lake_f_cpu_3.dat >> lake_f_cpu_0.dat

cat lake_f_gpu_1.dat >> lake_f_gpu_0.dat
cat lake_f_gpu_2.dat >> lake_f_gpu_0.dat
cat lake_f_gpu_3.dat >> lake_f_gpu_0.dat

gnuplot heatmap.gnu
rm lake_f*.dat