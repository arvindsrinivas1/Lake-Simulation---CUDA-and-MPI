# heatmap for lake.cu

set terminal png

set xrange[0:1]
set yrange[0:1]

set output 'lake_i.png'
plot 'lake_i.dat' using 1:2:3 with image

set output 'lake_f_cpu_mpi.png'
plot 'lake_f_cpu_0.dat' using 1:2:3 with image

set output 'lake_f_gpu_mpi.png'
plot 'lake_f_gpu_0.dat' using 1:2:3 with image