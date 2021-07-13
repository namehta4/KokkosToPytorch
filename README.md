# KokkosToPytorch
Modules needed to run this code on Cori:
1) PrgEnv-llvm/12.0.0-git_20210117
2) pytorch/1.8.0-gpu
3) cgpu
4) cuda/11.1.1
5) gcc/8.3.0
6) cudnn/8.0.5
7) cmake/3.18.4

* Ensure that kokkos is present in the home directory. Build install kokkos cuda in kokkos/install_cuda_reg. Ensure gcc/7.3.0 only when building Kokkos to avoid dependency issues
* Change lines 9 and 10 in CMakeLists.txt to reflect the location of python installation and python flag 

To compile and run the code on a GPU, use:
1) mkdir build
2) cd build/
3) CXX=/global/homes/n/namehta4/kokkos/install_cuda_reg/bin/nvcc_wrapper cmake -DCMAKE_PREFIX_PATH="/global/homes/n/namehta4/kokkos/install_cuda_reg" ../
4) make -j4
5) srun -n 1 ./FirstNN

******************************************************
Modules needed to run this code on Tulip(cray)
1) rocm/4.2.0
2) cmake/3.18.2
3) kokkos-hip backend built with cxx standard 14
4) conda environment containing pytorch (miniconda)

To compile and run the code on a GPU, use;
1) mkdir build
2) cd build
3) CXX=hipcc cmake -DCMAKE_PREFIX_PATH="/home/users/coe0221/miniconda3/envs/ExaRL/lib/python3.9/site-packages/torch;/home/users/coe0221/kokkos/install_hip" ../ 
4) make -j4
5) srun ./FirstNN
