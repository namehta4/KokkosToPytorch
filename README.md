# CudaToPytorch
Modules needed to run this code on Cori:
1) PrgEnv-llvm/12.0.0-git_20210117
2) cudnn/8.0.5
3) cgpu
4) cuda/11.1.1
5) gcc/8.3.0
6) pytorch/1.8.0-gpu

To compile and run the code on a GPU, use:
1) mkdir build
2) cd build/
3) CXX=/global/homes/n/namehta4/kokkos/install_pm_llvm/bin/nvcc_wrapper cmake -DCMAKE_PREFIX_PATH="/global/homes/n/namehta4/kokkos/install_pm_llvm" -DKok    kos_ROOT=/global/homes/n/namehta4/kokkos/install_pm_llvm ../
4) make -j
5) srun -n 1 ./FirstNN'

