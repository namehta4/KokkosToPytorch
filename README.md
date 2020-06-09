# KokkosToPytorch
Modules needed to run this code:
1) PrgEnv-llvm
2) python/3.7-anaconda
3) esslurm
4) cuda/10.2.89
5) gcc/8.3.0
6) pytorch/v1.5.0-gpu

Ensure that kokkos is present in the home directory. Build install kokkos cuda in kokkos/install_cuda
Ensure libtorch files are present in the home directory. Can be obtained using 'wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.0.zip'

To compile the code, use:
CXX=/global/homes/n/namehta4/kokkos/install_cuda/bin/nvcc_wrapper cmake -DCMAKE_PREFIX_PATH="/global/homes/n/namehta4/kokkos/install_cuda;/global/homes/n/namehta4/libtorch" ../
make -j
srun -n 1 ./FirstNN'
