# KokkosToPytorch
Modules needed to run this code:
1) PrgEnv-llvm
2) python/3.7-anaconda
3) esslurm
4) cuda/10.2.89
5) gcc/8.3.0
6) pytorch/v1.5.0-gpu

* Ensure that kokkos is present in the home directory. Build install kokkos cuda in kokkos/install_cuda
* Ensure libtorch files are present in the home directory. Can be obtained using 'wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.0.zip'
* Change lines 9 and 10 in CMakeLists.txt to reflect the location of python installation and python flag 

To compile and run the code on a GPU, use:
1) mkdir build
2) cd build/
3) CXX=clang++ cmake -DCMAKE_PREFIX_PATH="/usr/common/software/sles15_cgpu/cuda/11.0.3;/global/homes/n/namehta4/libtorch" ../
4) make -j
5) srun -n 1 ./FirstNN'

