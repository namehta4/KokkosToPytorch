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
3) CXX=/global/homes/n/namehta4/kokkos/install_cuda/bin/nvcc_wrapper cmake -DCMAKE_PREFIX_PATH="/global/homes/n/namehta4/kokkos/install_cuda;/global/homes/n/namehta4/libtorch" ../
4) make -j
5) srun -n 1 ./FirstNN'

To compile and run the code on a CPU, use:
1) mkdir build
2) cd build/
3) CC=clang CXX=clang++ cmake -DCMAKE_PREFIX_PATH="/global/homes/n/namehta4/kokkos/install_hsw_serial;/global/homes/n/namehta4/libtorch" ../
4) make -j
5) ./FirstNN'

NOTE: Although the NN written in C++ and python are similar, they are not exactly same. Therefore, output from the two will be different.

* Update 06/09/20: Code runs on both CPU and GPU. However, the C++ part of NN throws a seg fault after successfully running the NN.
* Update 06/10/20: Code runs on both CPU and GPU with no errors after ensuring cuda/10.2.89 is linked.
* Update 06/21/20: Commented out the loop part of NN in both, C++ and Python, because of thread leaks/cuda-memcheck fail on cori gpu nodes.


