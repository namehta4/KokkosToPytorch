# KokkosToPytorch
Modules needed to run this code on Cori:
1) PrgEnv-llvm/12.0.0-git_20210117
2) pytorch/1.8.0-gpu
3) cgpu
4) cuda/11.1.1
5) gcc/8.3.0
6) cudnn/8.0.5
7) cmake/3.18.4

module list as of 05/17/2021
  1) modules/3.2.11.4                                 12) gni-headers/5.0.12.0-7.0.1.1_6.40__g3b1768f.ari  23) esslurm
  2) altd/2.0                                         13) xpmem/2.2.20-7.0.1.1_4.21__g0475745.ari          24) cgpu/1.0
  3) darshan/3.2.1                                    14) job/2.2.4-7.0.1.1_3.48__g36b56f4.ari             25) gcc/8.3.0
  4) craype-network-aries                             15) dvs/2.12_2.2.167-7.0.1.1_17.2__ge473d3a2         26) openmpi/4.0.3
  5) intel/19.0.3.199                                 16) alps/6.6.58-7.0.1.1_6.20__g437d88db.ari          27) llvm/12.0.0-git_20210117
  6) craype/2.6.2                                     17) rca/2.2.20-7.0.1.1_4.62__g8e3fb5b.ari            28) PrgEnv-llvm/12.0.0-git_20210117
  7) cray-libsci/19.06.1                              18) atp/2.1.3                                        29) cuda/11.1.1
  8) udreg/2.3.2-7.0.1.1_3.49__g8175d3d.ari           19) PrgEnv-intel/6.0.5                               30) cudnn/8.0.5
  9) ugni/6.0.14.0-7.0.1.1_7.51__ge78e5b0.ari         20) craype-haswell                                   31) pytorch/1.8.0-gpu
 10) pmi/5.0.14                                       21) cray-mpich/7.7.10                                32) cmake/3.18.4

* Ensure that kokkos is present in the home directory. Build install kokkos cuda in kokkos/install_cuda_reg. Ensure gcc/7.3.0 only when building Kokkos to avoid dependency issues
* Ensure libtorch files are present in the home directory. Can be obtained using 'wget https://download.pytorch.org/libtorch/current*version*pytorch'
* Change lines 9 and 10 in CMakeLists.txt to reflect the location of python installation and python flag 

To compile and run the code on a GPU, use:
1) mkdir build
2) cd build/
3) CXX=/global/homes/n/namehta4/kokkos/install_cuda_reg/bin/nvcc_wrapper cmake -DCMAKE_PREFIX_PATH="/global/homes/n/namehta4/kokkos/install_cuda_reg;/global/homes/n/namehta4/libtorch" ../
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


