# HipToPytorch
Modules needed to run this code on AMD MI60:
1) rocm/4.2.0
2) cmake
3) conda environment with pytorch

To compile and run the code on a GPU, use:
1) mkdir build
2) cd build/
3) CXX=clang++ cmake -DCMAKE_PREFIX_PATH="/home/users/coe0221/miniconda3/envs/ExaRL/lib/python3.9/site-packages/torch" ../
4) make -j4
5) srun ./FirstNN'

