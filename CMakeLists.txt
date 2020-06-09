cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FirstNN)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
find_package(Kokkos REQUIRED)
#set(Torch_DIR /global/homes/n/namehta4/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

string(APPEND CMAKE_EXE_LINKER_FLAGS "-lpython3.7m")
link_directories(BEFORE /usr/common/software/python/3.7-anaconda-2019.10/lib)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(FirstNN ${SOURCES})
target_link_libraries(FirstNN Kokkos::kokkos)
target_link_libraries(FirstNN "${TORCH_LIBRARIES}")
set_property(TARGET FirstNN PROPERTY CXX_STANDARD 14)
target_compile_definitions(FirstNN PRIVATE -DCUSTOM_SYS_PATH="../include")
