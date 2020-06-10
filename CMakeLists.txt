cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FirstNN)

find_package(Kokkos REQUIRED)
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

string(APPEND CMAKE_EXE_LINKER_FLAGS "-lpython3.7m")
link_directories(BEFORE /usr/common/software/python/3.7-anaconda-2019.10/lib)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(FirstNN ${SOURCES})
target_link_libraries(FirstNN Kokkos::kokkos)
target_link_libraries(FirstNN "${TORCH_LIBRARIES}")
target_compile_definitions(FirstNN PRIVATE -DCUSTOM_SYS_PATH="../include")
set_property(TARGET FirstNN PROPERTY CXX_STANDARD 14)
