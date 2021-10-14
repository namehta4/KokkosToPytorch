cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(KCuToPytorch LANGUAGES CXX)

# find dependencies
add_subdirectory(${KCuToPytorch_SOURCE_DIR}/pybind11)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(CUDAToolkit REQUIRED)
# targets
#file(GLOB SOURCES "src/*.cpp")
#add_executable(main ${SOURCES})
pybind11_add_module(main ${KCuToPytorch_SOURCE_DIR}/src/main.cpp)

# properties
# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(main PUBLIC cxx_std_14)
set_target_properties(main PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link/include
target_compile_definitions(main PRIVATE -DCUSTOM_SYS_PATH="${KCuToPytorch_SOURCE_DIR}/include")
target_include_directories(main SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(main PRIVATE $<BUILD_INTERFACE:${KCuToPytorch_SOURCE_DIR}/include>)
target_link_libraries(main PRIVATE pybind11::headers)
target_link_libraries(main PRIVATE Python::Python)
target_link_libraries(main PRIVATE CUDA::cudart CUDA::cuda_driver rt)
