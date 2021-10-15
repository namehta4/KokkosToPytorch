cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(Py2Kokkos2Py LANGUAGES CXX)

# find dependencies
add_subdirectory(${Py2Kokkos2Py_SOURCE_DIR}/pybind11)
find_package(Kokkos REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# targets
pybind11_add_module(main ${Py2Kokkos2Py_SOURCE_DIR}/src/main.cpp)

# properties
# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(main PUBLIC cxx_std_14)
set_target_properties(main PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link/include
target_compile_definitions(main PRIVATE -DCUSTOM_SYS_PATH="${Py2Kokkos2Py_SOURCE_DIR}/include")
target_include_directories(main SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(main PRIVATE $<BUILD_INTERFACE:${Py2Kokkos2Py_SOURCE_DIR}/include>)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_link_libraries(main PRIVATE pybind11::headers)
target_link_libraries(main PRIVATE Python::Python)
