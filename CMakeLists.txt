cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(Test_Kokkos LANGUAGES CXX)

# find dependencies
add_subdirectory(${Test_Kokkos_SOURCE_DIR}/pybind11)
find_package(Kokkos REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# targets
#file(GLOB SOURCES "src/*.cpp")
#add_executable(test_nnsnap ${SOURCES})
pybind11_add_module(test_kokkos ${Test_Kokkos_SOURCE_DIR}/src/test_kokkos.cpp)

# properties
# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(test_kokkos PUBLIC cxx_std_14)
set_target_properties(test_kokkos PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link/include
target_compile_definitions(test_kokkos PRIVATE -DCUSTOM_SYS_PATH="${Test_Kokkos_SOURCE_DIR}/include")
target_include_directories(test_kokkos SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(test_kokkos PRIVATE $<BUILD_INTERFACE:${Test_Kokkos_SOURCE_DIR}/include>)
target_link_libraries(test_kokkos PRIVATE Kokkos::kokkos)
target_link_libraries(test_kokkos PRIVATE pybind11::headers)
target_link_libraries(test_kokkos PRIVATE Python::Python)
