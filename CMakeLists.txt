cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(KokkosToPytorch LANGUAGES CXX)

# find dependencies
find_package(Kokkos REQUIRED)
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# targets
file(GLOB SOURCES "src/*.cpp")
add_executable(FirstNN ${SOURCES})

# properties
# C++ properties: at least a C++14 capable compiler is needed
target_compile_features(FirstNN PUBLIC cxx_std_14)
set_target_properties(FirstNN PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# link/include
target_compile_definitions(FirstNN PRIVATE -DCUSTOM_SYS_PATH="${KokkosToPytorch_SOURCE_DIR}/include")
target_include_directories(FirstNN SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(FirstNN PRIVATE $<BUILD_INTERFACE:${KokkosToPytorch_SOURCE_DIR}/include>)
target_link_libraries(FirstNN PRIVATE Kokkos::kokkos)
target_link_libraries(FirstNN PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(FirstNN PRIVATE Python::Python)
