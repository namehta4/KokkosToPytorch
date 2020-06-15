cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FirstNN)

# find dependencies
find_package(Kokkos REQUIRED)
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)

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
target_compile_definitions(FirstNN PRIVATE -DCUSTOM_SYS_PATH="${FirstNN_SOURCE_DIR}/include")
target_include_directories(FirstNN SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(FirstNN PRIVATE $<BUILD_INTERFACE:${FirstNN_SOURCE_DIR}/include>)
target_link_libraries(FirstNN PUBLIC Kokkos::kokkos)
target_link_libraries(FirstNN PUBLIC "${TORCH_LIBRARIES}")

# hack: assume python 3.7
string(APPEND CMAKE_EXE_LINKER_FLAGS "-lpython3.7m")
link_directories(BEFORE /usr/common/software/python/3.7-anaconda-2019.10/lib)
