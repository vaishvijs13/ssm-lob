cmake_minimum_required(VERSION 3.18)
project(StreamSSM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

#75=turing, 80=ampere, 86=ampere mobile, 89=ada, 90=hopper
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/framework
  ${CMAKE_SOURCE_DIR}/src/ssm
  ${CMAKE_SOURCE_DIR}/src/cuda
)

add_library(ssm_cuda STATIC
  src/cuda/selective_scan_fwd.cu
  src/cuda/selective_scan_bwd.cu
)

target_link_libraries(ssm_cuda CUDA::cudart)

execute_process(
  COMMAND git rev-parse --short HEAD
  OUTPUT_VARIABLE GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT GIT_HASH)
  set(GIT_HASH "unknown")
endif()

add_executable(benchmark_cuda src/benchmark_cuda.cpp)
target_link_libraries(benchmark_cuda ssm_cuda CUDA::cudart)
target_compile_definitions(benchmark_cuda PRIVATE BUILD_GIT_HASH="${GIT_HASH}")
