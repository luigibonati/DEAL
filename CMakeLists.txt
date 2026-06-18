cmake_minimum_required(VERSION 3.18...3.26)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(deal_sgp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ENV{CONDA_PREFIX})
  include_directories($ENV{CONDA_PREFIX}/include)
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")
  set(CMAKE_INCLUDE_PATH "$ENV{CONDA_PREFIX}/include")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(DEAL_SGP_SOURCES
    native/src/flare_pp/y_grad.cpp
    native/src/flare_pp/radial.cpp
    native/src/flare_pp/cutoffs.cpp
    native/src/flare_pp/structure.cpp
    native/src/flare_pp/bffs/sparse_gp.cpp
    native/src/flare_pp/bffs/gp.cpp
    native/src/flare_pp/descriptors/descriptor.cpp
    native/src/flare_pp/descriptors/b2.cpp
    native/src/flare_pp/descriptors/b2_norm.cpp
    native/src/flare_pp/descriptors/b2_simple.cpp
    native/src/flare_pp/descriptors/b3.cpp
    native/src/flare_pp/descriptors/wigner3j.cpp
    native/src/flare_pp/descriptors/two_body.cpp
    native/src/flare_pp/descriptors/three_body.cpp
    native/src/flare_pp/descriptors/three_body_wide.cpp
    native/src/flare_pp/descriptors/four_body.cpp
    native/src/flare_pp/kernels/kernel.cpp
    native/src/flare_pp/kernels/normalized_dot_product.cpp
    native/src/flare_pp/kernels/dot_product.cpp
    native/src/flare_pp/kernels/norm_dot_icm.cpp
    native/src/flare_pp/kernels/squared_exponential.cpp)

include_directories(
  native/src/flare_pp
  native/src/flare_pp/descriptors
  native/src/flare_pp/kernels
  native/src/flare_pp/bffs)

add_library(deal_sgp_native STATIC ${DEAL_SGP_SOURCES})
set_target_properties(deal_sgp_native PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(
  deal_sgp_native
  PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json)

if(NOT DEFINED ENV{NO_OMP})
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(deal_sgp_native PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

if(NOT DEFINED ENV{NO_LAPACK})
  find_package(LAPACK REQUIRED)
  target_link_libraries(deal_sgp_native PUBLIC LAPACK::LAPACK)

  if (LAPACK_LIBRARIES MATCHES mkl)
    target_compile_definitions(deal_sgp_native PUBLIC EIGEN_USE_MKL_ALL=1)
  else()
    find_package(LAPACKE)
    if(LAPACKE_FOUND)
      target_link_libraries(deal_sgp_native PUBLIC ${LAPACKE_LIBRARIES})
      target_compile_definitions(deal_sgp_native PUBLIC EIGEN_USE_LAPACKE=1)
    endif()
  endif()
endif()

pybind11_add_module(deal_sgp_module native/src/ace_binding.cpp)
target_link_libraries(deal_sgp_module PUBLIC deal_sgp_native)
set_target_properties(deal_sgp_module PROPERTIES OUTPUT_NAME "_C_deal_sgp")
install(TARGETS deal_sgp_module LIBRARY DESTINATION deal/sgp)
