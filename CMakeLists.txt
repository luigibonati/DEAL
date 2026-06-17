cmake_minimum_required(VERSION 3.18...3.26)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(deal_sgp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/)

include(ExternalProject)
include(FetchContent)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/json
    GIT_TAG v3.9.1)
FetchContent_GetProperties(json)
FetchContent_Populate(json)
add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR})
include_directories(${json_SOURCE_DIR})

ExternalProject_Add(
    eigen_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/Eigen3"
    URL "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.tar.gz"
    URL_HASH MD5=77a2c934eaf35943c43ee600a83b72df
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(eigen_project SOURCE_DIR)
add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 SYSTEM INTERFACE ${SOURCE_DIR})
include_directories(${SOURCE_DIR})

set(PYBIND11_FINDPYTHON ON)
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
add_dependencies(deal_sgp_native eigen_project)
target_link_libraries(deal_sgp_native PUBLIC nlohmann_json::nlohmann_json)

if(NOT DEFINED ENV{NO_OMP})
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(deal_sgp_native PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

if(DEFINED ENV{CONDA_PREFIX})
  include_directories($ENV{CONDA_PREFIX}/include)
  set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")
  set(CMAKE_INCLUDE_PATH "$ENV{CONDA_PREFIX}/include")
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
