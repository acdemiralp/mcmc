##################################################    Project     ##################################################
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project               (mcmc VERSION 1.0 LANGUAGES CXX)
list                  (APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set_property          (GLOBAL PROPERTY USE_FOLDERS ON)
set                   (CMAKE_CXX_STANDARD 17)

##################################################    Options     ##################################################
option(BUILD_TESTS "Build tests." OFF)

##################################################    Sources     ##################################################
set(PROJECT_SOURCES
  CMakeLists.txt
  cmake/assign_source_group.cmake
  cmake/import_library.cmake
  readme.md

  include/mcmc/samplers/adaptive_equi_energy_sampler.hpp
  include/mcmc/samplers/differential_evolution_sampler.hpp
  include/mcmc/samplers/gibbs_sampler.hpp
  include/mcmc/samplers/hamiltonian_monte_carlo_sampler.hpp
  include/mcmc/samplers/independent_metropolis_hastings_sampler.hpp
  include/mcmc/samplers/metropolis_adjusted_langevin_sampler.hpp
  include/mcmc/samplers/metropolis_hastings_sampler.hpp
  include/mcmc/samplers/random_walk_metropolis_hastings_sampler.hpp
  include/mcmc/samplers/riemannian_manifold_hamiltonian_monte_carlo_sampler.hpp
  include/mcmc/markov_chain.hpp
  include/mcmc/random_number_generator.hpp
)
include(assign_source_group)
assign_source_group(${PROJECT_SOURCES})

##################################################  Dependencies  ##################################################
include(import_library)

##################################################    Targets     ##################################################
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/external>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/external>)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_INCLUDE_DIRS})
target_link_libraries     (${PROJECT_NAME} INTERFACE ${PROJECT_LIBRARIES})
target_compile_options    (${PROJECT_NAME} INTERFACE -DNOMINMAX)

# Hack for header-only project to appear in the IDEs.
add_library(${PROJECT_NAME}_ STATIC ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME}_ 
  PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/external 
  ${CMAKE_CURRENT_BINARY_DIR})     
target_include_directories(${PROJECT_NAME}_ PUBLIC ${PROJECT_INCLUDE_DIRS})
target_link_libraries     (${PROJECT_NAME}_ PUBLIC ${PROJECT_LIBRARIES})
target_compile_options    (${PROJECT_NAME}_ PUBLIC -DNOMINMAX)
set_target_properties     (${PROJECT_NAME}_ PROPERTIES LINKER_LANGUAGE CXX)

##################################################    Testing     ##################################################
if(BUILD_TESTS)
  enable_testing()

  set(PROJECT_TEST_SOURCES
    tests/adaptive_equi_energy_sampler_test.cpp
    tests/differential_evolution_sampler_test.cpp
    tests/gibbs_sampler_test.cpp
    tests/hamiltonian_monte_carlo_sampler_test.cpp
    tests/independent_metropolis_hastings_sampler_test.cpp
    tests/markov_chain_test.cpp
    tests/metropolis_adjusted_langevin_sampler_test.cpp
    tests/metropolis_hastings_sampler_test.cpp
    tests/random_walk_metropolis_hastings_sampler_test.cpp
    tests/riemannian_manifold_hamiltonian_monte_carlo_sampler_test.cpp
  )

  foreach(_SOURCE ${PROJECT_TEST_SOURCES})
    get_filename_component(_NAME ${_SOURCE} NAME_WE)
    set                   (_SOURCES tests/catch.hpp tests/main.cpp ${_SOURCE})
    add_executable        (${_NAME} ${_SOURCES})
    target_link_libraries (${_NAME} ${PROJECT_NAME})
    target_compile_options(${_NAME} PUBLIC -DNOMINMAX)
    add_test              (${_NAME} ${_NAME})
    set_property          (TARGET ${_NAME} PROPERTY FOLDER "Tests")
    source_group          ("source" FILES ${_SOURCES})
  endforeach()
endif()

##################################################  Installation  ##################################################
install(TARGETS ${PROJECT_NAME} EXPORT "${PROJECT_NAME}-config")
install(DIRECTORY include/ DESTINATION include)
install(EXPORT  "${PROJECT_NAME}-config" DESTINATION "cmake")
export (TARGETS "${PROJECT_NAME}"        FILE        "${PROJECT_NAME}-config.cmake")
