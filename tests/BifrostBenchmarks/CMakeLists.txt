set(PROJECT_NAME "BifrostBenchmarks")
project(${PROJECT_NAME})

set(SRCS
  Benchmark.h
  main.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE .)

target_link_libraries(${PROJECT_NAME} Bifrost)

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Tests"
)

# Ensure benchmarks use the same optimization flags as the library
if (NOT MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE -O3 -mavx2 -mfma)
endif()
