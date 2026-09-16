cmake_minimum_required(VERSION 3.26)
project(HPC_Exercises)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optimise for Release builds
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")


# List of executables – each exercise lives in its own source file
#add_executable(2a 2a.c)
#add_executable(2b 2b.c)
add_executable(2c 2c.c read_write_pgm_image.c)

# OpenMP support
find_package(OpenMP)
if(OPENMP_FOUND)
  target_link_libraries(2c
    PUBLIC OpenMP::OpenMP_C      # OpenMP
    PRIVATE ${MPI_C_LIBRARIES}   # MPI
    PRIVATE m                    # math
  )
else()
  target_link_libraries(2c
    PRIVATE ${MPI_C_LIBRARIES}   # MPI
    PRIVATE m                    # math
  )
endif()

# MPI support
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})