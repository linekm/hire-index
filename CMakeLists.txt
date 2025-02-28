cmake_minimum_required(VERSION 3.16)
project(LearnedTree LANGUAGES C CXX)

include(cmake/buildtype.cmake)
include(cmake/faster.cmake)
#include(cmake/fetchpackage.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O2 -g -DNDEBUG -march=native -march=native -flto -mavx2 -mavx512f -ffast-math")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O2 -g -DNDEBUG -fsanitize=address -march=native -flto -mavx2 -mavx512f")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -march=native -march=native -flto -mavx2 -mavx512f -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb -fno-inline -no-pie -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")

message(STATUS "Building flags ${CMAKE_CXX_FLAGS}.")
message(STATUS "CXX Flags (Debug): ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CXX Flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CXX Flags (RelWithDebInfo): ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CXX Flags (MinSizeRel): ${CMAKE_CXX_FLAGS_MINSIZEREL}")

include(cmake/colors.cmake)

set(CMAKE_CXX_STANDARD 17)

# Check if the user has defined the parameters
#if(NOT DEFINED FANOUT)
#    set(FANOUT 4)
#endif()
#
if (DEFINED FANOUT)
    message(STATUS "Fanout value: ${FANOUT}")
    add_definitions(-DFANOUT=${FANOUT})
endif()
if (DEFINED MAX_EPSILON)
    message(STATUS "Error bound value: ${MAX_EPSILON}")
    add_definitions(-DMAX_EPSILON=${MAX_EPSILON})
endif()
if (DEFINED MODEL_NODE_MIN_SIZE_FACTOR)
    message(STATUS "Model node min size factor: ${MODEL_NODE_MIN_SIZE_FACTOR}")
    add_definitions(-DMODEL_NODE_MIN_SIZE_FACTOR=${MODEL_NODE_MIN_SIZE_FACTOR})
endif()
if (DEFINED MODEL_NODE_MAX_SIZE_FACTOR)
    message(STATUS "Model node max size factor: ${MODEL_NODE_MAX_SIZE_FACTOR}")
    add_definitions(-DMODEL_NODE_MAX_SIZE_FACTOR=${MODEL_NODE_MAX_SIZE_FACTOR})
endif()
if (DEFINED DATASET_RATIO)
    message(STATUS "Use dataset ratio: ${DATASET_RATIO}")
    add_definitions(-DDATASET_RATIO=${DATASET_RATIO})
endif()
if (DEFINED QUERY_RATIO)
    message(STATUS "Use query ratio: ${QUERY_RATIO}")
    add_definitions(-DQUERY_RATIO=${QUERY_RATIO})
endif()
if (DEFINED EPOCH)
    message(STATUS "Run query in ${EPOCH} epochs")
    add_definitions(-DEPOCH=${EPOCH})
endif()
if (DEFINED QUERY_RANGE_SIZE)
    message(STATUS "Query range ${QUERY_RANGE_SIZE}")
    add_definitions(-DQUERY_RANGE_SIZE=${QUERY_RANGE_SIZE})
endif()
# include(FetchContent)
# # fmt
# FetchContent_Declare(fmt
# GIT_REPOSITORY https://github.com/fmtlib/fmt.git
# GIT_TAG 11.0.2
# OVERRIDE_FIND_PACKAGE
# )

# message(STATUS "Fetching fmtlib")
# FetchContent_MakeAvailable(fmt)

# # gtest
# FetchContent_Declare(
# gtest
# GIT_REPOSITORY https://github.com/google/googletest.git
# GIT_TAG v1.15.2
# )
# message(STATUS "Fetching gtest")
# FetchContent_MakeAvailable(gtest)

add_subdirectory(third/fmt)
add_subdirectory(third/googletest)
add_subdirectory(third/json)
add_subdirectory(third/unordered_dense)

# find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)

# Includes:
#
include(cmake/searchfiles.cmake)

include_directories(${FMT_INCLUDE_DIRS})
include_directories(third/unordered_dense/include)

enable_testing()
add_subdirectory(tests)

include(cmake/cppcheck.cmake)
include(cmake/clangformat.cmake)