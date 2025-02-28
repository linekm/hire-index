searchfiles()

# find_package(GTest REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_linear_model test_linear_model.cpp)
target_link_libraries(test_linear_model fmt::fmt unordered_dense)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_linear_model OpenMP::OpenMP_CXX)
endif()

include(GoogleTest)
#gtest_discover_tests(test_linear_model)
