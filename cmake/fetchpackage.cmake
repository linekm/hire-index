include(FetchContent)

# fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.0.2
)

message(STATUS "Fetching fmtlib")
FetchContent_MakeAvailable(fmt)

# gtest
FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
message(STATUS "Fetching gtest")
FetchContent_MakeAvailable(gtest)