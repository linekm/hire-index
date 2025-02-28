set(default_build_type "Debug")
#set(default_build_type "Release")

# if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
#   set(default_build_type "Debug")
# endif()

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  # This flag breaks addr2line:
  #
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  add_definitions(-DNODEBUG)
  message(STATUS "Build type is ${CMAKE_BUILD_TYPE} add NODEBUG definition.")
endif()

message(STATUS "Build type is ${CMAKE_BUILD_TYPE} (default is '${default_build_type}').")