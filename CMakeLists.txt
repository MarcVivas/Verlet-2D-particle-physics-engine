cmake_minimum_required(VERSION 3.16)
project(collisions LANGUAGES CXX CUDA)

include(FetchContent)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(FETCHCONTENT_QUIET FALSE)

# Add the CUDA libraries
find_package(CUDA REQUIRED)
enable_language(CUDA)

#
# glfw
#
set(GLFW_BUILD_DOCS FALSE)
set(GLFW_BUILD_EXAMPLES FALSE)
set(GLFW_BUILD_TESTS FALSE)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.4
  GIT_PROGRESS TRUE)

#
# glad
#
FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad
  GIT_TAG v0.1.34
  GIT_PROGRESS TRUE)

#
# glm
#
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG 0.9.9.8
  GIT_PROGRESS TRUE)

#
# fetch dependencies
#
FetchContent_MakeAvailable(glfw glad glm)
# N-body source code
file(GLOB PROJECT_SOURCES_CPP "src/*/*.cpp" "src/*/*/*.cpp")
file(GLOB PROJECT_SOURCES_CU "src/*/*.cu" "src/*/*/*.cu")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Include OpenMP
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES_CPP} ${PROJECT_SOURCES_CU})

add_dependencies(${PROJECT_NAME} glfw glad glm)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)

if (MSVC)
    set(my_cxx_flags /openmp::llvm)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${my_cxx_flags}> )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add CUDA-specific settings
set(CUDA_ARCH_LIST "35;50;60;61;75" CACHE STRING "List of CUDA architectures to generate code for")
foreach(arch ${CUDA_ARCH_LIST})
    string(APPEND CUDA_NVCC_FLAGS " -gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()

target_link_libraries(${PROJECT_NAME} PRIVATE glfw glad glm OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})