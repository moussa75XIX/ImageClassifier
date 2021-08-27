cmake_minimum_required(VERSION 3.19)
project(library)

set(CMAKE_CXX_STANDARD 20)

add_library(library SHARED MLP.cpp MLP.h LINEAR.cpp LINEAR.h)

set(Json_Path "../extern_libraries/")
set(Json_INCLUDE_DIR "${Json_Path}/json")
include_directories(${Json_INCLUDE_DIR})