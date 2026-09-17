cmake_minimum_required(VERSION 3.20)
project(CacheProblem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flag for performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# openGA uses std::thread, so we link threads
find_package(Threads REQUIRED)

# Add both source files into one executable
add_executable(solver sol.cpp evolAlgo.cpp
        global.h)

# Link pthread to the target
target_link_libraries(solver PRIVATE Threads::Threads)

add_executable(judge
        "judgeHashCode2017.cpp"
)