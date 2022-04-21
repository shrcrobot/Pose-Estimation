cmake_minimum_required (VERSION 3.0)

project (pcd_prep)

add_definitions(-std=c++11)

find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(VTK REQUIRED)

include( ${VTK_USE_FILE} )

if (PCL_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  link_directories(${PCL_LIBRARY_DIRS})
endif()

 
# ${MPI_CXX_INCLUDE_PATH}

link_directories(main ${Boost_LIBRARY_DIRS})

# ${MPI_CXX_LINK_FLAGS}

add_executable(main ./src/main.cpp)

target_link_libraries(main ${DEPENDENCIES} ${Boost_LIBRARIES} ${PCL_FILTERS_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_COMMON_LIBRARIES}  ${VTK_LIBRARIES})



