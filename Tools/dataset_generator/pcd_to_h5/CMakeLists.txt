cmake_minimum_required (VERSION 3.0)

project (pcdToH5)

add_definitions(-std=c++11)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(MPI REQUIRED COMPONENTS CXX MPICXX)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(VTK REQUIRED)

include( ${VTK_USE_FILE} )

if (PCL_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  link_directories(${PCL_LIBRARY_DIRS})
endif()

#IF(MPI_CXX_FOUND)
#   include_directories(${MPI_CXX_INCLUDE_PATH})
#   LIST(APPEND SCR_EXTERNAL_LIBS ${MPI_CXX_LIBRARIES})
#ENDIF(MPI_CXX_FOUND)

add_definitions(-DOMPI_SKIP_MPICXX)

include_directories(${HDF5_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
 
# ${MPI_CXX_INCLUDE_PATH}

link_directories(main ${Boost_LIBRARY_DIRS})

# ${MPI_CXX_LINK_FLAGS}


add_executable(main ./src/main.cpp ./src/my_io.cpp)

target_link_libraries(main ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_COMMON_LIBRARIES}  ${VTK_LIBRARIES} ${MPI_CXX_LIBRARIES})



