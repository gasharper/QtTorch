cmake_minimum_required(VERSION 2.8)
project(QtTorch)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(Torch_DIR D:/libtorch140/share/cmake/Torch)
set(OpenCV_DIR D:/openCV341_msvc/x64/vc15/lib)
set(CMAKE_PREFIX_PATH D:/Qt595/5.10.0/msvc2017_64)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets Core Gui)

set(HEADERS
    main.h
)

set(SOURCES
    main.cpp
)

set(RESOURCES

)

set(QMLS

)


add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} ${RESOURCES} ${QMLS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES} Qt5::Core Qt5::Gui Qt5::Widgets)


