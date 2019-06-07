cmake_minimum_required(VERSION 3.10)
project("HolographicApp")
set(CMAKE_CXX_STANDARD 17)
message("Generating ${PROJECT_NAME}")

if(MSVC_VERSION GREATER 1800 AND "${CMAKE_SYSTEM_NAME}" STREQUAL "WindowsStore")
set(CMAKE_CONFIGURATION_TYPES Release;Debug)
set(COMPILER_VERSION "15")
set(MIN_VERSION 10.0.17134.0)
set(MAX_VERSION 10.0.17763.134) # Current Latest Windows SDK Version as of Jan 8th 2019
set(CMAKE_EXE_LINKER_FLAGS "/LARGEADDRESSAWARE")

file(GLOB ASSETS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/*.png" "${CMAKE_CURRENT_SOURCE_DIR}/../Data/hypersphere_0_2.ppm" "${CMAKE_CURRENT_SOURCE_DIR}/../Data/hypersphere_1_2.ppm" "${CMAKE_CURRENT_SOURCE_DIR}/../Data/hypersphere.ppm" "${CMAKE_CURRENT_SOURCE_DIR}/../Data/output_test.ppm")
file(COPY ${ASSETS_FILES}
   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

# Use debug cube
add_definitions(-DUSE_CUBE_DEBUG)

# Generaate AppXManifest File
set(APP_MANIFEST_NAME Package.appxmanifest)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/GuidGenConsole/GuidGen.exe" OUTPUT_VARIABLE PACKAGE_GUID)
string(STRIP "${PACKAGE_GUID}" PACKAGE_GUID)
set(PUBLISHER_ID "00000000-0000-0000-0000-000000000000")
configure_file(
    Package.appxmanifest.in
    ${CMAKE_CURRENT_BINARY_DIR}/${APP_MANIFEST_NAME}
    @ONLY)

file(GLOB APP_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file(GLOB APP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB COMMON_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/Common/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/Common/*.hpp")
file(GLOB COMMON_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/Common/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/src/Common/*.cpp")
file(GLOB CONTENT_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/Content/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/Content/*.hpp")
file(GLOB CONTENT_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/Content/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/src/Content/*.cpp")

file(GLOB PIXEL_SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*PixelShader.hlsl")
file(GLOB VERTEX_SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*VertexShader.hlsl")
file(GLOB GEOMETRY_SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*GeometryShader.hlsl")

set(CONTENT_FILES 
    ${ASSETS_FILES}
    ${MARKERS_FILES}
    ${PIXEL_SHADER_FILES}
    ${VERTEX_SHADER_FILES}
    ${GEOMETRY_SHADER_FILES}
    ${CMAKE_CURRENT_BINARY_DIR}/${APP_MANIFEST_NAME}
)

set(RESOURCE_FILES
    ${CONTENT_FILES}
    # pfx file
)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Content"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Common"
)

set_property(SOURCE ${CONTENT_FILES} PROPERTY VS_DEPLOYMENT_CONTENT 1)
set_property(SOURCE ${ASSETS_FILES} PROPERTY VS_DEPLOYMENT_CONTENT 1)
set_property(SOURCE ${ASSETS_FILES} PROPERTY VS_DEPLOYMENT_LOCATION "assets")

set_property(SOURCE ${PIXEL_SHADER_FILES} PROPERTY VS_SHADER_TYPE Pixel)
set_property(SOURCE ${PIXEL_SHADER_FILES} PROPERTY VS_SHADER_MODEL 5.0)

set_property(SOURCE ${VERTEX_SHADER_FILES} PROPERTY VS_SHADER_TYPE Vertex)
set_property(SOURCE ${VERTEX_SHADER_FILES} PROPERTY VS_SHADER_MODEL 5.0)

set_property(SOURCE ${GEOMETRY_SHADER_FILES} PROPERTY VS_SHADER_TYPE Geometry)
set_property(SOURCE ${GEOMETRY_SHADER_FILES} PROPERTY VS_SHADER_MODEL 5.0)

source_group("Content\\Source Files" FILES ${CONTENT_SOURCE_FILES})
source_group("Content\\Header Files" FILES ${CONTENT_HEADER_FILES})
source_group("Common\\Source Files" FILES ${COMMON_SOURCE_FILES})
source_group("Common\\Header Files" FILES ${COMMON_HEADER_FILES})
source_group("App\\Source Files" FILES ${APP_SOURCE_FILES})
source_group("App\\Header Files" FILES ${APP_HEADER_FILES})
source_group("Assets Files" FILES ${ASSETS_FILES})
source_group("Shaders\\Pixel" FILES ${PIXEL_SHADER_FILES})
source_group("Shaders\\Vertex" FILES ${VERTEX_SHADER_FILES})
source_group("Shaders\\Geometry" FILES ${GEOMETRY_SHADER_FILES})

add_executable(${PROJECT_NAME} 
    WIN32
    ${CONTENT_SOURCE_FILES}
    ${CONTENT_HEADER_FILES}
    ${COMMON_SOURCE_FILES}
    ${COMMON_HEADER_FILES}
    ${APP_SOURCE_FILES}
    ${APP_HEADER_FILES}
    ${RESOURCE_FILES}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_WINRT_COMPONENT TRUE)
target_link_libraries(${PROJECT_NAME} d3d11)

set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

else()
    message(FETAL_ERROR ": Only support MSVC 14 or MSVC 15, SYSTEM=WindowsStore and SDK VERSION=10.0")
endif()