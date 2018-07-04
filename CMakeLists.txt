include(ExternalProject)

#
# Download some not so common dependencies
#

# webp
ExternalProject_Add( 
        webp
        PREFIX "${CMAKE_BINARY_DIR}/webp"
        URL "http://downloads.webmproject.org/releases/webp/libwebp-0.5.1.tar.gz"
        # do not update
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND CFLAGS=-fPIC ../webp/configure --disable-gl --disable-png --disable-jpeg  --disable-tiff --disable-gif --disable-wic
        BUILD_COMMAND "make"
        # do not install
        INSTALL_COMMAND ""
)
ExternalProject_Get_Property( webp SOURCE_DIR BINARY_DIR )
set( webp_INCLUDE_DIR "${SOURCE_DIR}/src" )
set( webp_LIBRARY "${BINARY_DIR}/src/.libs/libwebp.a" )


# lz4
ExternalProject_Add( 
        lz4
        PREFIX "${CMAKE_BINARY_DIR}/lz4"
	URL "https://github.com/lz4/lz4/archive/v1.7.5.tar.gz"
        # do not update
        UPDATE_COMMAND ""
        # copy SOURCE_DIR CMakeLists.txt
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/lz4_cmakelists.txt <SOURCE_DIR>/CMakeLists.txt
        CMAKE_CACHE_ARGS "-DLZ4_BUNDLED_MODE:string=off" "-DBUILD_STATIC_LIBS:string=on" "-DBUILD_SHARED_LIBS:string=on"
        # do not install
        INSTALL_COMMAND ""
)
ExternalProject_Get_Property( lz4 SOURCE_DIR BINARY_DIR )
set( lz4_INCLUDE_DIR "${SOURCE_DIR}/lib" )
set( lz4_STATIC_LIB "${BINARY_DIR}/contrib/cmake_unofficial/liblz4.a" )
