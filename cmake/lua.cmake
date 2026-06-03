# =============================================================================
# Lua CMake configuration
#
# This file fetches the Lua interpreter from the official source distribution
# and compiles it into a static library, so that the project does NOT depend on
# a system-installed Lua (e.g. Homebrew). This mirrors how DuckDB and statcpp
# are managed: the source is downloaded into download/ and built locally.
#
# Lua ships with a plain Makefile (not CMake), but its library is just a set of
# C sources. We therefore compile those sources directly as a CMake static
# library target (lua_static), excluding the two files that contain a main()
# function: lua.c (the standalone interpreter) and luac.c (the bytecode
# compiler). This keeps the same toolchain/flags as the rest of the project and
# requires no external "make".
#
# Download directory: ${CMAKE_CURRENT_SOURCE_DIR}/download/lua
# Source directory:   ${CMAKE_CURRENT_SOURCE_DIR}/download/lua/lua-<version>
#
# Additionally, this module defines the compile-time constant:
#   LUA_STATS_SCRIPT_PATH  — absolute path to src/lua/stats.lua in the source tree.
#
# License: MIT License (this cmake file).
# Note: Lua itself is licensed under the MIT License.
# =============================================================================

include_guard(GLOBAL)

message(STATUS "===============================================================")
message(STATUS "Lua configuration:")

# Path to download/source directories
set(LUA_DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/download/lua)
set(LUA_VERSION "5.5.0")
set(LUA_SRC_DIR ${LUA_DOWNLOAD_DIR}/lua-${LUA_VERSION})
set(LUA_URL "https://www.lua.org/ftp/lua-${LUA_VERSION}.tar.gz")

message(STATUS "LUA_SRC_DIR = ${LUA_SRC_DIR}")

# =============================================================================
# Lua: Download and Extract (cached in download/ directory)
# =============================================================================
if(EXISTS ${LUA_SRC_DIR}/src/lua.h)
    message(STATUS "Lua source already present: ${LUA_SRC_DIR}")
else()
    # Check if the archive is already cached in download/
    set(LUA_CACHED_ARCHIVE ${LUA_DOWNLOAD_DIR}/lua-${LUA_VERSION}.tar.gz)

    if(EXISTS ${LUA_CACHED_ARCHIVE})
        message(STATUS "Lua archive already cached: ${LUA_CACHED_ARCHIVE}")
    else()
        file(MAKE_DIRECTORY ${LUA_DOWNLOAD_DIR})
        message(STATUS "Downloading Lua ${LUA_VERSION} from ${LUA_URL} ...")
        file(DOWNLOAD
            ${LUA_URL}
            ${LUA_CACHED_ARCHIVE}
            SHOW_PROGRESS
            TIMEOUT 120
            INACTIVITY_TIMEOUT 30
            STATUS DOWNLOAD_STATUS
        )
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
        if(NOT DOWNLOAD_RESULT EQUAL 0)
            list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR)
            file(REMOVE ${LUA_CACHED_ARCHIVE})
            message(FATAL_ERROR
                "Lua download failed: ${DOWNLOAD_ERROR}\n"
                "You can manually download and place the file:\n"
                "  curl -L -o download/lua/lua-${LUA_VERSION}.tar.gz ${LUA_URL}\n"
                "Then re-run cmake."
            )
        endif()
    endif()

    # Extract archive into download/lua/ (creates lua-<version>/)
    message(STATUS "Extracting Lua ${LUA_VERSION} ...")
    file(ARCHIVE_EXTRACT
        INPUT ${LUA_CACHED_ARCHIVE}
        DESTINATION ${LUA_DOWNLOAD_DIR}
    )

    # Verify extraction
    if(NOT EXISTS ${LUA_SRC_DIR}/src/lua.h)
        message(FATAL_ERROR "Lua extraction failed: ${LUA_SRC_DIR}/src not found")
    endif()

    message(STATUS "Lua ${LUA_VERSION} source ready: ${LUA_SRC_DIR}")
endif()

# =============================================================================
# Build Lua as a static library (lua_static)
# =============================================================================
# Collect all library sources, then drop the two files that define main().
file(GLOB LUA_LIB_SOURCES "${LUA_SRC_DIR}/src/*.c")
list(REMOVE_ITEM LUA_LIB_SOURCES
    "${LUA_SRC_DIR}/src/lua.c"   # standalone interpreter (has main)
    "${LUA_SRC_DIR}/src/luac.c"  # bytecode compiler (has main)
)

add_library(lua_static STATIC ${LUA_LIB_SOURCES})

# Public include dir propagates to anything that links lua_static
target_include_directories(lua_static PUBLIC ${LUA_SRC_DIR}/src)

# Build as PIC and select the platform configuration macro.
# LUA_USE_MACOSX / LUA_USE_LINUX enable POSIX features and dlopen-based loadlib.
set_target_properties(lua_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(APPLE)
    target_compile_definitions(lua_static PRIVATE LUA_USE_MACOSX)
elseif(UNIX)
    target_compile_definitions(lua_static PRIVATE LUA_USE_LINUX)
endif()

# Lua is C and is third-party: do not apply the project's strict warnings to it.
target_compile_options(lua_static PRIVATE -w)

# dlopen / math libraries (no-ops on macOS where they live in libSystem)
target_link_libraries(lua_static PUBLIC ${CMAKE_DL_LIBS})
if(UNIX AND NOT APPLE)
    target_link_libraries(lua_static PUBLIC m)
endif()

# =============================================================================
# Link Lua into the project
# =============================================================================
target_link_libraries(${PROJECT_NAME} PRIVATE lua_static)

# Compile-time constant: path to the Lua stats script (used in main.cpp as LUA_STATS_SCRIPT_PATH)
# CMAKE_CURRENT_SOURCE_DIR when included via include() == the project root (caller's directory).
target_compile_definitions(${PROJECT_NAME} PRIVATE
    LUA_STATS_SCRIPT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/lua/stats.lua"
)

message(STATUS "Lua ${LUA_VERSION} built from source and linked to ${PROJECT_NAME}")
message(STATUS "LUA_STATS_SCRIPT_PATH = ${CMAKE_CURRENT_SOURCE_DIR}/src/lua/stats.lua")
message(STATUS "===============================================================")
