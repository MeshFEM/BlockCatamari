cmake_minimum_required(VERSION 3.16)
project(Catamari)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(BLA_VENDOR Apple)
else()
    set(BLA_VENDOR Intel10_64_dyn)
    # set(BLA_VENDOR Intel10_64lp_seq)
endif()

find_package(BLAS REQUIRED)

option(CATAMARI_WITH_OPENMP "Enable OpenMP support" OFF)
if (CATAMARI_WITH_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

option(CATAMARI_INT64 "Use 64 bit integers" OFF)
option(CATAMARI_VECTORIZE "Use march=native and enable intrinsics-based kernels" ON)

include(FetchContent)

if(NOT TARGET MeshFEMCore)
    set(MESHFEMCORE_GIT_REPOSITORY "git@github.com:MeshFEM/MeshFEMCore.git"
        CACHE STRING "MeshFEMCore git repository")
    set(MESHFEMCORE_GIT_TAG "2867a326f79269660da80846d273e4440817cbe8"
        CACHE STRING "MeshFEMCore git revision")
    if(NOT DEFINED MESHFEM_EXTERNAL)
        set(MESHFEM_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/subprojects"
            CACHE PATH "External dependency source directory")
    endif()

    FetchContent_Declare(MeshFEMCore
        GIT_REPOSITORY ${MESHFEMCORE_GIT_REPOSITORY}
        GIT_TAG        ${MESHFEMCORE_GIT_TAG}
        SOURCE_DIR     ${MESHFEM_EXTERNAL}/MeshFEMCore
    )
    FetchContent_MakeAvailable(MeshFEMCore)
endif()

FetchContent_GetProperties(MeshFEMCore)
if(meshfemcore_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${meshfemcore_SOURCE_DIR}/cmake")
elseif(DEFINED MESHFEMCORE_ROOT)
    list(APPEND CMAKE_MODULE_PATH "${MESHFEMCORE_ROOT}/cmake")
endif()
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

if(NOT COMMAND meshfem_download_project)
    include(MeshFEMCoreDownloadExternal)
endif()

################################################################################
# Bring in `mimalloc` to accelerate Schur complement memory allocation.
################################################################################
FetchContent_Declare(
  mimalloc
  GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
  GIT_TAG dfa50c37d951128b1e77167dd9291081aa88eea4
)

set(MI_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(MI_BUILD_SHARED ON  CACHE BOOL "" FORCE)
set(MI_BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(MI_OVERRIDE     OFF CACHE BOOL "" FORCE) # Don't replace all malloc/free calls! That degrades performance; mi_malloc must be used selectively.

FetchContent_MakeAvailable(mimalloc)

################################################################################
# Bring in Catamari's header-only subprojects. This replaces the old Meson
# invocation that existed only to populate these source directories.
################################################################################
meshfem_download_project(mantis
    SOURCE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/mantis
    GIT_REPOSITORY https://github.com/jpanetta/mantis.git
    GIT_TAG        b238ee85b04be4586a53f2d83deed55e32d9b65d
    GIT_SUBMODULES_RECURSE TRUE
)

meshfem_download_project(quotient
    SOURCE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/quotient
    GIT_REPOSITORY https://github.com/MeshFEM/quotient
    GIT_TAG        90234550d2899153424d30fbb1f85f001bae90cd
    GIT_SUBMODULES_RECURSE TRUE
)

meshfem_download_project(specify
    SOURCE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/specify
    GIT_REPOSITORY https://gitlab.com/hodge_star/specify.git
    GIT_TAG        3e2cf6f55b516815e998cd276522ae74ca76f70d
)

add_library(mantis INTERFACE)
target_include_directories(mantis SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/mantis/include)
target_compile_features(mantis INTERFACE cxx_std_14)

add_library(specify INTERFACE)
target_include_directories(specify SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/specify/include)
target_compile_features(specify INTERFACE cxx_std_14)

add_library(quotient INTERFACE)
target_include_directories(quotient SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/quotient/include)
target_compile_features(quotient INTERFACE cxx_std_14)
target_link_libraries(quotient INTERFACE mantis specify)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_compile_definitions(mantis INTERFACE -DX86 -DLINUX -DMANTIS_GCC_FUSED_MULTIPLY_ADD)
    target_compile_definitions(quotient INTERFACE -DX86 -DLINUX -DMANTIS_GCC_FUSED_MULTIPLY_ADD)
endif()

if (CATAMARI_INT64)
    target_compile_definitions(quotient INTERFACE -DQUOTIENT_USE_64BIT)
endif()

################################################################################
# Create the `catamari` library target.
################################################################################
if (NOT TARGET catamari)
    add_library(catamari INTERFACE)
    # TODO: actually detect the proper flags here rather than hard-coding for
    # M1-based Mac and x86-based Linux.
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        target_compile_definitions(catamari INTERFACE -DDARWIN -DARM -DCATAMARI_HAVE_FENV_DISABLE_DENORMS -DCATAMARI_HAVE_BLAS -DCATAMARI_HAVE_LAPACK)
    elseif  (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        target_compile_definitions(catamari INTERFACE -DMANTIS_GCC_FUSED_MULTIPLY_ADD -DX86 -DLINUX -DCATAMARI_HAVE_XMMINTRIN -DMANTIS_GCC_FUSED_MULTIPLY_ADD -DCATAMARI_HAVE_MKL -DCATAMARI_HAVE_BLAS -DCATAMARI_HAVE_LAPACK) # -DMKL_DIRECT_CALL_SEQ) # DIRECT_CALL seems not to make much of a difference
        target_include_directories(catamari SYSTEM INTERFACE /usr/include/mkl) # Needed for Ubuntu MKL
    else()
        message(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME")
    endif()

    if (CATAMARI_WITH_OPENMP)
        target_compile_definitions(catamari INTERFACE -DCATAMARI_OPENMP)
        target_link_libraries(catamari INTERFACE OpenMP::OpenMP_CXX)
    endif()

    if (CATAMARI_INT64)
        target_compile_definitions(catamari INTERFACE -DQUOTIENT_USE_64BIT)
    endif()

    if (CATAMARI_VECTORIZE)
        target_compile_options(catamari INTERFACE -march=native) # TODO: also set macro to configure intrinsics kernel guard
    endif()

    target_link_libraries(catamari INTERFACE BLAS::BLAS mimalloc quotient mantis specify MeshFEMCore)
    target_include_directories(catamari SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(catamari SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/quotient/include)
    target_include_directories(catamari SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/mantis/include)
    target_include_directories(catamari SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/specify/include)

    target_include_directories(catamari SYSTEM INTERFACE ${mimalloc_SOURCE_DIR}/include)

    set_target_properties(catamari PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_14)

endif()
