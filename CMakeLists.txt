cmake_minimum_required(VERSION 3.16)
project(Catamari)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

if (NOT TARGET BLAS::BLAS) # Respect any existing BLAS target configured by the parent.
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(BLA_VENDOR Apple)
    else() # Assumes x86
        set(BLA_VENDOR Intel10_64_dyn)
        # set(BLA_VENDOR Intel10_64lp_seq)
    endif()

    find_package(BLAS REQUIRED)
endif()

option(CATAMARI_WITH_OPENMP "Enable OpenMP support" OFF)
if (CATAMARI_WITH_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CATAMARI_ON_X86 TRUE)
else()
    set(CATAMARI_ON_X86 FALSE)
endif()

option(CATAMARI_INT64 "Use 64 bit integers" OFF)
option(CATAMARI_INTRINSICS_KERNELS "Whether to enable the handwritten AVX intrinsics kernels for, e.g., Cholesky solve phase when possible" ${CATAMARI_ON_X86})

set(CATAMARI_INTRINSICS_KERNEL_FLAGS)
if(CATAMARI_INTRINSICS_KERNELS)
    if(MSVC)
        set(CATAMARI_INTRINSICS_KERNEL_FLAGS "/arch:AVX2")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|IntelLLVM")
        set(CATAMARI_INTRINSICS_KERNEL_FLAGS "-mavx;-mfma")
    endif()

    set(CMAKE_REQUIRED_FLAGS_OLD "${CMAKE_REQUIRED_FLAGS}")
    string(REPLACE ";" " " CATAMARI_INTRINSICS_KERNEL_FLAGS_STR "${CATAMARI_INTRINSICS_KERNEL_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${CATAMARI_INTRINSICS_KERNEL_FLAGS_STR}")
    set(CATAMARI_INTRINSICS_KERNEL_TEST_SOURCE "
        #include <immintrin.h>
        int main() {
            __m256d a = _mm256_set1_pd(1.0);
            __m256d b = _mm256_fmadd_pd(a, a, a);
            __m128d lo = _mm256_castpd256_pd128(b);
            __m128d hi = _mm256_extractf128_pd(b, 1);
            __m128d c = _mm_hadd_pd(lo, hi);
            return _mm_cvtsd_f64(c) == 4.0 ? 0 : 1;
        }
    ")
    if(CMAKE_CROSSCOMPILING)
        check_cxx_source_compiles("${CATAMARI_INTRINSICS_KERNEL_TEST_SOURCE}" CATAMARI_INTRINSICS_KERNELS_AVAILABLE)
    else()
        check_cxx_source_runs("${CATAMARI_INTRINSICS_KERNEL_TEST_SOURCE}" CATAMARI_INTRINSICS_KERNELS_AVAILABLE)
    endif()
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_OLD}")
else()
    set(CATAMARI_INTRINSICS_KERNELS_AVAILABLE OFF)
endif()

include(FetchContent)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MeshFEMFetchEditableRepo.cmake")

if(NOT TARGET MeshFEMCore)
    # TODO: change these to https once public.
    set(MESHFEMCORE_GIT_REPOSITORY "git@github.com:MeshFEM/MeshFEMCore.git"
        CACHE STRING "MeshFEMCore git repository")
    set(MESHFEMCORE_GIT_TAG "2867a326f79269660da80846d273e4440817cbe8"
        CACHE STRING "MeshFEMCore git revision")
    if(NOT DEFINED MESHFEM_EXTERNAL)
        set(MESHFEM_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/subprojects"
            CACHE PATH "External dependency source directory")
    endif()

    meshfem_prepare_editable_repo(MeshFEMCore
        "${MESHFEM_EXTERNAL}/MeshFEMCore" "${MESHFEMCORE_GIT_TAG}")
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
    target_compile_definitions(mantis INTERFACE X86 LINUX MANTIS_GCC_FUSED_MULTIPLY_ADD)
    target_compile_definitions(quotient INTERFACE X86 LINUX MANTIS_GCC_FUSED_MULTIPLY_ADD)
endif()

if (CATAMARI_INT64)
    target_compile_definitions(quotient INTERFACE -DQUOTIENT_USE_64BIT)
endif()

################################################################################
# Create the `catamari` library target.
################################################################################
if (NOT TARGET catamari)
    set(CATAMARI_SOURCES
        src/sparse_ldl/supernodal/factorization/trs_kernels.cpp
    )

    add_library(catamari STATIC ${CATAMARI_SOURCES})
    set_target_properties(catamari PROPERTIES POSITION_INDEPENDENT_CODE ON)

    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        target_compile_definitions(catamari PUBLIC DARWIN CATAMARI_HAVE_FENV_DISABLE_DENORMS CATAMARI_HAVE_BLAS CATAMARI_HAVE_LAPACK)
        if(CATAMARI_ON_X86)
            target_compile_definitions(catamari PUBLIC X86 CATAMARI_HAVE_XMMINTRIN)
        else()
            target_compile_definitions(catamari PUBLIC ARM)
        endif()
    elseif  (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        # TODO: support ARM-based Linux?
        target_compile_definitions(catamari PUBLIC MANTIS_GCC_FUSED_MULTIPLY_ADD X86 LINUX CATAMARI_HAVE_XMMINTRIN MANTIS_GCC_FUSED_MULTIPLY_ADD CATAMARI_HAVE_BLAS CATAMARI_HAVE_LAPACK)
        if(TARGET mkl::mkl)
            target_compile_definitions(catamari PUBLIC CATAMARI_HAVE_MKL)
        endif()
    else()
        message(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME")
    endif()

    if (CATAMARI_WITH_OPENMP)
        target_compile_definitions(catamari PUBLIC CATAMARI_OPENMP)
        target_link_libraries(catamari PUBLIC OpenMP::OpenMP_CXX)
    endif()

    if (CATAMARI_INT64)
        target_compile_definitions(catamari PUBLIC -DQUOTIENT_USE_64BIT)
    endif()

    if (CATAMARI_INTRINSICS_KERNELS)
        if (CATAMARI_INTRINSICS_KERNELS_AVAILABLE)
            target_compile_definitions(catamari PUBLIC CATAMARI_SOLVE_AVX_KERNELS)
            set_source_files_properties(
                src/sparse_ldl/supernodal/factorization/trs_kernels.cpp
                PROPERTIES COMPILE_OPTIONS "${CATAMARI_INTRINSICS_KERNEL_FLAGS}"
            )
        else()
            message(STATUS "Catamari AVX/FMA intrinsics kernels requested but unavailable for this compiler/target")
        endif()
    endif()

    target_link_libraries(catamari PUBLIC BLAS::BLAS mimalloc quotient mantis specify MeshFEMCore)
    target_include_directories(catamari SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(catamari SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/quotient/include)
    target_include_directories(catamari SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/mantis/include)
    target_include_directories(catamari SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/specify/include)

    target_include_directories(catamari SYSTEM PUBLIC ${mimalloc_SOURCE_DIR}/include)

    target_compile_features(catamari PUBLIC cxx_std_14)

endif()
