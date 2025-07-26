/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_

#include "catamari/blas_matrix.hpp"
#include "catamari/buffer.hpp"
#include "catamari/sparse_ldl/supernodal/diagonal_factor.hpp"
#include "catamari/sparse_ldl/supernodal/lower_factor.hpp"
#include "catamari/sparse_ldl/supernodal/supernode_utils.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/SchurComplementStorage.hpp"
#include <tbb/task_group.h>
#include <catamari/sparse_ldl/supernodal/supernode_utils-impl.hpp>
#include <stdexcept>

#ifdef CATAMARI_ENABLE_TIMERS
#include "quotient/timer.hpp"
#endif  // ifdef CATAMARI_ENABLE_TIMERS

#include <Eigen/Dense>

#include "catamari_config.hh"

namespace catamari {

// Sparse record for each (source) input matrix entry of its
// destination in factor_values_.
// This is stored in a compressed-sparse-column-type format.
struct ConversionPlan {
    // Destination and source of each input matrix entry appearing in the factor.
    // Crucially does no value initialization, unlike std::pair!
    struct Entry { Int dst, src; };

    void resize(size_t size) { m_entries.Resize(size); }
    bool empty() const { return m_entries.Size() == 0; }
    Int size() const { return m_entries.Size(); }

    const Entry *entries() const { return m_entries.Data(); }
          Entry *entries()       { return m_entries.Data(); }
    const Entry *columnData(int j) const { return entries() + columnOffsets[j]; }

    Eigen::Array<Int, Eigen::Dynamic, 1> columnOffsets;
private:
    Buffer<Entry> m_entries;
};

template<class Field>
auto eigenMap(BlasMatrixView<Field> &bm) {
    if (bm.leading_dim != bm.height) throw std::runtime_error("map fail!");
    return Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>>(bm.Data(), bm.height, bm.width);
}

template<class Field>
auto eigenMap(BlasMatrix<Field> &bm) {
    return Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>>(bm.Data(), bm.Height(), bm.Width());
}

template<class Field>
auto eigenMap(Buffer<Field> &b) {
    return Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, 1>>(b.Data(), b.Size());
}

template<class Field>
auto eigenMap(const ConstBlasMatrixView<Field> &bm) {
    if (bm.leading_dim != bm.height) throw std::runtime_error("map fail!");
    return Eigen::Map<const Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>>(bm.Data(), bm.height, bm.width);
}
template<class Field>

auto eigenMap(Field *ptr, int size) {
    return Eigen::Map<Eigen::Array<Field, Eigen::Dynamic, 1>>(ptr, size);
}

namespace supernodal_ldl {

// Configuration options for supernodal LDL' factorization.
template <typename Field>
struct Control {
  // Determines the style of the factorization.
  SymmetricFactorizationType factorization_type = kLDLAdjointFactorization;

  // Configuration for the supernodal relaxation.
  SupernodalRelaxationControl relaxation_control;

  // The choice of either left-looking or right-looking LDL' factorization.
  // There is currently no supernodal up-looking support.
  LDLAlgorithm algorithm = kAdaptiveLDL;

  // Whether pivoting within each supernodal diagonal block should be enabled.
  bool supernodal_pivoting = false;

  // The amount of dynamic regularization -- if any -- to use.
  DynamicRegularizationControl<Field> dynamic_regularization;

  // The following thresholds are now ignored in favor of selecting the
  // out-of-place code path based on the platform. This is because experiments
  // have shown the out-of-place update to be always faster than using
  // Accelerate BLAS on Apple Silicon and always slower than MKL on x86

  // // The minimal supernode size for an out-of-place trapezoidal solve to be
  // // used.
  // Int forward_solve_out_of_place_supernode_threshold = 20;

  // // The minimal supernode size for an out-of-place trapezoidal solve to be
  // // used.
  // Int backward_solve_out_of_place_supernode_threshold = 30;

  // The algorithmic block size for the factorization.
  Int block_size = 64;

  // The size of the matrix tiles for factorization OpenMP tasks.
  Int factor_tile_size = 128;

  // The size of the matrix tiles for dense outer product OpenMP tasks.
  Int outer_product_tile_size = 240;

  // The number of columns to group into a single task when multithreading
  // the addition of child Schur complement updates onto the parent.
  Int merge_grain_size = 500;

  // The number of columns to group into a single task when multithreading
  // the scalar structure formation.
  Int sort_grain_size = 200;

  // The minimum ratio of the amount of work in a subtree relative to the
  // nominal amount of flops assigned to each thread (total_work / max_threads)
  // before OpenMP subtasks are launched in the subtree.
  double parallel_ratio_threshold = 0.02;

  // The minimum number of flops in a subtree before OpenMP subtasks are
  // generated.
  double min_parallel_threshold = 1e5;

  bool legacy = false;

#ifdef CATAMARI_ENABLE_TIMERS
  // The max number of levels of the supernodal tree to visualize timings of.
  Int max_timing_levels = 4;

  // Whether isolated diagonal entries should have their timings visualized.
  bool avoid_timing_isolated_roots = true;

  // The name of the Graphviz file for the inclusive timing annotations.
  std::string inclusive_timings_filename = "inclusive.gv";

  // The name of the Graphviz file for the exclusive timing annotations.
  std::string exclusive_timings_filename = "exclusive.gv";
#endif  // ifdef CATAMARI_ENABLE_TIMERS
};

#ifdef CATAMARI_ENABLE_TIMERS
struct FactorizationProfile {
  quotient::Timer scalar_elimination_forest;

  quotient::Timer supernodal_elimination_forest;

  quotient::Timer relax_supernodes;

  quotient::Timer initialize_factors;

  quotient::Timer gemm;
  double gemm_gflops = 0;

  quotient::Timer gemm_unpack;

  quotient::Timer herk;
  double herk_gflops = 0;

  quotient::Timer herk_unpack;

  quotient::Timer trsm;
  double trsm_gflops = 0;

  quotient::Timer cholesky;
  double cholesky_gflops = 0;

  quotient::Timer merge;
  quotient::Timer initialize_columns;

  quotient::Timer left_looking;
  quotient::Timer left_looking_allocate;
  quotient::Timer left_looking_update;
  quotient::Timer left_looking_finalize;

  FactorizationProfile()
      : scalar_elimination_forest("scalar_elimination_forest"),
        supernodal_elimination_forest("supernodal_elimination_forest"),
        relax_supernodes("relax_supernodes"),
        initialize_factors("initialize_factors"),
        gemm("gemm"),
        gemm_unpack("gemm_unpack"),
        herk("herk"),
        herk_unpack("herk_unpack"),
        trsm("trsm"),
        cholesky("cholesky"),
        merge("merge"),
        initialize_columns("initialize_columns"),
        left_looking("left_looking"),
        left_looking_allocate("left_looking_allocate"),
        left_looking_update("left_looking_update"),
        left_looking_finalize("left_looking_finalize") {}

  void Reset() {
    scalar_elimination_forest.Reset(scalar_elimination_forest.Name());
    supernodal_elimination_forest.Reset(supernodal_elimination_forest.Name());
    relax_supernodes.Reset(relax_supernodes.Name());
    initialize_factors.Reset(initialize_factors.Name());
    gemm.Reset(gemm.Name());
    gemm_gflops = 0;
    gemm_unpack.Reset(gemm_unpack.Name());
    herk.Reset(herk.Name());
    herk_gflops = 0;
    herk_unpack.Reset(herk_unpack.Name());
    trsm.Reset(trsm.Name());
    trsm_gflops = 0;
    cholesky.Reset(cholesky.Name());
    cholesky_gflops = 0;
    merge.Reset(merge.Name());
    initialize_columns.Reset(initialize_columns.Name());
    left_looking.Reset(left_looking.Name());
    left_looking_allocate.Reset(left_looking_allocate.Name());
    left_looking_update.Reset(left_looking_update.Name());
    left_looking_finalize.Reset(left_looking_finalize.Name());
  }
};

std::ostream& operator<<(std::ostream& os,
                         const FactorizationProfile& profile) {
#if 0
  os << profile.scalar_elimination_forest << "\n"
     << profile.supernodal_elimination_forest << "\n"
     << profile.relax_supernodes << "\n"
     << profile.initialize_factors << "\n"
     << profile.merge << "\n"
     << profile.gemm << " (GFlops: " << profile.gemm_gflops
     << ", GFlop/sec: " << profile.gemm_gflops / profile.gemm.TotalSeconds()
     << ")\n"
     << profile.gemm_unpack << "\n"
     << profile.herk << " (GFlops: " << profile.herk_gflops
     << ", GFlop/sec: " << profile.herk_gflops / profile.herk.TotalSeconds()
     << ")\n"
     << profile.herk_unpack << "\n"
     << profile.trsm << " (GFlops: " << profile.trsm_gflops
     << ", GFlop/sec: " << profile.trsm_gflops / profile.trsm.TotalSeconds()
     << ")\n"
     << profile.cholesky << " (GFlops: " << profile.cholesky_gflops
     << ", GFlop/sec: "
     << profile.cholesky_gflops / profile.cholesky.TotalSeconds() << ")\n";
  if (profile.left_looking.TotalSeconds() > 0.) {
    os << profile.left_looking << "\n"
       << profile.left_looking_allocate << "\n"
       << profile.left_looking_update << "\n"
       << profile.left_looking_finalize << std::endl;
  }
#else
    os << "{ "
       << "\"scalar_elimination_forest\": " << profile.scalar_elimination_forest.TotalSeconds() << ", "
       << "\"supernodal_elimination_forest\": " << profile.supernodal_elimination_forest.TotalSeconds() << ", "
       << "\"relax_supernodes\": " << profile.relax_supernodes.TotalSeconds() << ", "
       << "\"initialize_factors\": " << profile.initialize_factors.TotalSeconds() << ", "
       << "\"merge\": " << profile.merge.TotalSeconds() << ", "
       << "\"gemm\": " << profile.gemm.TotalSeconds() << ", "
       << "\"gemm_unpack\": " << profile.gemm_unpack.TotalSeconds() << ", "
       << "\"herk\": " << profile.herk.TotalSeconds() << ", "
       << "\"herk_unpack\": " << profile.herk_unpack.TotalSeconds() << ", "
       << "\"trsm\": " << profile.trsm.TotalSeconds() << ", "
       << "\"cholesky\": " << profile.cholesky.TotalSeconds() << ", "
       << "\"initialize_columns\": " << profile.initialize_columns.TotalSeconds() << ", "
       << "\"left_looking\": " << profile.left_looking.TotalSeconds() << ", "
       << "\"left_looking_allocate\": " << profile.left_looking_allocate.TotalSeconds() << ", "
       << "\"left_looking_update\": " << profile.left_looking_update.TotalSeconds() << ", "
       << "\"left_looking_finalize\": " << profile.left_looking_finalize.TotalSeconds()
       << " }" << std::endl;
#endif
  return os;
}
#endif  // ifdef CATAMARI_ENABLE_TIMERS

// The user-facing data structure for storing a supernodal LDL' factorization.
template <class Field>
class Factorization {
 public:
#ifdef CATAMARI_ENABLE_TIMERS
  FactorizationProfile profile;
#endif  // ifdef CATAMARI_ENABLE_TIMERS
  using SolveSharedState = RightLookingSharedState<Field, FineGrainedTimersSolve>;

  // Factors the given matrix using the prescribed permutation.
  SparseLDLResult<Field> Factor(const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& manual_ordering,
                                const Control<Field>& control, bool symbolic_only = false);

  // Factors the given matrix after having previously factored another matrix
  // with the same sparsity pattern.
  // (Legacy version that reads data from `matrix`!)
  SparseLDLResult<Field> RefactorWithFixedSparsityPattern(
      const CoordinateMatrix<Field>& matrix);

  // Factors the given matrix after having previously factored another matrix
  // with the same sparsity pattern -- the control structure is allowed to
  // change in minor ways.
  // (currently disabled)
  SparseLDLResult<Field> RefactorWithFixedSparsityPattern(
      const CoordinateMatrix<Field>& matrix, const Control<Field>& control);

  SparseLDLResult<Field> RefactorWithFixedSparsityPattern(const ConversionPlan &cplan, Int blockSize, const Field *Ax, Field sigma = 0, const Field *Bx = nullptr) {
      CoordinateMatrix<Field> dummy;
      m_inputData.cplan = &cplan;
      m_inputData.Ax = Ax;
      m_inputData.Bx = Bx;
      m_inputData.sigma = sigma;
      if (control_.algorithm == kLeftLookingLDL) {
          SparseLDLResult<Field> result;
          if      (blockSize == 3) result = BlockLeftLooking<3>();
          else if (blockSize == 2) result = BlockLeftLooking<2>();
          else                     result = BlockLeftLooking<1>();
          // auto result = LeftLooking(dummy);
#ifdef CATAMARI_ENABLE_TIMERS
          std::cout << profile << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS
        return result;
      }
      if (blockSize == 3) return BlockRightLooking<3>();
      if (blockSize == 2) return BlockRightLooking<2>();
      return BlockRightLooking<1>();
      // return RightLooking(dummy);
  }

  struct MatrixData {
    const ConversionPlan *cplan = nullptr; // Plan for copying each entry lower_factor_.
    const Field *Ax = nullptr; // Nonzero values of matrix to factor
    Field sigma = 0;           // Hessian modification shift magnitude. This means we factor `A + sigma I` or `A + sigma B` depending on whether `Bx == nullptr`.
    const Field *Bx = nullptr; // Nonzero values of Hessian modification shift
    void injectEntries(const Int j, Field *factorVals, Field &diagEntry) {
      if (Bx) {
        for (const ConversionPlan::Entry *e = cplan->columnData(j); e < cplan->columnData(j + 1); ++e)
            factorVals[e->dst] = Ax[e->src] + sigma * Bx[e->src];
      }
      else {
        for (const ConversionPlan::Entry *e = cplan->columnData(j); e < cplan->columnData(j + 1); ++e)
            factorVals[e->dst] = Ax[e->src];
        diagEntry += sigma;
      }
    }

    // Inject the entries for columns jbegin...jend - 1
    void injectEntries(const Int jbegin, const Int jend, Field *factorVals) {
      const ConversionPlan::Entry *begin = cplan->columnData(jbegin);
      const ConversionPlan::Entry *end   = cplan->columnData(jend);
      if (Bx) {
        for (const ConversionPlan::Entry *e = begin; e < end; ++e)
            factorVals[e->dst] = Ax[e->src] + sigma * Bx[e->src];
      }
      else {
        for (const ConversionPlan::Entry *e = begin; e < end; ++e)
            factorVals[e->dst] = Ax[e->src];
      }
    }
  };
  MatrixData m_inputData;

  // Initialize column `j` of the factor by zero-initializing it and then copying
  // in values of the matrix `A` or `A + sigma B`
  void InitializeFactorColumn(Int j, Int local_j, BlasMatrixView<Field> &diagonal_block) {
      using VMap = Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, 1>>;
      // Note: diagonal_block.leading_dim is the full nonzero height due to how blocks are interleaved
      Int numColumnEntries = diagonal_block.leading_dim - local_j; // Number of nonzero entries on the diagonal of L and below
      VMap(diagonal_block.Pointer(local_j, local_j), numColumnEntries).setZero();
      m_inputData.injectEntries(j, factor_values_.Data(), diagonal_block(local_j, local_j));
  }

  template<Int BlockSize>
  void InitializeFactorBlockColumn(Int j, Int local_j, BlasMatrixView<Field> &diagonal_block) {
      // Zero out this block column
      FillZerosLowerTriangularMiddleCols(diagonal_block.data, local_j, local_j + BlockSize, diagonal_block.leading_dim);

      m_inputData.injectEntries(j, j + BlockSize, factor_values_.Data());
      if (m_inputData.sigma != 0) {
          for (Int c = 0; c < BlockSize; ++c)
              diagonal_block(local_j + c, local_j + c) += m_inputData.sigma;
      }
  }

  void InitializeFactorColumns(Int supernode_offset, Int jstart, Int jend, BlasMatrixView<Field> &diagonal_block) {
      FillZerosLowerTriangularMiddleCols(diagonal_block.data, jstart, jend, diagonal_block.leading_dim);

      m_inputData.injectEntries(supernode_offset + jstart, supernode_offset + jend, factor_values_.Data());
      if (m_inputData.sigma != 0) {
          for (Int c = jstart; c < jend; ++c)
              diagonal_block(c, c) += m_inputData.sigma;
      }
  }

  void InitializeFactorSupernodeColumns(Int supernode_offset, Int supernode_size, BlasMatrixView<Field> &diagonal_block) {
      FillZerosLowerTriangular(diagonal_block.data, diagonal_block.width, diagonal_block.leading_dim);

      m_inputData.injectEntries(supernode_offset, supernode_offset + supernode_size, factor_values_.Data());
      if (m_inputData.sigma != 0) {
          for (Int c = 0; c < supernode_size; ++c)
              diagonal_block(c, c) += m_inputData.sigma;
      }
  }

  // Returns the number of rows in the last factored matrix.
  Int NumRows() const;

  // Solve a set of linear systems using the factorization.
  void Solve(BlasMatrixView<Field>* right_hand_sides, bool already_permuted = false) const;

  // Solves a set of linear systems using the lower-triangular factor.
  void LowerTriangularSolve(BlasMatrixView<Field>* right_hand_sides) const;

  void OpenMPLowerTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides,
      SolveSharedState* shared_state) const;

  // Solves a set of linear systems using the diagonal factor.
  void DiagonalSolve(BlasMatrixView<Field>* right_hand_sides) const;

  void OpenMPDiagonalSolve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using the trasnpose (or adjoint) of the
  // lower-triangular factor.
  void LowerTransposeTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides) const;

  void OpenMPLowerTransposeTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides,
      SolveSharedState* shared_state) const;

  // Prints the diagonal of the factorization.
  void PrintDiagonalFactor(const std::string& label, std::ostream& os) const;

  // Prints the unit lower-triangular matrix.
  void PrintLowerFactor(const std::string& label, std::ostream& os) const;

  Int GetFactorNNZ() const { return nnz_; }

  // Returns a view of the given supernode's permutation vector.
  // NOTE: This is only valid when control.supernodal_pivoting is true.
  BlasMatrixView<Int> SupernodePermutation(Int supernode);

  // Returns a const view of the given supernode's permutation vector.
  // NOTE: This is only valid when control.supernodal_pivoting is true.
  ConstBlasMatrixView<Int> SupernodePermutation(Int supernode) const;

  // Returns an immutable reference to the permutation mapping the original
  // matrix indices into those used for the factorization.
  const Buffer<Int>& Permutation() const;

  // Returns an immutable reference to the permutation mapping the factorization
  // indices into those of the original matrix.
  const Buffer<Int>& InversePermutation() const;

  // Incorporates the details and work required to process the supernode with
  // the given size and degree into the factorization result.
  static void IncorporateSupernodeIntoLDLResult(Int supernode_size, Int degree,
                                                SparseLDLResult<Field>* result);

  // Adds in the contribution of a subtree into an overall result.
  static void MergeContribution(const SparseLDLResult<Field>& contribution,
                                SparseLDLResult<Field>* result);

  // Initializes a supernodal block column of the factorization using the
  // input matrix.
  // Legacy!
  void InitializeBlockColumn(Int supernode,
                             const CoordinateMatrix<Field>& matrix);
#ifdef CATAMARI_OPENMP
  void OpenMPInitializeBlockColumn(Int supernode,
                                   const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  std::unique_ptr<Factorization> Clone() const {
    std::unique_ptr<Factorization> result = std::make_unique<Factorization>();

    result->control_                            = control_;
    result->ordering_                           = ordering_;
    result->supernode_member_to_index_          = supernode_member_to_index_;
    result->max_degree_                         = max_degree_;
    result->max_lower_block_size_               = max_lower_block_size_;
    result->left_looking_workspace_size_        = left_looking_workspace_size_;
    result->left_looking_scaled_transpose_size_ = left_looking_scaled_transpose_size_;
    result->work_estimates_                     = work_estimates_;
    result->total_work_                         = total_work_;

    result->   lower_factor_ = std::make_unique<   LowerFactor<Field>>(*   lower_factor_);
    result->diagonal_factor_ = std::make_unique<DiagonalFactor<Field>>(*diagonal_factor_);
    result->factor_values_   = factor_values_;

    // Point the lower/diagonal factors at the correct data.
    Int dataPtrOffset = result->factor_values_.Data() - factor_values_.Data();
    const Int ns = ordering_.supernode_sizes.Size();
    for (Int s = 0; s < ns; ++s) {
        result->   lower_factor_->blocks[s].data += dataPtrOffset;
        result->diagonal_factor_->blocks[s].data += dataPtrOffset;
    }

    return result;
  }

  std::unique_ptr<Factorization> ExpandSymbolicFactorizationToScalar(Int block_size) const {
      BENCHMARK_SCOPED_TIMER_SECTION timer("ExpandSymbolicFactorizationToScalar");

      std::unique_ptr<Factorization> result = std::make_unique<Factorization>();
      result->control_  = control_;
      result->ordering_ = ordering_;

      // "Upgrade" the supernode sizes to be multiples of the block size.
      using VMap = Eigen::Map<Eigen::Matrix<Int, Eigen::Dynamic, 1>>;
      VMap(result->ordering_.supernode_sizes  .Data(), result->ordering_.supernode_sizes  .Size()) *= block_size;
      VMap(result->ordering_.supernode_offsets.Data(), result->ordering_.supernode_offsets.Size()) *= block_size;

      // Upgrade the ordering permutation
      {
          // BENCHMARK_SCOPED_TIMER_SECTION timer2("UpgradePermutation");
          auto upgrade_permutation = [block_size](Buffer<Int> &perm) {
              size_t m = perm.Size();
              std::decay_t<decltype(perm)> scalarPermutation(block_size * m);
              for (size_t i = 0; i < m; ++i) {
                  for (size_t j = 0; j < block_size; ++j)
                      scalarPermutation[block_size * i + j] = perm[i] * block_size + j;
              }
              perm = std::move(scalarPermutation);
          };

          upgrade_permutation(result->ordering_.permutation);
          upgrade_permutation(result->ordering_.inverse_permutation);
      }

      {
          // BENCHMARK_SCOPED_TIMER_SECTION timer2("Expand supernode_member_to_index");
          Buffer<Int> upgraded_supernode_member_to_index(supernode_member_to_index_.Size() * block_size);
          for (Int i = 0; i < supernode_member_to_index_.Size(); ++i) {
              const Int supernode = supernode_member_to_index_[i];
              for (Int j = 0; j < block_size; ++j)
                  upgraded_supernode_member_to_index[i * block_size + j] = supernode;
          }
          result->supernode_member_to_index_ = std::move(upgraded_supernode_member_to_index);
      }

      result->max_degree_ = max_degree_ * block_size;
      result->max_lower_block_size_ = max_lower_block_size_ * (block_size * block_size);

      const Int num_supernodes = ordering_.supernode_sizes.Size();
      {
          // BENCHMARK_SCOPED_TIMER_SECTION timer2("Allocate Factors");
          Buffer<Int> supernode_degrees(num_supernodes);
          for (Int s = 0; s < num_supernodes; ++s)
              supernode_degrees[s] = block_size * lower_factor_->blocks[s].height;

          result->m_allocateFactors(supernode_degrees);
      }

      {
          // BENCHMARK_SCOPED_TIMER_SECTION timer2("Expand Lower Factor");
          const auto &block_lf = *lower_factor_;
          const auto &block_df = *diagonal_factor_;
          auto      &scalar_lf = *result->lower_factor_;
          auto      &scalar_df = *result->diagonal_factor_;

          if (block_lf.HasValues() || block_df.HasValues())
              throw std::runtime_error("Legacy mode does not support block factorization!");

          // Expand the structure indices.
          parallel_for_range(num_supernodes, [&](size_t s) {
              size_t num_block_structure_indices = std::distance( block_lf.StructureBeg(s),  block_lf.StructureEnd(s));
              size_t num_structure_indices       = std::distance(scalar_lf.StructureBeg(s), scalar_lf.StructureEnd(s));
              if (block_size * num_block_structure_indices != num_structure_indices) { throw std::logic_error("Structure size mismatch"); }

              Int *dst = scalar_lf.StructureBeg(s);
              for (const Int *src = block_lf.StructureBeg(s); src != block_lf.StructureEnd(s); ++src) {
                  Int block_row = *src;
                  for (Int c = 0; c < block_size; ++c) {
                      *dst++ = block_size * block_row + c;
                  }
              }
          }, /* grain_size */ 64, /* parallelism_threshold */ 128);

          // Left-looking-specific structures and workspace sizes
          if (control_.algorithm == kLeftLookingLDL) {
              result->left_looking_workspace_size_        = left_looking_workspace_size_        * (block_size * block_size);
              result->left_looking_scaled_transpose_size_ = left_looking_scaled_transpose_size_ * (block_size * block_size);

              // Note: we could upgrade this rather than recomputing it if
              // we add appropriate accessor methods to `LowerFactor`.
              scalar_lf.FillIntersectionSizes(result->ordering_.supernode_sizes,
                                              result->supernode_member_to_index_);
          }
      }

      return result;
  }

  double EstimateTotalWork() const {
      const Int num_supernodes = ordering_.supernode_sizes.Size();
      double flop_estimate = 0;
      for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
          const ConstBlasMatrixView<Field> &lower_block = lower_factor_->blocks[supernode].ToConst();
          const Int s = lower_block.width;  // supernode size
          const Int d = lower_block.height; // degree
          flop_estimate += s * s * s / 3.; // Factor diagonal block: s^3 / 3
          flop_estimate += s * s * d;      // Solve against diagonal block: s^2 d
          flop_estimate += d * d * s;      // Schur complement update outer products: d^2 s
      }

      return flop_estimate;
  }

  void WriteFinegrainedTimerStats(const std::string &directory, Int max_levels = std::numeric_limits<Int>::max()) const {
      shared_state_.WriteFinegrainedTimerStats(directory, ordering_.assembly_forest, max_levels);
  }

  void WriteFinegrainedSolveTimerStats(const std::string &directory, Int max_levels = std::numeric_limits<Int>::max()) const {
      solve_shared_state_.WriteFinegrainedTimerStats(directory, ordering_.assembly_forest, max_levels);
  }

  void ResetFinegrainedSolveTimerStats() {
      solve_shared_state_.finegrained_timers.clear();
  }

  // Only can be called after `InitialFactorizationSetup` because it needs access to
  // the supernode degrees via an allocated `lower_factor_`.
  void WriteSupernodeStats(const std::string& directory, Int max_levels = std::numeric_limits<Int>::max()) const {
      if (factor_values_.Height() == 0) throw std::runtime_error("Factorization not initialized");
      const bool has_work_estimates = work_estimates_.Size() == ordering_.supernode_sizes.Size();
#if 0
      auto nodeSizeLabel = [this, has_work_estimates](Int supernode) {
          Int size = ordering_.supernode_sizes[supernode];
          const BlasMatrixView<Field> &lower_block = lower_factor_->blocks[supernode];
          std::stringstream ss;
          ss << "size: " << size << "\\ndeg: " << lower_block.Height();
          if (has_work_estimates) ss << "\\nwork: " << work_estimates_[supernode];
          return ss.str();
      };

      WriteTruncatedForestToDot(directory + "/supernodes.dot", nodeSizeLabel, ordering_.assembly_forest, max_levels, /* avoid_isolated_roots = */ false);

      // Also visualize the parallel flag
      for (size_t nt = 2; nt < 32; nt *= 2) {
          double min_parallel_ratio_work = (total_work_ * control_.parallel_ratio_threshold) / nt;
          double min_parallel_work = std::max(control_.min_parallel_threshold, min_parallel_ratio_work);

          WriteTruncatedForestToDot(directory + "/paralellism_for_nthreads_" + std::to_string(nt) + ".dot",
              [this, min_parallel_work](Int supernode) { return std::to_string(work_estimates_[supernode] > min_parallel_work); },
              ordering_.assembly_forest, max_levels, /* avoid_isolated_roots = */ false);
      }
#endif

      // Faster-to-parse format
      WriteForestWithNodeLabels(directory + "/supernodes.txt",
              [this, has_work_estimates](Int supernode) {
                  std::stringstream ss;
                  ss << "{ " << "\"size\": " << ordering_.supernode_sizes[supernode]
                             << ", \"deg\": " << lower_factor_->blocks[supernode].Height();
#if CATAMARI_FINEGRAINED_TIMERS
                  ss << ", \"thread\": " << shared_state_.finegrained_timers.assigned_thread[supernode];
#endif
                  if (has_work_estimates) {
                    ss << ", \"work\": " << work_estimates_[supernode];
                    ss << ", \"parallel_mask\": [";
                    for (size_t nt = 2; nt < 32; nt *= 2) {
                        double min_parallel_ratio_work = (total_work_ * control_.parallel_ratio_threshold) / nt;
                        double min_parallel_work = std::max(control_.min_parallel_threshold, min_parallel_ratio_work);
                        if (nt > 2) ss << ", ";
                        ss << (work_estimates_[supernode] > min_parallel_work);
                    }
                    ss << "]";
                  }
                  ss << " }";

                  return ss.str();
              },
              ordering_.assembly_forest);
  }

 private:
  // The control structure for the factorization.
  Control<Field> control_;

  // The representation of the permutation matrix P so that P A P' should be
  // factored. Typically, this permutation is the composition of a
  // fill-reducing ordering and a supernodal relaxation permutation.
public:
  SymmetricOrdering ordering_;

  // An array of length 'num_rows'; the i'th member is the index of the
  // supernode containing column 'i'.
  Buffer<Int> supernode_member_to_index_;
private:

  // The largest degree of a supernode in the factorization.
  Int max_degree_;

  // The largest number of entries in any supernode's lower block.
  Int max_lower_block_size_;

  // The maximum workspace needed for a diagonal or subdiagonal update of a
  // block column of a supernode.
  Int left_looking_workspace_size_;

  // The maximum workspace needed for storing the scaled transpose of an
  // intersection of a block column with an ancestor supernode.
  // This is only nonzero for LDL^T and LDL^H factorizations.
  Int left_looking_scaled_transpose_size_;

public:
  // The subdiagonal-block portion of the lower-triangular factor.
  std::unique_ptr<LowerFactor<Field>> lower_factor_;

  // The block-diagonal factor.
  std::unique_ptr<DiagonalFactor<Field>> diagonal_factor_;

  // Number of stored scalar nonzeros in L.
  // This is less than the values actually stored since the diagonal blocks are
  // not stored in a packed format.
  Int nnz_ = 0;

  BlasMatrix<Field> factor_values_;
private:

  // If supernodal_pivoting is enabled, all of the supernode permutation
  // vectors are stored within this single buffer.
  BlasMatrix<Int> supernode_permutations_;

  // Julian Panetta: cache work estimates
  Buffer<double> work_estimates_;
  double total_work_;
  mutable Buffer<Field> permute_scratch_;
  mutable SolveSharedState solve_shared_state_;

  // Julian Panetta: cache right-looking factorization shared state
  RightLookingSharedState<Field> shared_state_;

  // Performs the initial analysis (and factorization initialization) for a
  // particular sparsity pattern. Subsequent factorizations with the same
  // sparsity pattern can reuse the symbolic analysis.
  // When `block_size != 1`, each nonzero in `matrix` represents a
  // `block_size x block_size` nonzero block.
  void InitialFactorizationSetup(const CoordinateMatrix<Field>& matrix);
#ifdef CATAMARI_OPENMP
  void OpenMPInitialFactorizationSetup(const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  // TODO(Jack Poulson): Add ReinitializeFactorization.

  // Form the (possibly relaxed) supernodes for the factorization.
  void FormSupernodes(const CoordinateMatrix<Field>& matrix,
                      Buffer<Int>* supernode_degrees);

#ifdef CATAMARI_OPENMP
  void OpenMPFormSupernodes(const CoordinateMatrix<Field>& matrix,
                            Buffer<Int>* supernode_degrees);
#endif  // ifdef CATAMARI_OPENMP

private:
  void InitializeFactors(const CoordinateMatrix<Field>& matrix,
                         const Buffer<Int>& supernode_degrees);
#ifdef CATAMARI_OPENMP
  void OpenMPInitializeFactors(const CoordinateMatrix<Field>& matrix,
                               const Buffer<Int>& supernode_degrees);
#endif  // ifdef CATAMARI_OPENMP


  template<Int BlockSize>
  SparseLDLResult<Field> BlockLeftLooking(); // matrix data is accessed via the ConversionPlan!
  SparseLDLResult<Field> LeftLooking(const CoordinateMatrix<Field>& matrix);

  template<Int BlockSize>
  SparseLDLResult<Field> BlockRightLooking(); // matrix data is accessed via the ConversionPlan!
  SparseLDLResult<Field> OpenMPRightLooking( const CoordinateMatrix<Field>& matrix);

  template<Int BlockSize>
  bool BlockRightLookingSubtree(
      Int supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      const Buffer<double>& work_estimates, double min_parallel_work,
      RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_states,
      SparseLDLResult<Field>* result,
      SchurComplementStorage<Field> *subtreeStorage = nullptr);

  bool OpenMPRightLookingSubtree(
      Int supernode, const CoordinateMatrix<Field>& matrix,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      const Buffer<double>& work_estimates, double min_parallel_work,
      RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_states,
      SparseLDLResult<Field>* result,
      SchurComplementStorage<Field> *subtreeStorage = nullptr);

  template<Int BlockSize>
  void BlockLeftLookingSupernodeUpdate(Int main_supernode,
                                       LeftLookingSharedState* shared_state,
                                       PrivateState<Field>* private_state);

  void LeftLookingSupernodeUpdate(Int main_supernode,
                                  LeftLookingSharedState* shared_state,
                                  PrivateState<Field>* private_state);

  template<Int BlockSize>
  bool BlockRightLookingSupernodeFinalize(
      Int supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_state,
      SparseLDLResult<Field>* result);
  bool LeftLookingSupernodeFinalize(
      Int main_supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      SparseLDLResult<Field>* result);

  bool OpenMPRightLookingSupernodeFinalize(
      Int supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_state,
      SparseLDLResult<Field>* result);

  //////////////////////////////////////////////////////////////////////////////
  // Legacy versions (for performance comparisons)
  //////////////////////////////////////////////////////////////////////////////
  SparseLDLResult<Field> RightLooking(const CoordinateMatrix<Field>& matrix);
  SparseLDLResult<Field> OpenMPRightLookingLegacy( const CoordinateMatrix<Field>& matrix);
  bool RightLookingSubtree( // Legacy
      Int supernode, const CoordinateMatrix<Field>& matrix,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      RightLookingSharedState<Field>* shared_state,
      PrivateState<Field>* private_state, SparseLDLResult<Field>* result);
  bool OpenMPRightLookingLegacySubtree( // Legacy
      Int supernode, const CoordinateMatrix<Field>& matrix,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      const Buffer<double>& work_estimates, double min_parallel_work,
      RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_states,
      SparseLDLResult<Field>* result);
  bool RightLookingSupernodeFinalize(
      Int supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      RightLookingSharedState<Field>* shared_state,
      PrivateState<Field>* private_state, SparseLDLResult<Field>* result);
  bool OpenMPRightLookingLegacySupernodeFinalize(
      Int supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_state,
      SparseLDLResult<Field>* result);

  // Performs the portion of the lower-triangular solve corresponding to the
  // subtree with the given root supernode.
  void LowerTriangularSolveRecursion(Int supernode,
                                     BlasMatrixView<Field>* right_hand_sides,
                                     Buffer<Field>* workspace) const;
  template<size_t BLOCK_SIZE = 1>
  void OpenMPLowerTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      SolveSharedState* shared_state, int level) const;

  // Performs the trapezoidal solve associated with a particular supernode.
  void LowerSupernodalTrapezoidalSolve(Int supernode,
                                       BlasMatrixView<Field>* right_hand_sides,
                                       Buffer<Field>* workspace) const;
  void OpenMPLowerSupernodalTrapezoidalSolve(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      BlasMatrixView<Field> *supernode_schur_complement) const;

  // Performs the portion of the transposed lower-triangular solve
  // corresponding to the subtree with the given root supernode.
  void LowerTransposeTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      Buffer<Field>* packed_input_buf) const;
  void OpenMPLowerTransposeTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      SolveSharedState* shared_state, int level, tbb::task_group &tg) const;

  // Performs the trapezoidal solve associated with a particular supernode.
  void LowerTransposeSupernodalTrapezoidalSolve(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      Buffer<Field>* workspace) const;

  template<size_t BLOCK_SIZE = 1>
  void LowerTransposeSupernodalTrapezoidalSolve(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      BlasMatrixView<Field> &work_right_hand_sides) const;

  void m_allocateFactors(const Buffer<Int> &supernode_degrees);
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/sparse_ldl/supernodal/factorization/common-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/common_openmp-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/io-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/left_looking-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/block_left_looking-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/block_right_looking-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/right_looking_legacy_openmp-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/right_looking-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/right_looking_openmp-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/solve-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/solve_openmp-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_
