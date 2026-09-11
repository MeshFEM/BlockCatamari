/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_H_

#include <vector>
#include <iterator>
#include <algorithm>

#include "catamari/blas_matrix_view.hpp"
#include "catamari/buffer.hpp"
#include "catamari_config.hh"

namespace catamari {
namespace supernodal_ldl {

// Read-only scalar view for legacy algorithms and output; never materializes indices.
// Numerical block kernels use StructureBeg directly instead.
class ScalarStructureIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = Int;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = Int;
  ScalarStructureIterator() = default;
  ScalarStructureIterator(const Int *bases, Int block_size, difference_type position = 0)
      : bases_(bases), block_size_(block_size), position_(position) { }
  Int operator*() const { return bases_[position_ / block_size_] + position_ % block_size_; }
  Int operator[](difference_type n) const { return *(*this + n); }
  ScalarStructureIterator &operator++() { ++position_; return *this; }
  ScalarStructureIterator operator++(int) { auto old = *this; ++*this; return old; }
  ScalarStructureIterator &operator--() { --position_; return *this; }
  ScalarStructureIterator operator--(int) { auto old = *this; --*this; return old; }
  ScalarStructureIterator &operator+=(difference_type n) { position_ += n; return *this; }
  ScalarStructureIterator &operator-=(difference_type n) { position_ -= n; return *this; }
  ScalarStructureIterator operator+(difference_type n) const { auto r = *this; return r += n; }
  ScalarStructureIterator operator-(difference_type n) const { auto r = *this; return r -= n; }
  friend ScalarStructureIterator operator+(difference_type n, ScalarStructureIterator i) { return i += n; }
  difference_type operator-(const ScalarStructureIterator &other) const { return position_ - other.position_; }
  bool operator==(const ScalarStructureIterator &o) const { return bases_ == o.bases_ && position_ == o.position_; }
  bool operator!=(const ScalarStructureIterator &o) const { return !(*this == o); }
  bool operator<(const ScalarStructureIterator &o) const { return position_ < o.position_; }
  bool operator>(const ScalarStructureIterator &o) const { return o < *this; }
  bool operator<=(const ScalarStructureIterator &o) const { return !(o < *this); }
  bool operator>=(const ScalarStructureIterator &o) const { return !(*this < o); }
 private:
  const Int *bases_ = nullptr;
  Int block_size_ = 1;
  difference_type position_ = 0;
};

// The representation of the portion of the unit-lower triangular factor
// that is below the supernodal diagonal blocks.
template <class Field>
class LowerFactor {
 public:
  // Representations of the densified subdiagonal blocks of the factorization.
  Buffer<BlasMatrixView<Field>> blocks;

  LowerFactor(const Buffer<Int>& supernode_sizes,
              const Buffer<Int>& supernode_degrees,
              BlasMatrixView<Field> storage, Int index_block_size = 1);

  // Legacy constructor (noninterleaved lower/diagonal factor storage).
  LowerFactor(const Buffer<Int>& supernode_sizes,
              const Buffer<Int>& supernode_degrees)
      : LowerFactor(supernode_sizes, supernode_degrees, BlasMatrixView<Field>()) { }

  // Dense block dimensions and index values are scalar; index counts are in blocks.
  Int IndexBlockSize() const { return index_block_size_; }
  Int NumStructureEntries() const { return structure_index_offsets_[blocks.Size()]; }
  ScalarStructureIterator ScalarStructureBeg(Int s) const {
    return {StructureBeg(s), index_block_size_};
  }
  ScalarStructureIterator ScalarStructureEnd(Int s) const {
    return {StructureBeg(s), index_block_size_, blocks[s].height};
  }
  // Locate a scalar row with a single search of the compact block-base list.
  Int FindScalarRow(Int s, Int row) const {
    const Int component = row % index_block_size_;
    const Int base = row - component;
    const Int *beg = StructureBeg(s), *end = StructureEnd(s);
    const Int *it = std::lower_bound(beg, end, base);
    if (it == end || *it != base) throw std::runtime_error("Row absent from lower factor structure");
    return index_block_size_ * (it - beg) + component;
  }

  // One scalar base offset per index block; no component indices are stored.
  // Returns a pointer to the beginning of the structure of a supernode.
  Int* StructureBeg(Int supernode);

  // Returns an immutable pointer to the beginning of the structure of a
  // supernode.
  const Int* StructureBeg(Int supernode) const;

  // Returns a pointer to the end of the structure of a supernode.
  Int* StructureEnd(Int supernode);

  // Returns an immutable pointer to the end of the structure of a supernode.
  const Int* StructureEnd(Int supernode) const;

  // Returns a pointer to the beginning of the supernodal intersection sizes
  // of a supernode.
  Int* IntersectionSizesBeg(Int supernode);

  // Returns an immutable pointer to the beginning of the supernodal
  // intersection sizes of a supernode.
  const Int* IntersectionSizesBeg(Int supernode) const;

  // Returns a pointer to the end of the supernodal intersection sizes of a
  // supernode.
  Int* IntersectionSizesEnd(Int supernode);

  // Returns an immutable pointer to the end of the supernodal intersection
  // sizes of a supernode.
  const Int* IntersectionSizesEnd(Int supernode) const;

  void FillIntersectionSizes(const Buffer<Int>& supernode_sizes,
                             const Buffer<Int>& supernode_member_to_index);

  bool HasValues() const { return values_.Data() != nullptr; }

 private:
  Int index_block_size_ = 1;

  // The concatenation of the structures of the supernodes. The structure of
  // supernode j is stored between indices index_offsets[j] and
  // index_offsets[j + 1].
  Buffer<Int> structure_indices_;

  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // block degrees (excluding the diagonal blocks) of supernodes 0 through j - 1.
  Buffer<Int> structure_index_offsets_;

  // The concatenation of the number of rows in each supernodal intersection.
  // The supernodal intersection sizes for supernode j are stored in indices
  // intersect_size_offsets[j] through intersect_size_offsets[j + 1].
  Buffer<Int> intersect_sizes_;

  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // number of supernodes that supernodes 0 through j - 1 individually intersect
  // with.
  Buffer<Int> intersect_size_offsets_;

  // The concatenation of the numerical values of the supernodal structures.
  // The entries of supernode j are stored between indices value_offsets[j] and
  // value_offsets[j + 1] in a column-major manner.
  // Only used in legacy mode!
  Buffer<Field> values_;
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/sparse_ldl/supernodal/lower_factor-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_H_
