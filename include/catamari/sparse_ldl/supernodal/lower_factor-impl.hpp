/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_IMPL_H_

#include "catamari/sparse_ldl/supernodal/lower_factor.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
LowerFactor<Field>::LowerFactor(const Buffer<Int>& supernode_sizes,
                                const Buffer<Int>& supernode_degrees,
                                BlasMatrixView<Field> storage, Int index_block_size)
    : index_block_size_(index_block_size) {
  if (index_block_size < 1) throw std::invalid_argument("Invalid index block size");
  const Int num_supernodes = supernode_sizes.Size();

  Int degree_sum = 0;
  Int num_entries = 0;
  structure_index_offsets_.Resize(num_supernodes + 1);
  blocks.Resize(num_supernodes);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int degree = supernode_degrees[supernode];
    const Int supernode_size = supernode_sizes[supernode];

    structure_index_offsets_[supernode] = degree_sum;

    blocks[supernode].height = degree;
    blocks[supernode].width = supernode_size;
    blocks[supernode].leading_dim = degree;
    blocks[supernode].data = storage.Data() ? storage.Data() + num_entries : nullptr;

    if (degree % index_block_size || supernode_size % index_block_size)
      throw std::invalid_argument("Incomplete index block");
    degree_sum += degree / index_block_size;
    num_entries += degree * supernode_size;
  }

  if (storage.Data() == nullptr) {
    // Legacy mode: allocate owned storage before forming any block pointers.
    values_.Resize(num_entries);
    Int offset = 0;
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      auto &block = blocks[supernode];
      block.data = values_.Data() ? values_.Data() + offset : nullptr;
      offset += block.height * block.width;
    }
  }

  structure_index_offsets_[num_supernodes] = degree_sum;
  // Keep a valid base pointer even for an all-diagonal/empty factor.
  structure_indices_.Resize(std::max<Int>(1, degree_sum));
}

template <class Field>
Int* LowerFactor<Field>::StructureBeg(Int supernode) {
  return structure_indices_.Data() + structure_index_offsets_[supernode];
}

template <class Field>
const Int* LowerFactor<Field>::StructureBeg(Int supernode) const {
  return structure_indices_.Data() + structure_index_offsets_[supernode];
}

template <class Field>
Int* LowerFactor<Field>::StructureEnd(Int supernode) {
  return structure_indices_.Data() + structure_index_offsets_[supernode + 1];
}

template <class Field>
const Int* LowerFactor<Field>::StructureEnd(Int supernode) const {
  return structure_indices_.Data() + structure_index_offsets_[supernode + 1];
}

template <class Field>
Int* LowerFactor<Field>::IntersectionSizesBeg(Int supernode) {
  return &intersect_sizes_[intersect_size_offsets_[supernode]];
}

template <class Field>
const Int* LowerFactor<Field>::IntersectionSizesBeg(Int supernode) const {
  return &intersect_sizes_[intersect_size_offsets_[supernode]];
}

template <class Field>
Int* LowerFactor<Field>::IntersectionSizesEnd(Int supernode) {
  return &intersect_sizes_[intersect_size_offsets_[supernode + 1]];
}

template <class Field>
const Int* LowerFactor<Field>::IntersectionSizesEnd(Int supernode) const {
  return &intersect_sizes_[intersect_size_offsets_[supernode + 1]];
}

template <class Field>
void LowerFactor<Field>::FillIntersectionSizes(
    const Buffer<Int>& /* supernode_sizes */,
    const Buffer<Int>& supernode_member_to_index) {
  const Int num_supernodes = blocks.Size();

  // Compute the supernode offsets.
  std::size_t num_supernode_intersects = 0;
  intersect_size_offsets_.Resize(num_supernodes + 1);
  for (Int column_supernode = 0; column_supernode < num_supernodes;
       ++column_supernode) {
    intersect_size_offsets_[column_supernode] = num_supernode_intersects;
    Int last_supernode = -1;

    const Int* index_beg = StructureBeg(column_supernode);
    const Int* index_end = StructureEnd(column_supernode);
    for (const Int* row_ptr = index_beg; row_ptr != index_end; ++row_ptr) {
      const Int row = *row_ptr;
      const Int supernode = supernode_member_to_index[row];
      if (supernode != last_supernode) {
        last_supernode = supernode;
        ++num_supernode_intersects;
      }
    }
  }
  intersect_size_offsets_[num_supernodes] = num_supernode_intersects;

  // Fill the supernode intersection sizes (and simultaneously compute the
  // number of intersecting descendant entries for each supernode).
  intersect_sizes_.Resize(num_supernode_intersects);
  num_supernode_intersects = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    Int last_supernode = -1;
    Int intersect_size = 0;

    const Int* index_beg = StructureBeg(supernode);
    const Int* index_end = StructureEnd(supernode);
    for (const Int* row_ptr = index_beg; row_ptr != index_end; ++row_ptr) {
      const Int row = *row_ptr;
      const Int row_supernode = supernode_member_to_index[row];
      if (row_supernode != last_supernode) {
        if (last_supernode != -1) {
          // Close out the supernodal intersection.
          intersect_sizes_[num_supernode_intersects++] = intersect_size;
        }
        last_supernode = row_supernode;
        intersect_size = 0;
      }
      intersect_size += index_block_size_;
    }
    if (last_supernode != -1) {
      // Close out the last intersection count for this column supernode.
      intersect_sizes_[num_supernode_intersects++] = intersect_size;
    }
  }
  CATAMARI_ASSERT(num_supernode_intersects == intersect_sizes_.Size(),
                  "Incorrect number of supernode intersections");
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_IMPL_H_
