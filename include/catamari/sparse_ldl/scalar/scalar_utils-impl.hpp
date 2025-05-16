/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SCALAR_SCALAR_UTILS_IMPL_H_
#define CATAMARI_SPARSE_LDL_SCALAR_SCALAR_UTILS_IMPL_H_

#include "catamari/sparse_ldl/scalar/scalar_utils.hpp"

namespace catamari {
namespace scalar_ldl {

inline void ProcessEdge(Int row, Int column, Buffer<Int>* parents, Buffer<Int>* ancestors) {
  // We are traversing the strict lower triangle.
  if (column >= row) {
    return;
  }

  while (true) {
    const Int ancestor = (*ancestors)[column];
    if (ancestor == row) {
      // We reached the root of the subtree rooted at 'row', so there
      // was no change to the elimination tree.
      break;
    }

    // Compress the path from column to row.
    (*ancestors)[column] = row;

    if (ancestor == -1) {
      // We found a new edge in the elimination tree.
      (*parents)[column] = row;
      break;
    }

    // Move one more step up the tree.
    column = ancestor;
  }
}

template <class Field>
void EliminationForest(const CoordinateMatrix<Field>& matrix,
                       Buffer<Int>* parents, Buffer<Int>* ancestors) {
  const Int num_rows = matrix.NumRows();

  // Initialize all of the parent indices as unset.
  parents->Resize(num_rows, -1);

  // For performing path compression.
  ancestors->Resize(num_rows, -1);

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      ProcessEdge(row, entries[index].column, parents, ancestors);
    }
  }
}

template <class Field>
void EliminationForest(const CoordinateMatrix<Field>& matrix,
                       const SymmetricOrdering& ordering, Buffer<Int>* parents,
                       Buffer<Int>* ancestors) {
  BENCHMARK_SCOPED_TIMER_SECTION timer("EliminationForest");
  if (ordering.permutation.Empty()) {
    return EliminationForest(matrix, parents, ancestors);
  }
  const Int num_rows = matrix.NumRows();

  // Initialize all of the parent indices as unset.
  parents->Resize(num_rows, -1);

  // For performing path compression.
  ancestors->Resize(num_rows, -1);

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int orig_row = ordering.inverse_permutation[row];
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const Int orig_col = entries[index].column;
      // if (orig_col >= orig_row) continue; // Skip upper triangle
      ProcessEdge(row, ordering.permutation[orig_col], parents, ancestors);
      // ProcessEdge(ordering.permutation[entries[index].column], row, parents, ancestors); // JP: Symmetry
    }
  }
}

template <class Field>
void EliminationForest(const CoordinateMatrix<Field>& matrix,
                       Buffer<Int>* parents) {
  Buffer<Int> ancestors;
  EliminationForest(matrix, parents, &ancestors);
}

template <class Field>
void EliminationForest(const CoordinateMatrix<Field>& matrix,
                       const SymmetricOrdering& ordering,
                       Buffer<Int>* parents) {
  Buffer<Int> ancestors;
  EliminationForest(matrix, ordering, parents, &ancestors);
}

template <class Field>
void ParallelEliminationForestAndDegreesRecursion(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    Int root, bool keep_structures, Buffer<Int>* parents, Buffer<Int>* degrees,
    Buffer<Buffer<std::vector<Int>>>* private_children_lists,
    Buffer<Buffer<Int>>* structures, Buffer<Buffer<Int>>* private_pattern_flags,
    Buffer<Buffer<Int>>* private_tmp_structures) {
  const Int order_child_beg = ordering.assembly_forest.child_offsets[root];
  const Int order_child_end = ordering.assembly_forest.child_offsets[root + 1];

  tbb::task_group tg;
  for (Int index = order_child_beg; index < order_child_end; ++index) {
    const Int child = ordering.assembly_forest.children[index];

    tg.run([keep_structures, child, &matrix, &ordering, parents, degrees,
         &private_children_lists, &structures, &private_pattern_flags,
         &private_tmp_structures]() {
            ParallelEliminationForestAndDegreesRecursion(
                matrix, ordering, child, keep_structures, parents, degrees,
                private_children_lists, structures, private_pattern_flags,
                private_tmp_structures);
    });
  }
  tg.wait();

  const int thread = tbb::this_task_arena::current_thread_index();
  Buffer<std::vector<Int>>& children_list = (*private_children_lists)[thread];
  Buffer<Int>& pattern_flags = (*private_pattern_flags)[thread];
  Buffer<Int>& tmp_structure = (*private_tmp_structures)[thread];

  const Int supernode_size = ordering.supernode_sizes[root];
  const Int supernode_offset = ordering.supernode_offsets[root];
  const bool have_permutation = !ordering.permutation.Empty();
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int column = supernode_offset;
       column < supernode_offset + supernode_size; ++column) {
    const Int orig_column =
        have_permutation ? ordering.inverse_permutation[column] : column;
    const Int column_beg = matrix.RowEntryOffset(orig_column);
    const Int column_end = matrix.RowEntryOffset(orig_column + 1);

    Buffer<Int>& structure = (*structures)[column];

    // Merge all threads' children into this thread's list.
    std::vector<Int>& children = children_list[column];
    {
      std::size_t num_total_children = 0;
      for (std::size_t t = 0; t < private_children_lists->Size(); ++t) {
        num_total_children += (*private_children_lists)[t][column].size();
      }

      const std::size_t num_local_children = children.size();
      if (num_total_children > num_local_children) {
        children.resize(num_total_children);
        std::size_t offset = num_local_children;
        for (std::size_t t = 0; t < private_children_lists->Size(); ++t) {
          if (t == std::size_t(thread)) {
            continue;
          }
          std::vector<Int>& other_children =
              (*private_children_lists)[t][column];
          std::copy(other_children.begin(), other_children.end(),
                    children.begin() + offset);
          offset += other_children.size();
          other_children.clear();
          other_children.shrink_to_fit();
        }
        CATAMARI_ASSERT(offset == num_total_children, "Invalid child offset");
      }
    }
    const Int num_children = children.size();

    Int degree = 0;
    Int parent = -1;
    if (num_children == 0) {
      // Incorporate this column's structure.
      for (Int index = column_beg; index < column_end; ++index) {
        const MatrixEntry<Field>& entry = entries[index];
        const Int row = have_permutation ? ordering.permutation[entry.column]
                                         : entry.column;
        if (row > column) {
          if (!degree || row < parent) {
            parent = row;
          }
          tmp_structure[degree++] = row;
        }
      }
    } else {
      // Form this node's structure by unioning that of its direct children
      // (removing portions that intersect this column).
      //
      // We specially handle the first child to avoid unnecessary reads from
      // 'pattern_flags'.
      if (num_children >= 1) {
        const Int child = children[0];
        const Buffer<Int>& child_struct = (*structures)[child];
        for (const Int& row : child_struct) {
          if (row != column) {
            CATAMARI_ASSERT(row > column, "row was < column (EFaDR).");
            pattern_flags[row] = column;
            if (!degree || row < parent) {
              parent = row;
            }
            tmp_structure[degree++] = row;
          }
        }
      }
      for (Int child_index = 1; child_index < num_children; ++child_index) {
        const Int child = children[child_index];
        const Buffer<Int>& child_struct = (*structures)[child];
        for (const Int& row : child_struct) {
          if (row != column && pattern_flags[row] != column) {
            CATAMARI_ASSERT(row > column, "row was < column (EFaDR).");
            pattern_flags[row] = column;
            if (!degree || row < parent) {
              parent = row;
            }
            tmp_structure[degree++] = row;
          }
        }
      }

      // Incorporate this column's structure.
      for (Int index = column_beg; index < column_end; ++index) {
        const MatrixEntry<Field>& entry = entries[index];
        const Int row = have_permutation ? ordering.permutation[entry.column]
                                         : entry.column;
        if (row > column && pattern_flags[row] != column) {
          if (!degree || row < parent) {
            parent = row;
          }
          tmp_structure[degree++] = row;
        }
      }
    }
    structure.Resize(degree);
    std::copy(tmp_structure.begin(), tmp_structure.begin() + degree,
              structure.begin());

    (*parents)[column] = parent;
    (*degrees)[column] = degree;
    if (parent >= 0) {
      children_list[parent].push_back(column);
    }

    // Free the resources of this subtree.
    if (!keep_structures) {
      for (const Int& child : children) {
        (*structures)[child].Clear();
      }
    }
    children.clear();
    children.shrink_to_fit();
  }
}


template <class Field>
void ParallelEliminationForestAndDegrees(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    Buffer<Int>* parents, Buffer<Int>* degrees) {
  BENCHMARK_SCOPED_TIMER_SECTION timer("ParallelEliminationForestAndDegrees");
  const Int num_rows = matrix.NumRows();
  const int max_threads = get_max_num_tbb_threads();
  parents->Resize(num_rows);
  degrees->Resize(num_rows);

  Buffer<Buffer<std::vector<Int>>> private_children_lists(max_threads);

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor. Each thread potentially
  // needs its own since different subtrees can have intersecting structure.
  Buffer<Buffer<Int>> private_pattern_flags(max_threads);

  Buffer<Buffer<Int>> private_tmp_structures(max_threads);

  for (int t = 0; t < max_threads; ++t) {
      private_children_lists[t].Resize(num_rows);
      private_pattern_flags[t].Resize(num_rows, -1);
      private_tmp_structures[t].Resize(num_rows - 1);
  }

  const bool keep_structures = false;
  Buffer<Buffer<Int>> structures(num_rows);

  tbb::task_group tg;
  for (const Int root : ordering.assembly_forest.roots) {
    tg.run(
        [keep_structures, root, &matrix, &ordering, &parents, &degrees,
         &private_children_lists, &structures, &private_pattern_flags,
         &private_tmp_structures]() {
          ParallelEliminationForestAndDegreesRecursion(
              matrix, ordering, root, keep_structures, parents, degrees,
              &private_children_lists, &structures, &private_pattern_flags,
              &private_tmp_structures);
        });
  }
  tg.wait();
}


inline Int PostorderDepthFirstSearch(Int root, Int offset,
                                     const Buffer<Int>& child_lists,
                                     Buffer<Int>* child_list_heads,
                                     std::stack<Int>* node_stack,
                                     Buffer<Int>* postorder) {
  node_stack->push(root);
  while (!node_stack->empty()) {
    const Int top = node_stack->top();
    const Int child = (*child_list_heads)[top];
    if (child == -1) {
      node_stack->pop();
      (*postorder)[offset++] = top;
    } else {
      (*child_list_heads)[top] = child_lists[child];
      node_stack->push(child);
    }
  }
  return offset;
}

inline void PostorderFromEliminationForest(const Buffer<Int>& parents,
                                           Buffer<Int>* postorder) {
  BENCHMARK_SCOPED_TIMER_SECTION timer("PostorderFromEliminationForest");
  const Int num_rows = parents.Size();

  // Construct the linked lists for the lists of children.
  Buffer<Int> child_list_heads(num_rows, -1);
  Buffer<Int> child_lists(num_rows);
  for (Int row = num_rows - 1; row >= 0; --row) {
    const Int parent = parents[row];
    if (parent != -1) {
      // Insert this row into the linked list of the parent.
      child_lists[row] = child_list_heads[parent];
      child_list_heads[parent] = row;
    }
  }

  // TODO(Jack Poulson): Consider using an std::vector or Buffer so that we can
  // preallocate 'num_rows' rather than relying on the std::dequeue dynamic
  // allocation.
  std::stack<Int> node_stack;

  Int offset = 0;
  postorder->Resize(num_rows);
  for (Int row = 0; row < num_rows; ++row) {
    if (parents[row] == -1) {
      // Execute a depth first search from this root in the forst.
      offset = PostorderDepthFirstSearch(
          row, offset, child_lists, &child_list_heads, &node_stack, postorder);
    }
  }
}

// TODO(Jack Poulson): Prevent redundancy with version with an ordering.
template <class Field>
void DegreesFromEliminationForest(const CoordinateMatrix<Field>& matrix,
                                  const Buffer<Int>& parents,
                                  const Buffer<Int>& postorder,
                                  Buffer<Int>* degrees) {
  const Int num_rows = matrix.NumRows();
  degrees->Resize(num_rows);

  // Initialize the first descendants and levels of the elimination tree.
  Buffer<Int> levels(num_rows);
  Buffer<Int> first_descs(num_rows, -1);
  for (Int row_post = 0; row_post < num_rows; ++row_post) {
    const Int row = postorder[row_post];

    // As described in Fig. 3 of Gilbert,Ng, and Peyton's "An efficient
    // algorithm to compute row and column counts for sparse Cholesky
    // factorization", the column counts of leaves should be initialized as
    // 1, whereas those of non-leaves should start at zero.
    // Note that we can identify the 'wt' variable with 'cc'
    // (the column counts, which are one more than the final degrees).
    (*degrees)[row] = first_descs[row] == -1 ? 1 : 0;

    // Traverse up the tree until we either encounter an ancestor whose level
    // we already know, or we hit the root of the tree.
    Int length, ancestor;
    for (length = 0, ancestor = row;
         ancestor != -1 && first_descs[ancestor] == -1;
         ancestor = parents[ancestor], ++length) {
      first_descs[ancestor] = row_post;
    }
    if (ancestor == -1) {
      // Go back down to the root node.
      --length;
    } else {
      // We stopped at an ancestor whose level we knew.
      length += levels[ancestor];
    }

    // Fill in the levels between the start and where we stopped.
    for (Int a = row; a != ancestor; a = parents[a], --length) {
      levels[a] = length;
    }
  }

  Buffer<Int> prev_neighbors(num_rows, -1);
  Buffer<Int> prev_leafs(num_rows, -1);

  // Initialize trivial supernodes.
  Buffer<Int> set_parents(num_rows);
  for (Int i = 0; i < num_rows; ++i) {
    set_parents[i] = i;
  }

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row_post = 0; row_post < num_rows; ++row_post) {
    const Int row = postorder[row_post];

    // Subtract one from the column count of the parent (if it exists).
    const Int parent = parents[row];
    if (parent != -1) {
      --(*degrees)[parent];
    }

    // Mark the equivalent of pattern_flags.
    prev_neighbors[row] = row_post;

    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column = entry.column;
      if (column <= row) {
        continue;
      }

      if (first_descs[row] > prev_neighbors[column]) {
        // 'row' is a leaf in the subtree rooted at 'column'.
        ++(*degrees)[row];
        const Int prev_leaf = prev_leafs[column];
        if (prev_leaf != -1) {
          // Find the root of the set containing 'prev_leaf'.
          Int set_root = prev_leaf;
          while (set_root != set_parents[set_root]) {
            set_root = set_parents[set_root];
          }

          // Walk up the tree from the previous leaf to the set root, filling
          // in the set root as the ancestor of each traversed member.
          Int ancestor = prev_leaf;
          while (ancestor != set_root) {
            const Int next_ancestor = set_parents[ancestor];
            set_parents[ancestor] = set_root;
            ancestor = next_ancestor;
          }

          --(*degrees)[set_root];
        }

        prev_leafs[column] = row;
      }
      prev_neighbors[column] = row_post;
    }

    // Perform UNION(row, parent(row)).
    if (parents[row] != -1) {
      set_parents[row] = parents[row];
    }
  }

  // Accumulate the column counts up the tree and convert from column counts
  // into external degrees.
  for (Int row = 0; row < num_rows; ++row) {
    // Add this column count onto its parent, if it exists.
    const Int parent = parents[row];
    if (parent != -1) {
      (*degrees)[parent] += (*degrees)[row];
    }

    // Convert the column count into a scalar external degree.
    --(*degrees)[row];
  }
}

template <class Field>
void DegreesFromEliminationForest(const CoordinateMatrix<Field>& matrix,
                                  const SymmetricOrdering& ordering,
                                  const Buffer<Int>& parents,
                                  const Buffer<Int>& postorder,
                                  Buffer<Int>* degrees) {
  BENCHMARK_SCOPED_TIMER_SECTION timer("DegreesFromEliminationForest");
  const Int num_rows = matrix.NumRows();
  degrees->Resize(num_rows);

  // Initialize the first descendants and levels of the elimination tree.
  Buffer<Int> levels(num_rows);
  Buffer<Int> first_descs(num_rows, -1);
  for (Int row_post = 0; row_post < num_rows; ++row_post) {
    const Int row = postorder[row_post];

    // As described in Fig. 3 of Gilbert,Ng, and Peyton's "An efficient
    // algorithm to compute row and column counts for sparse Cholesky
    // factorization", the column counts of leaves should be initialized as
    // 1, whereas those of non-leaves should start at zero.
    // Note that we can identify the 'wt' variable with 'cc'
    // (the column counts, which are one more than the final degrees).
    (*degrees)[row] = first_descs[row] == -1 ? 1 : 0;

    // Traverse up the tree until we either encounter an ancestor whose level
    // we already know, or we hit the root of the tree.
    Int length, ancestor;
    for (length = 0, ancestor = row;
         ancestor != -1 && first_descs[ancestor] == -1;
         ancestor = parents[ancestor], ++length) {
      first_descs[ancestor] = row_post;
    }
    if (ancestor == -1) {
      // Go back down to the root node.
      --length;
    } else {
      // We stopped at an ancestor whose level we knew.
      length += levels[ancestor];
    }

    // Fill in the levels between the start and where we stopped.
    for (Int a = row; a != ancestor; a = parents[a], --length) {
      levels[a] = length;
    }
  }

  Buffer<Int> prev_neighbors(num_rows, -1);
  Buffer<Int> prev_leafs(num_rows, -1);

  // Initialize trivial supernodes.
  Buffer<Int> set_parents(num_rows);
  for (Int i = 0; i < num_rows; ++i) {
    set_parents[i] = i;
  }

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row_post = 0; row_post < num_rows; ++row_post) {
    const Int row = postorder[row_post];

    // Subtract one from the column count of the parent (if it exists).
    const Int parent = parents[row];
    if (parent != -1) {
      --(*degrees)[parent];
    }

    // Mark the equivalent of pattern_flags.
    prev_neighbors[row] = row_post;

    const Int orig_row = ordering.inverse_permutation[row];
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column = ordering.permutation[entry.column];
      if (column <= row) {
        continue;
      }

      if (first_descs[row] > prev_neighbors[column]) {
        // 'row' is a leaf in the subtree rooted at 'column'.
        ++(*degrees)[row];
        const Int prev_leaf = prev_leafs[column];
        if (prev_leaf != -1) {
          // Find the root of the set containing 'prev_leaf'.
          Int set_root = prev_leaf;
          while (set_root != set_parents[set_root]) {
            set_root = set_parents[set_root];
          }

          // Walk up the tree from the previous leaf to the set root, filling
          // in the set root as the ancestor of each traversed member.
          Int ancestor = prev_leaf;
          while (ancestor != set_root) {
            const Int next_ancestor = set_parents[ancestor];
            set_parents[ancestor] = set_root;
            ancestor = next_ancestor;
          }

          --(*degrees)[set_root];
        }

        prev_leafs[column] = row;
      }
      prev_neighbors[column] = row_post;
    }

    // Perform UNION(row, parent(row)).
    if (parents[row] != -1) {
      set_parents[row] = parents[row];
    }
  }

  // Accumulate the column counts up the tree and convert from column counts
  // into external degrees.
  for (Int row = 0; row < num_rows; ++row) {
    // Add this column count onto its parent, if it exists.
    const Int parent = parents[row];
    if (parent != -1) {
      (*degrees)[parent] += (*degrees)[row];
    }

    // Convert the column count into a scalar external degree.
    --(*degrees)[row];
  }
}

template <class Field>
void SimpleEliminationForestAndDegrees(const CoordinateMatrix<Field>& matrix,
                                       Buffer<Int>* parents,
                                       Buffer<Int>* degrees) {
  const Int num_rows = matrix.NumRows();

  // Initialize all of the parent indices as unset.
  parents->Resize(num_rows, -1);

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  Buffer<Int> pattern_flags(num_rows);

  // Initialize the number of subdiagonal entries that will be stored into
  // each column.
  degrees->Resize(num_rows, 0);

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    pattern_flags[row] = row;
    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column = entry.column;

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column >= row) {
        break;
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // row index would then be the parent.
      while (pattern_flags[column] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        pattern_flags[column] = row;
        ++(*degrees)[column];

        if ((*parents)[column] == -1) {
          // This is the first occurrence of 'column' in a row pattern.
          (*parents)[column] = row;
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        column = (*parents)[column];
      }
    }
  }
}

template <class Field>
void SimpleEliminationForestAndDegrees(const CoordinateMatrix<Field>& matrix,
                                       const SymmetricOrdering& ordering,
                                       Buffer<Int>* parents,
                                       Buffer<Int>* degrees) {
  if (ordering.permutation.Empty()) {
    SimpleEliminationForestAndDegrees(matrix, parents, degrees);
    return;
  }
  const Int num_rows = matrix.NumRows();

  // Initialize all of the parent indices as unset.
  parents->Resize(num_rows, -1);

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  Buffer<Int> pattern_flags(num_rows);

  // Initialize the number of subdiagonal entries that will be stored into
  // each column.
  degrees->Resize(num_rows, 0);

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    pattern_flags[row] = row;

    const Int orig_row = ordering.inverse_permutation[row];
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column = ordering.permutation[entry.column];

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column >= row) {
        continue;
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // row index would then be the parent.
      while (pattern_flags[column] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        pattern_flags[column] = row;
        ++(*degrees)[column];

        if ((*parents)[column] == -1) {
          // This is the first occurrence of 'column' in a row pattern.
          (*parents)[column] = row;
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        column = (*parents)[column];
      }
    }
  }
}

template <class Field>
void EliminationForestAndDegrees(const CoordinateMatrix<Field>& matrix,
                                 Buffer<Int>* parents, Buffer<Int>* degrees) {
  EliminationForest(matrix, parents);
  Buffer<Int> postorder;
  PostorderFromEliminationForest(*parents, &postorder);
  DegreesFromEliminationForest(matrix, *parents, postorder, degrees);
}

template <class Field>
void EliminationForestAndDegrees(const CoordinateMatrix<Field>& matrix,
                                 const SymmetricOrdering& ordering,
                                 Buffer<Int>* parents, Buffer<Int>* degrees) {
  if (ordering.permutation.Empty()) {
    EliminationForestAndDegrees(matrix, parents, degrees);
    return;
  }
  EliminationForest(matrix, ordering, parents);
  Buffer<Int> postorder;
  PostorderFromEliminationForest(*parents, &postorder);
  DegreesFromEliminationForest(matrix, ordering, *parents, postorder, degrees);
}

template <class Field>
Int ComputeRowPattern(const CoordinateMatrix<Field>& matrix,
                      const SymmetricOrdering& ordering,
                      const Buffer<Int>& parents, Int row, Int* pattern_flags,
                      Int* row_structure) {
  const bool have_permutation = !ordering.permutation.Empty();

  const Int orig_row =
      have_permutation ? ordering.inverse_permutation[row] : row;
  const Int row_beg = matrix.RowEntryOffset(orig_row);
  const Int row_end = matrix.RowEntryOffset(orig_row + 1);
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  Int num_packed = 0;
  for (Int index = row_beg; index < row_end; ++index) {
    const MatrixEntry<Field>& entry = entries[index];
    Int column =
        have_permutation ? ordering.permutation[entry.column] : entry.column;

    if (column >= row) {
      if (have_permutation) {
        continue;
      } else {
        break;
      }
    }

    // Walk up to the root of the current subtree of the elimination
    // forest, stopping if we encounter a member already marked as in the
    // row pattern.
    while (pattern_flags[column] != row) {
      // Place 'column' into the pattern of row 'row'.
      row_structure[num_packed++] = column;
      pattern_flags[column] = row;

      // Move up to the parent in this subtree of the elimination forest.
      column = parents[column];
    }
  }
  return num_packed;
}

template <class Field>
Int ComputeTopologicalRowPatternAndScatterNonzeros(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const Buffer<Int>& parents, Int row, Int* pattern_flags, Int* row_structure,
    Field* row_workspace) {
  const bool have_permutation = !ordering.permutation.Empty();
  Int start = matrix.NumRows();

  const Int orig_row =
      have_permutation ? ordering.inverse_permutation[row] : row;
  const Int row_beg = matrix.RowEntryOffset(orig_row);
  const Int row_end = matrix.RowEntryOffset(orig_row + 1);
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int index = row_beg; index < row_end; ++index) {
    const MatrixEntry<Field>& entry = entries[index];
    Int column =
        have_permutation ? ordering.permutation[entry.column] : entry.column;

    if (column > row) {
      if (have_permutation) {
        continue;
      } else {
        break;
      }
    }

    // Scatter matrix(row, column) into row_workspace.
    row_workspace[column] = entry.value;

    // Walk up to the root of the current subtree of the elimination
    // forest, stopping if we encounter a member already marked as in the
    // row pattern.
    Int num_packed = 0;
    while (pattern_flags[column] != row) {
      // Place 'column' into the pattern of row 'row'.
      row_structure[num_packed++] = column;
      pattern_flags[column] = row;

      // Move up to the parent in this subtree of the elimination forest.
      column = parents[column];
    }

    // Pack this ancestral sequence into the back of the pattern.
    while (num_packed > 0) {
      row_structure[--start] = row_structure[--num_packed];
    }
  }
  return start;
}

template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const SymmetricOrdering& ordering,
                          const AssemblyForest& forest,
                          const Buffer<Int>& degrees,
                          LowerStructure* lower_structure) {
  const Int num_rows = matrix.NumRows();
  const bool have_permutation = !ordering.permutation.Empty();

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  OffsetScan(degrees, &lower_structure->column_offsets);
  lower_structure->indices.Resize(lower_structure->column_offsets.Back());

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  Buffer<Int> pattern_flags(num_rows);

  // A set of pointers for keeping track of where to insert column pattern
  // indices.
  Buffer<Int> column_ptrs(num_rows);

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    pattern_flags[row] = row;
    column_ptrs[row] = lower_structure->ColumnOffset(row);

    const Int orig_row =
        have_permutation ? ordering.inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column =
          have_permutation ? ordering.permutation[entry.column] : entry.column;

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column >= row) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'.
      while (pattern_flags[column] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        pattern_flags[column] = row;
        lower_structure->indices[column_ptrs[column]++] = row;

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        column = forest.parents[column];
      }
    }
  }
}

}  // namespace scalar_ldl
}  // namespace catamari

#include "catamari/sparse_ldl/scalar/scalar_utils/openmp-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SCALAR_SCALAR_UTILS_IMPL_H_
