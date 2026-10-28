#ifndef CATAMARI_SPARSE_PATTERN_VIEW_H_
#define CATAMARI_SPARSE_PATTERN_VIEW_H_

#include <limits>
#include <stdexcept>
#include <type_traits>
#include "catamari/integers.hpp"

namespace catamari {

// Borrowed, value-free CSR (or symmetric CSC) pattern with sorted, unique
// column indices. The underlying data arrays must outlive analysis.
template <class Index>
class SparsePatternView {
 public:
  SparsePatternView(Index size, const Index* offsets, const Index* indices)
      : size_(size), offsets_(offsets), indices_(indices) {
    static_assert(std::is_integral<Index>::value, "Integral indices required");
    if (size < 0 || size > std::numeric_limits<Int>::max() || !offsets ||
        offsets[0] != 0 || offsets[size] < 0 ||
        offsets[size] > std::numeric_limits<Int>::max() ||
        (offsets[size] && !indices))
      throw std::invalid_argument("Invalid or oversized sparse pattern");
  }

  Int NumRows() const { return size_; }
  Int NumColumns() const { return size_; }
  Int NumEntries() const { return offsets_[size_]; }
  Int RowEntryOffset(Int row) const { return offsets_[row]; }
  Int ColumnIndex(Int entry) const { return indices_[entry]; }
  const Index* OffsetData() const { return offsets_; }
  const Index* IndexData() const { return indices_; }

 private:
  Index size_;
  const Index* offsets_;
  const Index* indices_;
};

}  // namespace catamari
#endif
