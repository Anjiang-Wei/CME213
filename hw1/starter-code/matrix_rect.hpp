#ifndef MATRIX_RECT
#define MATRIX_RECT

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

template <typename T>
class Matrix2D;

template <typename T>
bool Broadcastable(Matrix2D<T>& A, Matrix2D<T>& B) {
  // TODO: Write a function that returns true if either of the matrices can be
  // broadcast to be compatible with the other for elementwise multiplication.
  bool row_broadcastable = (A.size_rows() == 1) || (B.size_rows() == 1) || (A.size_rows() == B.size_rows());
  bool col_broadcastable = (A.size_cols() == 1) || (B.size_cols() == 1) || (A.size_cols() == B.size_cols());
  return row_broadcastable && col_broadcastable;
}

template <typename T>
class Matrix2D {
 private:
  // The size of the matrix is (n_rows, n_cols)
  unsigned int n_rows;
  unsigned int n_cols;

  // Dynamic array storing the data in row major order. Element (i,j) for 
  // 0 <= i < n_rows and 0 <= j < n_cols is stored at data[i * n_cols + j].
  T* data_;

 public:
  // Empty matrix
  Matrix2D() { 
    n_rows = 0;
    n_cols = 0;
  }

  // Constructor takes argument (m,n) = matrix dimension.
  Matrix2D(const int m, const int n) {
      // TODO: Hint: allocate memory for m * n elements using keyword 'new'
      data_ = new T[m * n];
      n_rows = m;
      n_cols = n;
  }

  // Destructor
  ~Matrix2D() {
    // TODO: Hint: Use keyword 'delete'
    n_rows = 0;
    n_cols = 0;
    delete[] data_;
  }

  // Copy constructor
  Matrix2D(const Matrix2D& other) : n_rows(other.n_rows), n_cols(other.n_cols) {
    // TODO
    // Allocate new memory for data_
    data_ = new T[n_rows * n_cols];

    // Copy the elements from the other matrix
    for (unsigned int i = 0; i < n_rows * n_cols; ++i) {
        data_[i] = other.data_[i];
    }
  }

  // Copy assignment operator
  Matrix2D& operator=(const Matrix2D& other) {
    // TODO
    n_rows = other.n_rows;
    n_cols = other.n_cols;
    data_ = new T[n_rows * n_cols];
    for (unsigned int i = 0; i < n_rows * n_cols; ++i) {
        data_[i] = other.data_[i];
    }
    return *this;
  }

  // Move constructor
  Matrix2D(Matrix2D&& other) noexcept 
    : n_rows(other.n_rows), n_cols(other.n_cols), data_(other.data_) {
    // TODO
    other.data_ = nullptr;
    other.n_rows = 0;
    other.n_cols = 0;
  }

  // Move assignment operator
  Matrix2D& operator=(Matrix2D&& other) noexcept {
    // TODO
    n_rows = other.n_rows;
    n_cols = other.n_cols;
    data_ = other.data_;
    return *this;
  }

  unsigned int size_rows() const { return n_rows; } // TODO
  unsigned int size_cols() const { return n_cols; } // TODO

  // Returns reference to matrix element (i, j).
  T& operator()(int i, int j) {
    // TODO: Hint: Element (i,j) for 0 <= i < n_rows and 0 <= j < n_cols 
    // is stored at data[i * n_cols + j]. 
    return data_[i * n_cols + j];
  }
    
  void Print(std::ostream& ostream) {
    // TODO
    for (int i = 0; i < n_rows; i++) {
      for (int j = 0; j < n_cols; j++) {
        ostream << (*this)(i, j) << " ";
      }
      ostream << std::endl;
    }
  }

  Matrix2D<T> dot(Matrix2D<T>& mat) {
    if (n_rows == mat.size_rows() && n_cols == mat.size_cols()) {
      Matrix2D<T> ret(n_rows, n_cols);
      for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = 0; j < n_cols; j++) {
          ret(i, j) = (*this)(i, j) * mat(i, j);
        }
      }
      return ret;
    } else if (Broadcastable<T>(*this, mat)) {
      // TODO: Replace the code in this scope.
      // Compute and return the elementwise product of the two Matrix2D's
      // "*this" and "mat" after appropriate broadcasting.
      unsigned int ret_row = std::max(n_rows, mat.size_rows());
      unsigned int ret_col = std::max(n_cols, mat.size_cols());
      Matrix2D<T> ret(ret_row, ret_col);
      for (unsigned int i = 0; i < ret_row; i++) {
        for (unsigned int j = 0; j < ret_col; j++) {
          {
            int this_i = std::min(i, n_rows - 1);
            int this_j = std::min(j, n_cols - 1);
            int mat_i = std::min(i, mat.size_rows() - 1);
            int mat_j = std::min(j, mat.size_cols() - 1);
            ret(i, j) = (*this)(this_i, this_j) * mat(mat_i, mat_j);
          }
        }
      }

      return ret;
    } else {
      throw std::invalid_argument("Incompatible shapes of the two matrices.");
    }
  }

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix2D<U>& m);
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix2D<T>& m) {
  // TODO
  m.Print(stream);
  return stream;
}

#endif /* MATRIX_RECT */
