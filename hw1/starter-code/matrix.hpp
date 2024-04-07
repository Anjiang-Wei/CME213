#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <ostream>
#include <vector>

/*
This is the pure abstract base class specifying general set of functions for a
square matrix.

Concrete classes for specific types of matrices, like MatrixSymmetric, should
implement these functions.
*/
template <typename T>
class Matrix {
  // Sets value of matrix element (i, j).
  virtual void set(int i, int j, T value) = 0;
  // Returns value of matrix element (i, j).
  virtual T operator()(int i, int j) = 0;
  // Number of non-zero elements in matrix.
  virtual unsigned NormL0() const = 0;
  // Enables printing all matrix elements using the overloaded << operator
  virtual void Print(std::ostream& ostream) = 0;

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix<U>& m);
};

/* TODO: Overload the insertion operator by modifying the ostream object */
template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix<T>& m) {
  return stream;
}

/* MatrixDiagonal Class is a subclass of the Matrix class */
template <typename T>
class MatrixDiagonal : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;

  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixDiagonal() : n_(0) {
    // make the data_ vector of size 0
    data_.clear();
  }

  // TODO: Constructor that takes matrix dimension as argument
  MatrixDiagonal(const int n) : n_(n) {
    if (n < 0)
    {
      throw std::invalid_argument("Argument cannot be negative");
    }
    for (int i = 0; i < n; i++) {
      data_.push_back(0);
    }
  }

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  // TODO: Function that sets value of matrix element (i, j).
  void set(int i, int j, T value) override
  {
    if (i == j)
      data_[i] = value;
  }

  // TODO: Function that returns value of matrix element (i, j).
  T operator()(int i, int j) override
  {
    if (i >= (int) data_.size())
    {
      throw std::invalid_argument("Index out of range");
    }
    if (i == j)
      return data_[i];
    else
      return 0;
  }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override
  {
    unsigned int cnt = 0;
    for (size_t i = 0; i < n_; i++) {
      if (data_[i] != 0)
        cnt++;
    }
    return cnt;
  }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {
    for (size_t i = 0; i < n_; i++) {
      for (size_t j = 0; j < n_; j++) {
        if (i == j)
          ostream << data_[i] << " ";
        else
          ostream << 0 << " ";
      }
      ostream << std::endl;
    }
  }
};

/* MatrixSymmetric Class is a subclass of the Matrix class */
template <typename T>
class MatrixSymmetric : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;
  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixSymmetric() {
    n_ = 0;
    data_.clear();
  }

  // TODO: Constructor that takes matrix dimension as argument
  MatrixSymmetric(const int n) {
    if (n < 0)
    {
      throw std::invalid_argument("Argument cannot be negative");
    }
    n_ = n;
    for (int i = 0; i < (n * (n + 1) / 2); i++) {
      data_.push_back(0);
    }
  }

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  // TODO: Function that sets value of matrix element (i, j).
  void set(int i, int j, T value) override {
    if (i > j)
    {
      int tmp = i;
      i = j;
      j = tmp;
    }
    // ensure that i <= j
    int linearize = i * n_ + j - (i + 1) * i / 2;
    // assert linearize < data_.size();
    data_[linearize] = value;
  }

  // TODO: Function that returns value of matrix element (i, j).
  T operator()(int i, int j) override
  {
    if (i > j)
    {
      int tmp = i;
      i = j;
      j = tmp;
    }
    // ensure that i <= j
    int linearize = i * n_ + j - (i + 1) * i / 2;
    if (linearize < (int) data_.size())
      return data_[linearize];
    else
      throw std::invalid_argument("Index out of range");
  }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override
  {
    unsigned int cnt = 0;
    for (size_t i = 0; i < data_.size(); i++) {
      if (data_[i] != 0)
        cnt++;
    }
    return cnt;
  }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override
  {
    for (size_t i = 0; i < n_; i++) {
      for (size_t j = 0; j < n_; j++) {
        ostream << (*this)(i, j) << " ";
      }
      ostream << std::endl;
    }
  }
};

#endif /* MATRIX_HPP */