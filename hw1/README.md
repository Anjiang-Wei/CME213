Problem 1:
I implement all the required test cases, and the output is as follows:
```
[==========] Running 8 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 8 tests from testMatrix
[ RUN      ] testMatrix.sampleTest
[       OK ] testMatrix.sampleTest (0 ms)
[ RUN      ] testMatrix.default_constructor
[       OK ] testMatrix.default_constructor (0 ms)
[ RUN      ] testMatrix.second_constructor
[       OK ] testMatrix.second_constructor (0 ms)
[ RUN      ] testMatrix.exception_second_constructor
[       OK ] testMatrix.exception_second_constructor (0 ms)
[ RUN      ] testMatrix.norm_l0
[       OK ] testMatrix.norm_l0 (0 ms)
[ RUN      ] testMatrix.retrieve
[       OK ] testMatrix.retrieve (0 ms)
[ RUN      ] testMatrix.exception_out_of_bound_access
[       OK ] testMatrix.exception_out_of_bound_access (0 ms)
[ RUN      ] testMatrix.TestPrint
[       OK ] testMatrix.TestPrint (0 ms)
[----------] 8 tests from testMatrix (0 ms total)

[----------] Global test environment tear-down
[==========] 8 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 8 tests.
```
- Data layout for "MatrixDiagonal" class: only store the elements on the diagonal

- Data layout for the "MatrixSymmetric" class: only store elements where i <= j.
I linearize the 2D matrix into a 1D array according to this formula:
    int linearize = i * n_ + j - (i + 1) * i / 2;

Problem 2:
I implement the 4 required tests. Result is
```
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from testMatrix2D
[ RUN      ] testMatrix2D.sampleTest
[       OK ] testMatrix2D.sampleTest (0 ms)
[ RUN      ] testMatrix2D.test1
[       OK ] testMatrix2D.test1 (0 ms)
[ RUN      ] testMatrix2D.test2
[       OK ] testMatrix2D.test2 (0 ms)
[ RUN      ] testMatrix2D.test3
[       OK ] testMatrix2D.test3 (0 ms)
[ RUN      ] testMatrix2D.test4
[       OK ] testMatrix2D.test4 (0 ms)
[----------] 5 tests from testMatrix2D (0 ms total)

[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 5 tests.
```
The broadcasting rule is simple:
If one of the two dimensions is 1, or the sizes of the two dimensions are the same, then the dimension is broadcastable.
For a 2D matrix to be broadcastable, both row and column have to be broadcastable.
```
  bool row_broadcastable = (A.size_rows() == 1) || (B.size_rows() == 1) || (A.size_rows() == B.size_rows());
  bool col_broadcastable = (A.size_cols() == 1) || (B.size_cols() == 1) || (A.size_cols() == B.size_cols());
  return row_broadcastable && col_broadcastable;
```

In the implementation of the multiplication, we use the following logic:
```
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
```
The result matrix's size is decided by the maximum of the two matrices' dimensions. the "std::min" will compute to 0 as the index for the 1-size dimension.

Copy/Move constructor is necessary for dynamically allocated array because of the need for deep copy. Shallow copy can lead to double-free problem and dangling pointer problem.
Also, move construcotr and move assignment operator can make resource transfer more efficient and prevent resource leak.


Problem 3:

1)  Run-time Polymorphism: In the given code, run-time polymorphism is implemented through the use of virtual functions and inheritance. The Matrix class declares a pure virtual function repr(). This function is then overridden in the derived classes SparseMatrix and ToeplitzMatrix. The choice of which repr() function to call (from SparseMatrix or ToeplitzMatrix) is made at run time, depending on the type of object pointed to by the Matrix pointer in the vector. This run-time behavior is a classic example of polymorphism in object-oriented programming.
Compile-time Polymorphism: The use of templates of Matrix in std::vector and std::shared_ptr is an example of compile-time polymorphism, as these templates are resolved at compile-time to work with specific types (Matrix, concretized by SparseMatrix, ToeplitzMatrix).

2) How the Compiler Executes Overridden Functions:

The C++ compiler uses a mechanism known as the virtual table (vtable) to support run-time polymorphism. Each class that uses virtual functions has its own vtable. This table is a compile-time construct that holds addresses of the virtual functions for instances of the class. When you call a virtual function (like repr()) on an object, the compiler looks up the function address in the vtable of that object's class, and the function call is resolved to the appropriate function at run time. This is how the overridden repr() function in SparseMatrix or ToeplitzMatrix is executed instead of the one in Matrix, even when using a pointer to the base class.


3) Difference Between Raw Pointer and std::shared_ptr:

A raw pointer is a simple pointer to a memory address, without any additional features. It doesn't manage the lifecycle of the object it points to (i.e., it does not automatically delete the object when it's no longer needed).
std::shared_ptr, on the other hand, is a smart pointer that manages the memory of the object it points to. It keeps track of how many shared_ptrs point to the same object and automatically deletes the object when the last shared_ptr that points to it is destroyed or reassigned. This prevents memory leaks and dangling pointers.

Modifications for Raw Pointers:
If the code were to use raw pointers instead of std::shared_ptr, you would need to manually manage the memory allocated for SparseMatrix and ToeplitzMatrix instances. This would include allocating memory with new when creating the instances and deallocating it with delete when it's no longer needed.

Problem 4:
