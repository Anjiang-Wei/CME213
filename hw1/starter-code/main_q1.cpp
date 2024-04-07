#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(testMatrix, sampleTest) {
  ASSERT_EQ(1000, 1000)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(2000, 2000)
      << "This does not fail, hence this message is not printed.";
  // If uncommented, the following line will make this test fail.
  // EXPECT_EQ(2000, 3000) << "This expect statement fails, and this message
  // will be printed.";
}

/*
TODO:

For both the MatrixDiagonal and the MatrixSymmetric classes, do the following:

Write at least the following tests to get full credit here:
1. Declare an empty matrix with the default constructor for MatrixSymmetric.
Assert that the NormL0 and size functions return appropriate values for these.
2. Using the second constructor that takes size as argument, create a matrix of
size zero. Repeat the assertions from (1).
3. Provide a negative argument to the second constructor and assert that the
constructor throws an exception.
4. Create and initialize a matrix of some size, and verify that the NormL0
function returns the correct value.
5. Create a matrix, initialize some or all of its elements, then retrieve and
check that they are what you initialized them to.
6. Create a matrix of some size. Make an out-of-bounds access into it and check
that an exception is thrown.
7. Test the stream operator using std::stringstream and using the "<<" operator.

*/

TEST(testMatrix, default_constructor) {
  MatrixDiagonal<float> m;
  ASSERT_EQ(m.NormL0(), 0)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m.NormL0(), 0)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(m.size(), 0)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m.size(), 0)
      << "This does not fail, hence this message is not printed.";
  MatrixSymmetric<float> m2;
  ASSERT_EQ(m2.NormL0(), 0)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m2.NormL0(), 0)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(m2.size(), 0)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m2.size(), 0)
      << "This does not fail, hence this message is not printed.";
}

TEST(testMatrix, second_constructor) {
  MatrixDiagonal<float> m(0);
  ASSERT_EQ(m.NormL0(), 0)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m.NormL0(), 0)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(m.size(), 0)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m.size(), 0)
      << "This does not fail, hence this message is not printed.";
  MatrixSymmetric<float> m2(0);
  ASSERT_EQ(m2.NormL0(), 0)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m2.NormL0(), 0)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(m2.size(), 0)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m2.size(), 0)
      << "This does not fail, hence this message is not printed.";
}

TEST(testMatrix, exception_second_constructor) {
  ASSERT_ANY_THROW(MatrixDiagonal<float> m(-1)) << "This should throw an exception.";
  ASSERT_ANY_THROW(MatrixSymmetric<float> m2(-1)) << "This should throw an exception.";
}

TEST(testMatrix, norm_l0) {
  MatrixDiagonal<float> m(3);
  m.set(0, 0, 1.5);
  m.set(1, 1, 2.0);
  std::cout << m;
  ASSERT_EQ(m.NormL0(), 2)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m.NormL0(), 2)
      << "This does not fail, hence this message is not printed.";
  MatrixSymmetric<float> m2(3);
  m2.set(0, 0, 1.5);
  m2.set(1, 1, 2.0);
  ASSERT_EQ(m2.NormL0(), 2)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m2.NormL0(), 2)
      << "This does not fail, hence this message is not printed.";
}


TEST(testMatrix, retrieve) {
  MatrixDiagonal<float> m(3);
  m.set(1, 1, 2);
  std::cout << m;
  ASSERT_EQ(m(1, 1), 2)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m(1, 1), 2)
      << "This does not fail, hence this message is not printed.";
  MatrixSymmetric<float> m2(3);
  m2.set(1, 1, 2);
  ASSERT_EQ(m2(1, 1), 2)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(m2(1, 1), 2)
      << "This does not fail, hence this message is not printed.";
}

TEST(testMatrix, exception_out_of_bound_access) {
  MatrixDiagonal<float> m(3);
  ASSERT_ANY_THROW(m(4, 4)) << "This should throw an exception.";
  MatrixSymmetric<float> m2(3);
  ASSERT_ANY_THROW(m2(4, 4)) << "This should throw an exception.";
}


TEST(testMatrix, TestPrint) {
  MatrixSymmetric<int> matrix(3);
  matrix.set(0, 0, 1);
  matrix.set(1, 1, 2);
  matrix.set(1, 2, 3);
  std::stringstream ss;
  matrix.Print(ss); // Redirect output to stringstream
  // Define the expected output
  std::string expected = "1 0 0 \n0 2 3 \n0 3 0 \n";
  
  EXPECT_EQ(ss.str(), expected);
  ASSERT_EQ(ss.str(), expected);

  MatrixDiagonal<int> matrix2(3);
  matrix2.set(0, 0, 1);
  matrix2.set(1, 1, 2);
  matrix2.set(2, 2, 3);
  std::stringstream ss2;
  matrix2.Print(ss2); // Redirect output to stringstream
  // Define the expected output
  std::string expected2 = "1 0 0 \n0 2 0 \n0 0 3 \n";

  EXPECT_EQ(ss2.str(), expected2);
  ASSERT_EQ(ss2.str(), expected2);
}