#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix_rect.hpp"

TEST(testMatrix2D, sampleTest) {
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
Test your implementation by writing tests that cover most scenarios of 2D matrix
broadcasting. Say you are testing the result C = A * B, test with:
1. A of shape (m != 1, n != 1), B of shape (m != 1, n != 1)
2. A of shape (1, n != 1), B of shape (m != 1, n != 1)
3. A of shape (m != 1, n != 1), B of shape (m != 1, 1)
4. A of shape (1, 1), B of shape (m != 1, n != 1)
Please test any more cases that you can think of.
*/


TEST(testMatrix2D, test1) {
  Matrix2D<int> a(2, 2);
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(1, 0) = 3;
  a(1, 1) = 4;
  Matrix2D<int> b(a);
  Matrix2D<int> c = a.dot(b);
  ASSERT_EQ(c(0,0), 1)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(0,1), 4)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(1,0), 9)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(1,1), 16)
      << "This does not fail, hence this message is not printed.";
}

TEST(testMatrix2D, test2) {
  Matrix2D<int> a(1, 2);
  a(0, 0) = 1;
  a(0, 1) = 2;
  Matrix2D<int> b(2, 2);
  b(0, 0) = 1;
  b(0, 1) = 2;
  b(1, 0) = 3;
  b(1, 1) = 4;
  Matrix2D<int> c = a.dot(b);
  ASSERT_EQ(c(0,0), 1)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(0,1), 4)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(1,0), 3)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(1,1), 8)
      << "This does not fail, hence this message is not printed.";
}

TEST(testMatrix2D, test3) {
  Matrix2D<int> a(2, 2);
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(1, 0) = 3;
  a(1, 1) = 4;
  Matrix2D<int> b(2, 1);
  b(0, 0) = 1;
  b(1, 0) = 2;
  Matrix2D<int> c = a.dot(b);
  ASSERT_EQ(c(0,0), 1)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(0,1), 2)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(1,0), 6)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(1,1), 8)
      << "This does not fail, hence this message is not printed.";
}

TEST(testMatrix2D, test4) {
  Matrix2D<int> a(1, 1);
  a(0, 0) = 2;
  Matrix2D<int> b(2, 2);
  b(0, 0) = 1;
  b(0, 1) = 2;
  b(1, 0) = 3;
  b(1, 1) = 4;
  Matrix2D<int> c = a.dot(b);
  ASSERT_EQ(c(0,0), 2)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(0,1), 4)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(1,0), 6)
      << "This does not fail, hence this message is not printed.";
  ASSERT_EQ(c(1,1), 8)
      << "This does not fail, hence this message is not printed.";
}