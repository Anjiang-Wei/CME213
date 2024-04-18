#define ARMA_ALLOW_FAKE_GCC
#include <algorithm>
#include <armadillo>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <memory>

#include "gpu_func.h"

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                          DeviceAllocator 						//
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

DeviceAllocator::DeviceAllocator(nn_real *cpu_data, int n)
{
  // TODO: implement this constructor
  this->nbytes = n * sizeof(nn_real);
  // Allocate memory on the GPU and copy the CPU data to the GPU.
  checkCudaErrors(cudaMalloc(&this->data, this->nbytes));
  checkCudaErrors(cudaMemcpy(this->data, cpu_data, this->nbytes, cudaMemcpyHostToDevice));
}

DeviceAllocator::DeviceAllocator(int n)
{
  // TODO: implement this constructor
  // Only allocate memory on the GPU.
  checkCudaErrors(cudaMalloc(&this->data, n * sizeof(nn_real)));
  this->nbytes = n * sizeof(nn_real);
}

DeviceAllocator::~DeviceAllocator()
{
  // TODO: implement this destructor
  // Deallocate the memory on the GPU.
  checkCudaErrors(cudaFree(this->data));
}

void DeviceAllocator::to_cpu(nn_real *cpu_data)
{
  // TODO: implement this function
  // Copy the GPU data to the CPU pointer.
  checkCudaErrors(cudaMemcpy(cpu_data, this->data, this->nbytes, cudaMemcpyDeviceToHost));
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                          DeviceMatrix 							//
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

DeviceMatrix::DeviceMatrix(int n_rows, int n_cols)
{
  // TODO: implement this constructor
  this->n_rows = n_rows;
  this->n_cols = n_cols;
  this->allocator = std::make_unique<DeviceAllocator>(n_rows * n_cols);
  this->data = this->allocator->data;
}

DeviceMatrix::DeviceMatrix(arma::Mat<nn_real> &cpu_mat)
{
  // TODO: implement this constructor
  this->n_rows = cpu_mat.n_rows;
  this->n_cols = cpu_mat.n_cols;
  this->allocator = std::make_unique<DeviceAllocator>(cpu_mat.memptr(), cpu_mat.n_elem);
  this->data = this->allocator->data;
}

void DeviceMatrix::to_cpu(arma::Mat<nn_real> &cpu_mat)
{
  this->allocator->to_cpu(cpu_mat.memptr());
}

__device__ nn_real &DeviceMatrix::operator()(int row, int col)
{
  // Note that arma matrices are column-major
  return data[col * this->n_rows + row];
}

int DeviceMatrix::total_bytes()
{
  return allocator->nbytes;
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                           CUDA kernels                           //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

/**
 * A CUDA kernel function that applies the sigmoid function element-wise to a
 * matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
__global__ void MatSigmoid(DeviceMatrix src, DeviceMatrix dst)
{
  // TODO: implement this kernel function
  // Hint: Use Exp() from common.h
  // Sigmod = 1 / (1 + exp(-x))
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
          i < src.n_rows;
          i += blockDim.x * gridDim.x) {
            for (int j = blockIdx.y * blockDim.y + threadIdx.y;
                    j < src.n_cols;
                    j += blockDim.y * gridDim.y) {
                      dst(i, j) = 1 / (1 + Exp(-src(i, j)));
                    }
          }
  // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   printf("dst(0, 0) = %f, dst(0, 1) = %f\n", dst(0, 0), dst(0, 1));
  // }
}

/**
 * A CUDA kernel function that repeats each column of the source matrix `repeat`
 * times and stores the result in the destination matrix.
 *
 * @param src The source matrix to repeat.
 * @param dst The destination matrix to store the repeated columns.
 * @param repeat The number of times to repeat each column.
 */
__global__ void MatRepeatColVec(DeviceMatrix src, DeviceMatrix dst,
                                int repeat)
{
  // TODO: implement this kernel function
}

/**
 * A CUDA kernel function that computes the sum of a matrix along a specified
 * axis on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param alpha The scaling factor for the sum.
 * @param axis The axis along which to compute the sum (0 for rows, 1 for
 * columns).
 */
__global__ void MatSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha,
                       int axis)
{
  // TODO: implement this kernel function
}

/**
 * A CUDA kernel function that applies the softmax function along a specified
 * axis to a matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param axis The axis along which to apply the softmax function (0 for rows, 1
 * for columns).
 */
__global__ void MatSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis)
{
  /**
   * TODO: implement this kernel function
   * Hint: Use Exp() from common.h
   * A possible implementation is to have one thread per row (or  column,
   * depending on axis), compute the sum of exponentials of all elements in
   * the row by iterating through elements in the row, and then replace
   * dst(row, col) with the exponential of src(row, col) divided by the sum.
   */
}

/**
 * A CUDA kernel function that computes the cross-entropy loss between predicted
 * and true labels on the GPU.
 *
 * @param y_pred The predicted label matrix.
 * @param y The true label matrix.
 * @param loss The output loss matrix.
 */
__global__ void MatCrossEntropyLoss(DeviceMatrix y_pred, DeviceMatrix y,
                                    DeviceMatrix loss)
{
  /**
   * TODO: implement this kernel function
   * Hint: This kernel computes loss = -y * log(y_pred) where * denotes
   * element-wise multiplication and log is applied element-wise. Use
   * Log() from common.h
   */
}

/**
 * A CUDA kernel function that performs element-wise arithmetic operations on
 * two matrices on the GPU. A = alpha * (A + beta * B)
 *
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param alpha The scaling factor for the first input matrix.
 * @param beta The scaling factor for the second input matrix.
 */
__global__ void MatElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha,
                             nn_real beta)
{
  // TODO: implement this kernel function
}

/**
 * A CUDA kernel function that computes the element-wise square of a matrix on
 * the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
__global__ void MatSquare(DeviceMatrix src, DeviceMatrix dst)
{
  // TODO: implement this kernel function
}

/**
 * A CUDA kernel function that computes backpropagation for sigmoid function on
 * the GPU.
 *
 * @param da1 The upstream derivative matrix.
 * @param a1 The activation matrix.
 * @param dz1 The output derivative matrix.
 */
__global__ void MatSigmoidBackProp(DeviceMatrix da1, DeviceMatrix a1,
                                   DeviceMatrix dz1)
{
  /**
   * TODO: implement this kernel function
   * Hint: This kernel computes dz1 = da1 * a1 * (1 - a1), where * denotes
   * element-wise multiplication.
   */
}

__global__ void Warmup() {}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                       GPU kernel wrappers                        //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

void DSigmoid(DeviceMatrix src, DeviceMatrix dst)
{
  // TODO: implement this function
  dim3 block = {32, 32};
  dim3 grid = {src.n_rows + block.x - 1 / block.x, src.n_cols + block.y - 1 / block.y};
  MatSigmoid<<<grid, block>>>(src, dst);
  CHECK_LAUNCH("DSigmoid");
}

void DRepeatColVec(DeviceMatrix src, DeviceMatrix dst, int repeat)
{
  // TODO: implement this function

  CHECK_LAUNCH("DRepeatColVec");
}

void DSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha, int axis)
{
  // TODO: implement this function

  CHECK_LAUNCH("DSum");
}

void DSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis)
{
  // TODO: implement this function

  CHECK_LAUNCH("DSoftmax");
}

void DCELoss(DeviceMatrix y_pred, DeviceMatrix y, DeviceMatrix loss)
{
  /**
   * TODO: implement this function
   * Hint: Initialize a temporary matrix T to store the loss and then call
   * MatCrossEntropyLoss. Call DSum twice to compute the sum of all elements
   * in T.
   */
  CHECK_LAUNCH("DCELoss");
}

void DElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha, nn_real beta)
{
  // TODO: implement this function

  CHECK_LAUNCH("DElemArith");
}

void DSquare(DeviceMatrix src, DeviceMatrix dst)
{
  // TODO: implement this function

  CHECK_LAUNCH("DSquare");
}

void DSigmoidBackprop(DeviceMatrix da1, DeviceMatrix a1, DeviceMatrix dz1)
{
  // TODO: implement this function

  CHECK_LAUNCH("DSigmoidBackprop");
}

void DWarmup() { Warmup<<<1, 1>>>(); }
