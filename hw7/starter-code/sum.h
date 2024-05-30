#include <vector>

std::vector<uint> serialSum(const std::vector<uint> &v) {
    std::vector<uint> sums(2);
    // TODO
    sums[0] = 0; // Sum of even elements
    sums[1] = 0; // Sum of odd elements
    for (unsigned int num : v) {
        if (num % 2 == 0) {
            sums[0] += num;
        } else {
            sums[1] += num;
        }
    }
    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint> &v) {
    std::vector<uint> sums(2);
    // TODO
    unsigned int sum_even = 0;
    unsigned int sum_odd = 0;

    // Use OpenMP to parallelize the computation with reduction
#pragma omp parallel for reduction(+ : sum_even, sum_odd)
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] % 2 == 0) {
            sum_even += v[i];
        } else {
            sum_odd += v[i];
        }
    }
    sums[0] = sum_even;
    sums[1] = sum_odd;
    return sums;
}