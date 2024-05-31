#ifndef PARALLEL_RADIX_SORT
#define PARALLEL_RADIX_SORT

#include <algorithm>
#include <vector>
#include "test_util.h"


constexpr const uint kSizeTestVector = 4000000;
constexpr const uint kNumBits = 16; // must be a divider of 32 for this program to work
constexpr const uint kRandMax = 1 << 31;


/* Function: computeBlockHistograms
 * --------------------------------
 * Splits keys into numBlocks and computes an histogram with numBuckets buckets
 * Remember that numBuckets and numBits are related; same for blockSize and numBlocks.
 * Should work in parallel.
 */
std::vector<uint> computeBlockHistograms(
    const std::vector<uint> &keys,
    uint numBlocks, 
    uint numBuckets,
    uint numBits, 
    uint startBit, 
    uint blockSize
) {
    std::vector<uint> blockHistograms(numBlocks * numBuckets, 0);
    // TODO
#pragma omp parallel for
    for (uint block = 0; block < numBlocks; ++block) {
        std::vector<uint> localHistogram(numBuckets, 0);
        uint startIdx = block * blockSize;
        uint endIdx = std::min(startIdx + blockSize, static_cast<uint>(keys.size()));

        for (uint i = startIdx; i < endIdx; ++i) {
            uint key = (keys[i] >> startBit) & (numBuckets - 1);
            localHistogram[key]++;
        }

        // Copy local histogram to the global block histogram
        for (uint j = 0; j < numBuckets; ++j) {
            blockHistograms[block * numBuckets + j] = localHistogram[j];
        }
    }
    return blockHistograms;
}

/* Function: reduceLocalHistoToGlobal
 * ----------------------------------
 * Takes as input the local histogram of size numBuckets * numBlocks and "merges"
 * them into a global histogram of size numBuckets.
 */
std::vector<uint> reduceLocalHistoToGlobal(
    const std::vector<uint> &blockHistograms,
    uint numBlocks, 
    uint numBuckets
) {
    std::vector<uint> globalHisto(numBuckets, 0);
    // TODO
    uint* globalHistoPtr = &globalHisto[0];
#pragma omp parallel for reduction(+:globalHistoPtr[:numBuckets])
    for (uint block = 0; block < numBlocks; ++block) {
        for (uint bucket = 0; bucket < numBuckets; ++bucket) {
            globalHistoPtr[bucket] += blockHistograms[block * numBuckets + bucket];
        }
    }
    return globalHisto;
}

/* Function: scanGlobalHisto
 * -------------------------
 * This function should simply scan the global histogram.
 */
std::vector<uint> scanGlobalHisto(
    const std::vector<uint> &globalHisto,
    uint numBuckets
) {
    std::vector<uint> globalHistoExScan(numBuckets, 0);
    // TODO
    for (uint i = 1; i < numBuckets; ++i) {
        globalHistoExScan[i] = globalHistoExScan[i - 1] + globalHisto[i - 1];
    }
    return globalHistoExScan;
}

/* Function: computeBlockExScanFromGlobalHisto
 * -------------------------------------------
 * Takes as input the globalHistoExScan that contains the global histogram after the scan
 * and the local histogram in blockHistograms. Returns a local histogram that will be used
 * to populate the sorted array.
 */
std::vector<uint> computeBlockExScanFromGlobalHisto(
    uint numBuckets,
    uint numBlocks,
    const std::vector<uint> &globalHistoExScan,
    const std::vector<uint> &blockHistograms
) {
    std::vector<uint> blockExScan(numBuckets * numBlocks, 0);
    // TODO
    // Compute the block-wise exclusive scan
#pragma omp parallel for
    for (uint bucket = 0; bucket < numBuckets; ++bucket) {
        // Start with the global exclusive scan value
        uint prefixSum = globalHistoExScan[bucket];

        for (uint block = 0; block < numBlocks; ++block) {
            blockExScan[block * numBuckets + bucket] = prefixSum;
            prefixSum += blockHistograms[block * numBuckets + bucket];
        }
    }
    return blockExScan;
}

/* Function: populateOutputFromBlockExScan
 * ---------------------------------------
 * Takes as input the blockExScan produced by the splitting of the global histogram
 * into blocks and populates the vector sorted.
 */
void populateOutputFromBlockExScan(
    const std::vector<uint> &blockExScan,
    uint numBlocks, 
    uint numBuckets, 
    uint startBit,
    uint numBits, 
    uint blockSize, 
    const std::vector<uint> &keys,
    std::vector<uint> &sorted
) {
    // TODO
    uint mask = (1 << numBits) - 1;

#pragma omp parallel for
    for (uint block = 0; block < numBlocks; ++block) {
        std::vector<uint> localOffsets(numBuckets, 0);

        // Calculate the starting offset for this block
        for (uint bucket = 0; bucket < numBuckets; ++bucket) {
            localOffsets[bucket] = blockExScan[block * numBuckets + bucket];
        }

        // Populate the sorted array
        uint startIdx = block * blockSize;
        uint endIdx = std::min(startIdx + blockSize, static_cast<uint>(keys.size()));
        for (uint i = startIdx; i < endIdx; ++i) {
            uint key = (keys[i] >> startBit) & mask;
            uint pos = localOffsets[key]++;
            sorted[pos] = keys[i];
        }
    }
}

/* Function: radixSortParallelPass
 * -------------------------------
 * A pass of radixSort on numBits starting after startBit.
 */
void radixSortParallelPass(
    std::vector<uint> &keys, 
    std::vector<uint> &sorted,
    uint numBits, 
    uint startBit,
    uint blockSize
) {
    uint numBuckets = 1 << numBits;

    // Choose numBlocks so that numBlocks * blockSize >= keys.size().
    uint numBlocks = (keys.size() + blockSize - 1) / blockSize;

    // go over each block and compute its local histogram
    std::vector<uint> blockHistograms = computeBlockHistograms(
        keys, numBlocks, numBuckets, numBits, startBit, blockSize);

    // first reduce all the local histograms into a global one
    std::vector<uint> globalHisto = reduceLocalHistoToGlobal(
        blockHistograms, numBlocks, numBuckets);

    // now we scan this global histogram
    std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);

    // now we do a local histogram in each block and add in the global value to get global position
    std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(
        numBuckets, numBlocks, globalHistoExScan, blockHistograms);

    // populate the sorted vector
    populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, startBit,
                                  numBits, blockSize, keys, sorted);
}

int radixSortParallel(
    std::vector<uint> &keys, 
    std::vector<uint> &keys_tmp,
    uint numBits, 
    uint numBlocks
) {
    for (uint startBit = 0; startBit < 32; startBit += 2 * numBits)
    {
        radixSortParallelPass(keys, keys_tmp, numBits, startBit, keys.size() / numBlocks);
        radixSortParallelPass(keys_tmp, keys, numBits, startBit + numBits, keys.size() / numBlocks);
    }

    return 0;
}


void runBenchmark(std::vector<uint>& keys_parallel, std::vector<uint>& temp_keys, uint kNumBits, uint n_blocks, uint n_threads){
    // TODO: Call omp_set_num_threads() with the correct argument

    double startRadixParallel = omp_get_wtime();
    // TODO: Call radixSortParallel() with the correct arguments

    double endRadixParallel = omp_get_wtime();

    printf("%8.3f", endRadixParallel - startRadixParallel);
}
#endif
