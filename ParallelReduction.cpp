#include <iostream>
#include <omp.h>
#include <limits>
#include <cstdlib>  // For generating random numbers
#include <chrono>   // For measuring execution time

using namespace std;
using namespace std::chrono;

// Sequential Min, Max, Sum, Average
void sequentialReduction(int *arr, int size, int &sum, int &minVal, int &maxVal, double &avg) {
    sum = 0;
    minVal = numeric_limits<int>::max();
    maxVal = numeric_limits<int>::min();

    for (int i = 0; i < size; i++) {
        sum += arr[i];
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
    }

    avg = static_cast<double>(sum) / size;
}

// Parallel Min, Max, Sum, Average using OpenMP
void parallelReduction(int *arr, int size, int &sum, int &minVal, int &maxVal, double &avg) {
    sum = 0;
    minVal = numeric_limits<int>::max();
    maxVal = numeric_limits<int>::min();

    #pragma omp parallel for reduction(+:sum) reduction(min:minVal) reduction(max:maxVal)
    for (int i = 0; i < size; i++) {
        sum += arr[i];
        minVal = min(minVal, arr[i]);
        maxVal = max(maxVal, arr[i]);
    }

    avg = static_cast<double>(sum) / size;
}

int main() {
    int size = 100000000; // 100 million elements
    int *arr = new int[size];

    // Generate large dataset with random numbers
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 1000;  // Random values between 0-999
    }

    int sumSeq, minSeq, maxSeq, sumPar, minPar, maxPar;
    double avgSeq, avgPar;

    // Measure sequential execution time
    auto startSeq = high_resolution_clock::now();
    sequentialReduction(arr, size, sumSeq, minSeq, maxSeq, avgSeq);
    auto stopSeq = high_resolution_clock::now();
    auto durationSeq = duration_cast<milliseconds>(stopSeq - startSeq);

    // Measure parallel execution time
    auto startPar = high_resolution_clock::now();
    parallelReduction(arr, size, sumPar, minPar, maxPar, avgPar);
    auto stopPar = high_resolution_clock::now();
    auto durationPar = duration_cast<milliseconds>(stopPar - startPar);

    // Display results
    cout << "Sequential:  Sum = " << sumSeq << ", Min = " << minSeq << ", Max = " << maxSeq << ", Avg = " << avgSeq << endl;
    cout << "Parallel:    Sum = " << sumPar << ", Min = " << minPar << ", Max = " << maxPar << ", Avg = " << avgPar << endl;
    
    cout << "\nTime Taken:" << endl;
    cout << "Sequential Execution: " << durationSeq.count() << " ms" << endl;
    cout << "Parallel Execution:   " << durationPar.count() << " ms" << endl;
    cout << "Speedup: " << (double)durationSeq.count() / durationPar.count() << "x" << endl;

    // Clean up memory
    delete[] arr;

    return 0;
}