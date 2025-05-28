#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include "cuda_runtime.h"

using namespace std;


void cudaCheck(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        cerr << "CUDA error during " << context << ": "
            << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}


__global__ void computeMatrixProduct(const int* matA, const int* matB, int* matResult, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dim && col < dim) {
        int partial = 0;
        for (int k = 0; k < dim; ++k) {
            partial += matA[row * dim + k] * matB[k * dim + col];
        }
        matResult[row * dim + col] = partial;
    }
}


int randInRange(int low, int high) {
    static mt19937 rng(random_device{}());
    uniform_int_distribution<int> dist(low, high);
    return dist(rng);
}


vector<vector<int>> createRandMatrix(int size, int min = 1, int max = 100) {
    vector<vector<int>> matrix(size, vector<int>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            matrix[i][j] = randInRange(min, max);
    return matrix;
}


void storeMatrix(const vector<vector<int>>& matrix, const string& filename) {
    ofstream outfile(filename);
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            outfile << row[i];
            if (i < row.size() - 1) outfile << ",";
        }
        outfile << "\n";
    }
}


float measureKernelExecutionTime(const int* d_A, const int* d_B, int* d_Result, int n) {
    dim3 blockSize(8, 8);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
        (n + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start), "event create start");
    cudaCheck(cudaEventCreate(&stop), "event create stop");

    cudaCheck(cudaEventRecord(start, 0), "event record start");
    computeMatrixProduct << <gridSize, blockSize >> > (d_A, d_B, d_Result, n);
    cudaCheck(cudaGetLastError(), "kernel execution");
    cudaCheck(cudaEventRecord(stop, 0), "event record stop");

    cudaCheck(cudaEventSynchronize(stop), "event synchronize stop");

    float milliseconds = 0;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed time");

    cudaCheck(cudaEventDestroy(start), "event destroy start");
    cudaCheck(cudaEventDestroy(stop), "event destroy stop");

    return milliseconds;
}


float gpuMatrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result) {
    int n = A.size();

    vector<int> flatA(n * n), flatB(n * n), flatResult(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            flatA[i * n + j] = A[i][j];
            flatB[i * n + j] = B[i][j];
        }

    int* d_A, * d_B, * d_Result;
    size_t bytes = n * n * sizeof(int);

    cudaCheck(cudaMalloc(&d_A, bytes), "malloc A");
    cudaCheck(cudaMalloc(&d_B, bytes), "malloc B");
    cudaCheck(cudaMalloc(&d_Result, bytes), "malloc Result");

    cudaCheck(cudaMemcpy(d_A, flatA.data(), bytes, cudaMemcpyHostToDevice), "copy A");
    cudaCheck(cudaMemcpy(d_B, flatB.data(), bytes, cudaMemcpyHostToDevice), "copy B");

    float milliseconds = measureKernelExecutionTime(d_A, d_B, d_Result, n);

    cudaCheck(cudaMemcpy(flatResult.data(), d_Result, bytes, cudaMemcpyDeviceToHost), "copy Result");

    cudaCheck(cudaFree(d_A), "free A");
    cudaCheck(cudaFree(d_B), "free B");
    cudaCheck(cudaFree(d_Result), "free Result");

    result.resize(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result[i][j] = flatResult[i * n + j];

    return milliseconds;
}


int main() {
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount == 0) {
        cerr << "CUDA-capable devices not found" << endl;
        return 1;
    }
    cout << "Detected CUDA devices: " << deviceCount << endl;

    const string outputDir = "C:/Users/Sofia/Desktop/CudaRuntime1/CudaRuntime1.sln/results";
    ofstream perfData("gpu_performance.txt");

    for (int dim = 250; dim <= 3000; dim += 250) {
        cout << "Matrix dimension: " << dim << "x" << dim << endl;

        auto matA = createRandMatrix(dim);
        auto matB = createRandMatrix(dim);

        string fileA = outputDir + "/matrix_" + to_string(dim) + "_A.csv";
        string fileB = outputDir + "/matrix_" + to_string(dim) + "_B.csv";
        storeMatrix(matA, fileA);
        storeMatrix(matB, fileB);

        vector<vector<int>> product;
        float kernelTime = gpuMatrixMultiply(matA, matB, product);

        string resultFile = outputDir + "/gpu_result_" + to_string(dim) + ".csv";
        storeMatrix(product, resultFile);

        perfData << dim << "\t" << fixed << setprecision(3) << kernelTime << endl;
        cout << "  Kernel time: " << fixed << setprecision(3) << kernelTime << " ms" << endl;
    }

    perfData.close();
    cout << "Performance data saved to gpu_performance.txt" << endl;
    return 0;
}
