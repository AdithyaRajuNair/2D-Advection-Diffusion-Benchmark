#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <cmath>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

// CUDA forward declarations
__global__ void advectionDiffusionKernel(double* U_new, const double* U_old, int N, double D, const double* Vx, const double* Vy, double DX, double DY, double DT);
void runCUDABenchmark(int N, int num_steps, double D, std::vector<double>& Vx, std::vector<double>& Vy, double DX, double DY, double DT, std::vector<double>& results);

// Constants for the simulation
const double D_COEFF = 0.5; // Diffusion coefficient
std::vector<double> VX;      // Advection velocity in x
std::vector<double> VY;      // Advection velocity in y

const double DX = 1.0;      // Grid spacing in x
const double DY = 1.0;      // Grid spacing in y
const int NUM_STEPS = 2000; // Number of time steps

// Uncomment to save velocity fields for visualization
// void saveVelocity(const std::vector<double>& u, const std::vector<double>& v, int N, int step) {

//     std::ofstream file("velocity_step_" + std::to_string(step) + ".csv");

//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < N; ++j) {
//             int idx = i * N + j;
//             file << u[idx] << "," << v[idx];
//             if (j < N - 1) file << ",";
//         }
//         file << "\n";
//     }

//     file.close();
// }

void mountainPassWind(int N, std::vector<double>& u, std::vector<double>& v) {
    u.assign(N * N, 0.0);
    v.assign(N * N, 0.0);

    int center = N / 2;
    double pass_half_width = N / 15.0;
    double base_speed = 0.2;
    double max_speed = 1.5;   // slightly reduced for stability

    for (int i = 0; i < N; ++i) {
        double dist = std::abs(i - center);

        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;

            
            // Smooth mountain mask
            double mountain = 1.0 / (1.0 + std::exp(-(dist - pass_half_width)));
            double pass_factor = 1.0 - mountain;

            
            // Jet acceleration in pass
            double jet_profile = std::exp(-(dist * dist) / (2.0 * pass_half_width * pass_half_width));

            double u_val = base_speed * (1.0 - pass_factor) + max_speed * pass_factor * jet_profile;

            
            // Base vertical deflection
            double offset = (i - center) / (double)N;
            double v_val = -0.4 * offset * u_val;

            
            // Post-pass expansion
            if (j > N / 2) {
                double spread = (i - center) / (double)N;

                // outward expansion
                v_val += 1.0 * spread * u_val;

                // velocity decay (jet weakens downstream)
                double decay = std::exp(-(j - N/2) / 60.0);
                u_val *= decay;

                // mild turbulence (reduced for stability)
                v_val += 0.1 * std::sin(0.2 * i) * std::sin(0.1 * j);
            }

            
            // Velocity normalization
            double mag = std::sqrt(u_val * u_val + v_val * v_val);
            double max_allowed = 1.5;

            if (mag > max_allowed) {
                u_val *= max_allowed / mag;
                v_val *= max_allowed / mag;
            }

            u[idx] = u_val;
            v[idx] = v_val;
        }
    }
}

// Uncomment to save grid for visualization
// void saveGrid(const std::vector<double>& U, int N, int step) {
//     std::ofstream file("output_step_" + std::to_string(step) + ".csv");

//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < N; ++j) {
//             file << U[i * N + j];
//             if (j < N - 1) file << ",";
//         }
//         file << "\n";
//     }
// }

// Function to initialize the grid with a heat pulse
void initializeHeatPulse(std::vector<double>& U, int N) {
    int center_x = N / 2; 
    int center_y = N / 4;
    double pulse_radius_sq = (N / 10.0) * (N / 10.0); // Radius of the heat pulse

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double dist_sq = (i - center_x) * (i - center_x) + (j - center_y) * (j - center_y);
            if (dist_sq < pulse_radius_sq) {
                U[i * N + j] = 100.0 * (1.0 - dist_sq / pulse_radius_sq); // Gaussian-like pulse
            } else {
                U[i * N + j] = 0.0;
            }
        }
    }
}

// Serial Implementation
void runSerialBenchmark(int N, int num_steps, double D, std::vector<double>& Vx, std::vector<double>& Vy, double DX, double DY, double DT, std::vector<double>& results) {
    std::vector<double> U_old(N * N);
    std::vector<double> U_new(N * N);

    initializeHeatPulse(U_old, N);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int idx = i * N + j;

                // Diffusion terms
                double diff_x = (U_old[idx + 1] - 2 * U_old[idx] + U_old[idx - 1]) / (DX * DX);
                double diff_y = (U_old[idx + N] - 2 * U_old[idx] + U_old[idx - N]) / (DY * DY);

                // Advection terms (central difference)
                double adv_x = (U_old[idx + 1] - U_old[idx - 1]) / (2 * DX);
                double adv_y = (U_old[idx + N] - U_old[idx - N]) / (2 * DY);

                U_new[idx] = U_old[idx] + DT * (D * (diff_x + diff_y) - (Vx[idx] * adv_x + Vy[idx] * adv_y));
            }
        }

        // Apply boundary conditions (Dirichlet: zero)
        for (int i = 0; i < N; ++i) {
            U_new[i] = 0.0;                 // top row
            U_new[(N - 1) * N + i] = 0.0;   // bottom row
            U_new[i * N] = 0.0;             // left column
            U_new[i * N + (N - 1)] = 0.0;   // right column
        }

        std::swap(U_old, U_new);

        // Save output
        // if (step % 100 == 0) {
        //     saveGrid(U_old, N, step);
        //     saveVelocity(Vx, Vy, N, step);
        // }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    results.push_back(elapsed_time.count() / num_steps);
}

// OpenMP Implementation
void runOpenMPBenchmark(int N, int num_steps, double D, std::vector<double>& Vx, std::vector<double>& Vy, double DX, double DY, double DT, std::vector<double>& results) {
    std::vector<double> U_old(N * N);
    std::vector<double> U_new(N * N);

    initializeHeatPulse(U_old, N);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int idx = i * N + j;

                // Diffusion terms
                double diff_x = (U_old[idx + 1] - 2 * U_old[idx] + U_old[idx - 1]) / (DX * DX);
                double diff_y = (U_old[idx + N] - 2 * U_old[idx] + U_old[idx - N]) / (DY * DY);

                // Advection terms (central difference)
                double adv_x = (U_old[idx + 1] - U_old[idx - 1]) / (2 * DX);
                double adv_y = (U_old[idx + N] - U_old[idx - N]) / (2 * DY);

                U_new[idx] = U_old[idx] + DT * (D * (diff_x + diff_y) - (Vx[idx] * adv_x + Vy[idx] * adv_y));
            }
        }
        
                // Apply boundary conditions (Dirichlet: zero)
        for (int i = 0; i < N; ++i) {
            U_new[i] = 0.0;                 // top row
            U_new[(N - 1) * N + i] = 0.0;   // bottom row
            U_new[i * N] = 0.0;             // left column
            U_new[i * N + (N - 1)] = 0.0;   // right column
        }

        std::swap(U_old, U_new);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    results.push_back(elapsed_time.count() / num_steps);
}

int main() {
    std::cout << "Size,Mode,TimePerStep" << std::endl;

    std::vector<int> grid_sizes = {64, 128, 256, 512, 1024}; // Example grid sizes

    for (int N : grid_sizes) {
        // Calculate DT based on stability criteria for diffusion
        // DT <= DX*DX / (4*D)
        // For advection, CFL condition: DT <= DX / max(|Vx|, |Vy|)
        // We'll use a conservative DT
        
        mountainPassWind(N, VX, VY);

        double max_vel = 0.0;
        for (int i = 0; i < N*N; i++) {
            double mag = std::sqrt(VX[i]*VX[i] + VY[i]*VY[i]);
            if (mag > max_vel) max_vel = mag;
        }

        double dt_adv = DX / (max_vel + 1e-6);
        double dt_diff = DX * DX / (4.0 * D_COEFF);

        double DT = 0.4 * std::min(dt_adv, dt_diff);

        std::vector<double> serial_times;
        runSerialBenchmark(N, NUM_STEPS, D_COEFF, VX, VY, DX, DY, DT, serial_times);
        std::cout << N << ",Serial," << std::fixed << std::setprecision(10) << serial_times[0] << std::endl;

        std::vector<double> openmp_times;
        runOpenMPBenchmark(N, NUM_STEPS, D_COEFF, VX, VY, DX, DY, DT, openmp_times);
        std::cout << N << ",OpenMP," << std::fixed << std::setprecision(10) << openmp_times[0] << std::endl;

        std::vector<double> cuda_times;
        runCUDABenchmark(N, NUM_STEPS, D_COEFF, VX, VY, DX, DY, DT, cuda_times);
        std::cout << N << ",CUDA," << std::fixed << std::setprecision(10) << cuda_times[0] << std::endl;
    }

    return 0;
}

// CUDA Kernel Implementation
__global__ void advectionDiffusionKernel(double* U_new, const double* U_old, int N, double D, const double* Vx, const double* Vy, double DX, double DY, double DT) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Ensure threads stay within the computational domain (excluding boundaries)
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        int idx = i * N + j;

        // Diffusion terms
        double diff_x = (U_old[idx + 1] - 2 * U_old[idx] + U_old[idx - 1]) / (DX * DX);
        double diff_y = (U_old[idx + N] - 2 * U_old[idx] + U_old[idx - N]) / (DY * DY);

        // Advection terms (central difference)
        // Note: These accesses are not perfectly coalesced in the Y direction,
        // but for typical N, the X direction (j) will dominate coalescing benefits.
        double adv_x = (U_old[idx + 1] - U_old[idx - 1]) / (2 * DX);
        double adv_y = (U_old[idx + N] - U_old[idx - N]) / (2 * DY);

        U_new[idx] = U_old[idx] + DT * (D * (diff_x + diff_y) - (Vx[idx] * adv_x + Vy[idx] * adv_y));
    }
}

// CUDA Benchmark Wrapper
void runCUDABenchmark(int N, int num_steps, double D, std::vector<double>& Vx, std::vector<double>& Vy, double DX, double DY, double DT, std::vector<double>& results) {
    double *h_U_old, *h_U_new; // Host pointers
    double *d_U_old, *d_U_new; // Device pointers

    size_t mem_size = N * N * sizeof(double);

    // Allocate host memory
    h_U_old = (double*)malloc(mem_size);
    h_U_new = (double*)malloc(mem_size);

    // Initialize host data
    std::vector<double> initial_U(N * N);
    initializeHeatPulse(initial_U, N);
    memcpy(h_U_old, initial_U.data(), mem_size);

    // Allocate device memory
    cudaMalloc((void**)&d_U_old, mem_size);
    cudaMalloc((void**)&d_U_new, mem_size);

    // Copy initial data from host to device
    cudaMemcpy(d_U_old, h_U_old, mem_size, cudaMemcpyHostToDevice);

    double *d_Vx, *d_Vy;
    cudaMalloc(&d_Vx, mem_size);
    cudaMalloc(&d_Vy, mem_size);

    cudaMemcpy(d_Vx, Vx.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vy, Vy.data(), mem_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    // For coalesced access, map threads to columns (j) primarily.
    // A block size of 32x32 is common.
    int block_size_x = 32;
    int block_size_y = 32;
    dim3 blockDim(block_size_x, block_size_y);
    dim3 gridDim((N + block_size_x - 1) / block_size_x, (N + block_size_y - 1) / block_size_y);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);

    for (int step = 0; step < num_steps; ++step) {
        advectionDiffusionKernel<<<gridDim, blockDim>>>(d_U_new, d_U_old, N, D, d_Vx, d_Vy, DX, DY, DT);
        // Swap device pointers for next iteration
        double* temp = d_U_old;
        d_U_old = d_U_new;
        d_U_new = temp;
    }

    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    results.push_back((double)(milliseconds / 1000.0) / num_steps); // Convert ms to seconds and then per step

    // Free device memory
    cudaFree(d_U_old);
    cudaFree(d_U_new);
    cudaFree(d_Vx);
    cudaFree(d_Vy);

    // Free host memory
    free(h_U_old);
    free(h_U_new);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}
