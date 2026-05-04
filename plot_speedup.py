import matplotlib.pyplot as plt
import pandas as pd
import sys
import io

def plot_speedup(csv_data):
    df = pd.read_csv(io.StringIO(csv_data))

    # Get unique grid sizes
    grid_sizes = df['Size'].unique()

    plt.figure(figsize=(10, 6))

    for size in grid_sizes:
        size_df = df[df['Size'] == size]
        
        serial_time = size_df[size_df['Mode'] == 'Serial']['TimePerStep'].iloc[0]
        
        if 'OpenMP' in size_df['Mode'].values:
            openmp_time = size_df[size_df['Mode'] == 'OpenMP']['TimePerStep'].iloc[0]
            openmp_speedup = serial_time / openmp_time
            plt.bar(f'{size} OpenMP', openmp_speedup, label=f'OpenMP (Size {size})')
        
        if 'CUDA' in size_df['Mode'].values:
            cuda_time = size_df[size_df['Mode'] == 'CUDA']['TimePerStep'].iloc[0]
            cuda_speedup = serial_time / cuda_time
            plt.bar(f'{size} CUDA', cuda_speedup, label=f'CUDA (Size {size})')

    plt.xlabel('Mode and Grid Size')
    plt.ylabel('Speedup Ratio (vs. Serial)')
    plt.title('Advection-Diffusion Benchmark Speedup')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('speedup_plot.png')
    # plt.show() # Removed to avoid blocking and allow saving to file

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
        with open(csv_file_path, 'r') as f:
            csv_data = f.read()
    else:
        print("Usage: python3 plot_speedup.py <csv_file>")
        sys.exit(1)
    
    plot_speedup(csv_data)
