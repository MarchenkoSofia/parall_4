import os
import matplotlib.pyplot as plt


def parse_gpu_performance_file(filepath):
    sizes = []
    times = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line and 'ms' in line:
                try:
                    size_str, time_str = line.split(':')
                    size = int(size_str.strip())
                    time = float(time_str.strip().replace('ms', '').strip())
                    sizes.append(size)
                    times.append(time)
                except ValueError:
                    continue

    return sizes, times


def plot_all_files(folder):
    plt.figure(figsize=(10, 6))
    files_plotted = 0

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            sizes, times = parse_gpu_performance_file(filepath)
            if sizes and times:
                label = os.path.splitext(filename)[0]
                plt.plot(sizes, times, marker='o', label=label)
                files_plotted += 1

    if files_plotted == 0:
        print("Нет подходящих данных для построения графика.")
        return

    plt.xlabel("Размер матрицы")
    plt.ylabel("Время выполнения (мс)")
    plt.title("Сравнение производительности")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("With_1_threads.png")
    plt.show()


def main():
    folder = "results"
    plot_all_files(folder)


if __name__ == "__main__":
    main()
