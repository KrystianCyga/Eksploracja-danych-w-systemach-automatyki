import numpy as np
import matplotlib.pyplot as plt

def generate_periodic_signal(frequency=5, sampling_rate=100, duration=10):
    t = np.linspace(0, duration, sampling_rate * duration)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

def generate_random_signal(N=1000):
    np.random.seed(0)  # Dla powtarzalności
    signal = np.random.randn(N)
    t = np.arange(N)
    return t, signal

def generate_lorenz_system(initial_state=(1.0, 1.0, 1.0), sigma=10.0, rho=28.0, beta=8/3, dt=0.01, N=10000):
    xs = np.empty(N)
    ys = np.empty(N)
    zs = np.empty(N)
    x, y, z = initial_state
    for i in range(N):
        xs[i] = x
        ys[i] = y
        zs[i] = z
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
    return np.arange(N)*dt, xs  # Można również zwrócić y i z, jeśli potrzebne

def generate_henon(length, a=1.4, b=0.3):
    x = np.zeros(length)
    y = np.zeros(length)
    x[0], y[0] = 0.0, 0.0
    for i in range(1, length):
        x[i] = 1 - a * x[i-1]**2 + y[i-1]
        y[i] = b * x[i-1]
    return x, y

def compute_mutual_information(x, delay, bins=20):
    if delay == 0:
        x1 = x.copy()
        x2 = x.copy()
    else:
        x1 = x[:-delay]
        x2 = x[delay:]
    if len(x1) < 2 or len(x2) < 2:
        return 0.0

    joint_counts, x_edges, y_edges = np.histogram2d(x1, x2, bins=bins)
    total = np.sum(joint_counts)
    if total == 0:
        return 0.0

    joint_prob = joint_counts / total
    p_x1 = np.sum(joint_prob, axis=1) + 1e-12
    p_x2 = np.sum(joint_prob, axis=0) + 1e-12

    valid = joint_prob > 0
    ratio = np.divide(joint_prob, np.outer(p_x1, p_x2), where=valid)
    term = np.where(valid, joint_prob * np.log2(ratio), 0)
    return np.sum(term)

def calculate_Td(signal, T_max=100, bins=20):
    signal_normalized = (signal - np.mean(signal)) / np.std(signal)
    N = len(signal_normalized)
    T_max = min(T_max, N - 2)
    MI_values = []

    for T in range(0, T_max + 1):
        mi = compute_mutual_information(signal_normalized, T, bins)
        MI_values.append(mi)

    MI_series = np.array(MI_values)
    minima = []
    for i in range(1, len(MI_series) - 1):
        if MI_series[i] < MI_series[i-1] and MI_series[i] < MI_series[i+1]:
            minima.append(i)

    if minima:
        Td = minima[0]
    else:
        target = 0.8 * MI_series[0]
        for i, mi in enumerate(MI_series[1:], 1):
            if mi <= target:
                Td = i
                break
        else:
            Td = 1

    return Td, MI_series

def plot_MI(Td, MI_series, signal_name):
    T = np.arange(len(MI_series))
    plt.figure(figsize=(10, 6))
    plt.plot(T, MI_series, label='MI(T)')
    plt.axvline(x=Td, color='r', linestyle='--', label=f'Td = {Td}')
    plt.title(f'Średnia Wzajemna Informacja dla {signal_name}')
    plt.xlabel('Opóźnienie T (probek)')
    plt.ylabel('MI(T)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generowanie sygnałów
signals = [
    ("Sygnał Okresowy", generate_periodic_signal()[1]),
    ("Sygnał Losowy", generate_random_signal()[1]),
    ("System Lorenza", generate_lorenz_system(N=10000)[1]),
    ("Mapa Hénona X", generate_henon(1000)[0]),
    ("Mapa Hénona Y", generate_henon(1000)[1])
]

# Obliczanie Td i rysowanie wykresów MI
for name, signal in signals:
    Td, MI_series = calculate_Td(signal)
    print(f"{name}: Td = {Td}")
    plot_MI(Td, MI_series, name)