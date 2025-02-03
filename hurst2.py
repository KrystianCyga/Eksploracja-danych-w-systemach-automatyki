import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D  # Do wizualizacji 3D
from scipy import stats

def mutual_information(time_series, lag, bins=32):
    hist_2d, _, _ = np.histogram2d(time_series[:-lag], time_series[lag:], bins=bins)
    prob_2d = hist_2d / np.sum(hist_2d)
    prob_x = np.sum(prob_2d, axis=1)
    prob_y = np.sum(prob_2d, axis=0)
    nzs = prob_2d > 0
    mi = np.sum(prob_2d[nzs] * np.log(prob_2d[nzs] / (prob_x[:, None] * prob_y[None, :])[nzs]))
    return mi

def find_time_delay(time_series, max_lag=100):
    mi = [mutual_information(time_series, lag) for lag in range(1, max_lag)]
    plt.figure(figsize=(10,5))
    plt.plot(range(1, max_lag), mi, marker='o')
    plt.xlabel('Opóźnienie czasowe (lag)')
    plt.ylabel('Wzajemna Informacja')
    plt.title('Funkcja Wzajemnej Informacji vs Opóźnienie')
    plt.grid(True)
    plt.show()
    T = np.argmin(mi) + 1  # +1 ponieważ indeksowanie zaczyna się od 1
    print(f'Optymalne opóźnienie czasowe (lag): {T}')
    return T

def reconstruct_phase_space(time_series, dE, T):
    N = len(time_series) - (dE - 1) * T
    Y = np.empty((N, dE))
    for i in range(dE):
        Y[:, i] = time_series[i * T : i * T + N]
    return Y

def false_nearest_neighbors(time_series, max_dim, T, threshold=2.0):
    """
    Oblicza procent fałszywych najbliższych sąsiadów dla różnych wymiarów przestrzeni fazowej.
    
    Parametry:
    - time_series: Jednowymiarowa tablica NumPy z danymi czasowymi.
    - max_dim: Maksymalny rozmiar przestrzeni fazowej do analizy.
    - T: Opóźnienie czasowe.
    - threshold: Próg do określenia fałszywych najbliższych sąsiadów.
    
    Zwraca:
    - Lista procentów fałszywych najbliższych sąsiadów dla każdego wymiaru.
    """
    FNN_percent = []
    for dim in range(1, max_dim + 1):
        # Rekonstrukcja przestrzeni fazowej dla aktualnego wymiaru
        Y = reconstruct_phase_space(time_series, dim, T)
        
        if dim < max_dim:
            # Rekonstrukcja przestrzeni fazowej dla wymiaru +1
            Y_next = reconstruct_phase_space(time_series, dim + 1, T)
            
            # Ustalenie minimalnej długości między Y a Y_next
            min_len = min(len(Y), len(Y_next))
            
            # Przycięcie Y i Y_next do minimalnej długości
            Y = Y[:min_len]
            Y_next = Y_next[:min_len]
            
            # Znajdowanie najbliższych sąsiadów w aktualnym wymiarze
            neigh = NearestNeighbors(n_neighbors=2)
            neigh.fit(Y)
            distances, indices = neigh.kneighbors(Y)
            
            # Indeks najbliższego sąsiada (ignorujemy pierwszy sąsiad, którym jest punkt sam siebie)
            idx = indices[:, 1]
            
            # Upewnienie się, że indeksy są w zakresie Y_next
            valid_indices = idx < len(Y_next)
            
            # Obliczanie różnic w następnym wymiarze
            R = np.abs(Y_next[valid_indices, dim] - Y_next[idx[valid_indices], dim])
            
            # Obliczanie fałszywych najbliższych sąsiadów
            FNN = np.sum(R / distances[valid_indices, 1] > threshold)
            FNN_percent.append(FNN / len(Y) * 100)
        else:
            # Dla maksymalnego wymiaru nie obliczamy FNN
            FNN_percent.append(0)
    
    # Wizualizacja procentu FNN względem wymiaru
    plt.figure(figsize=(10,5))
    plt.plot(range(1, max_dim + 1), FNN_percent, marker='o')
    plt.xlabel('Wymiar przestrzeni fazowej')
    plt.ylabel('Procent Fałszywych Najbliższych Sąsiadów (%)')
    plt.title('False Nearest Neighbors')
    plt.grid(True)
    plt.show()
    
    return FNN_percent

def calculate_rs_analysis(time_series, min_n=10, max_n=1000, step=10):
    """
    Oblicza przeskalowany zasięg R/S dla różnych długości podciągów n.
    
    Parametry:
    - time_series: jednowymiarowa tablica NumPy z danymi czasowymi.
    - min_n: minimalna długość podciągu.
    - max_n: maksymalna długość podciągu.
    - step: krok odwoływania długości podciągów.
    
    Zwraca:
    - n_values: lista długości podciągów.
    - rs_values: lista średnich przeskalowanych zasięgów R/S dla każdego n.
    """
    n_values = []
    rs_values = []
    
    for n in range(min_n, max_n + 1, step):
        if len(time_series) < n:
            break
        k = len(time_series) // n  # liczba podciągów
        if k == 0:
            continue
        rs_sum = 0
        for m in range(k):
            start = m * n
            end = start + n
            segment = time_series[start:end]
            Em = np.mean(segment)
            Sm = np.std(segment)
            if Sm == 0:
                continue
            zi = segment - Em
            yi = np.cumsum(zi)
            Rm = np.max(yi) - np.min(yi)
            rs = Rm / Sm
            rs_sum += rs
        rs_avg = rs_sum / k
        n_values.append(n)
        rs_values.append(rs_avg)
    
    return n_values, rs_values

def plot_rs_analysis(n_values, rs_values, system_name):
    """
    Rysuje wykres ln(R/S) vs ln(n) oraz wykonuje regresję liniową.
    
    Parametry:
    - n_values: lista długości podciągów.
    - rs_values: lista średnich przeskalowanych zasięgów R/S.
    - system_name: nazwa systemu dla tytułu wykresu.
    
    Zwraca:
    - H: wykładnik Hursta.
    - intercept: wyraz wolny regresji.
    """
    log_n = np.log(n_values)
    log_rs = np.log(rs_values)
    
    # Regresja liniowa
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)
    H = slope
    
    # Wykres
    plt.figure(figsize=(10,6))
    plt.plot(log_n, log_rs, 'o', label='Dane')
    plt.plot(log_n, intercept + slope * log_n, 'r', label=f'Regresja liniowa: H={H:.4f}')
    plt.xlabel(r'$\ln(n)$')
    plt.ylabel(r'$\ln(R/S)$')
    plt.title(f'Analiza R/S dla {system_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f'Wykładnik Hursta dla {system_name}: H = {H:.4f}')
    return H, intercept

def perform_hurst_analysis(time_series, system_name, min_n=10, max_n=1000, step=10):
    """
    Wykonuje pełną analizę R/S na podanym szeregu czasowym.
    
    Parametry:
    - time_series: jednowymiarowa tablica NumPy z danymi czasowymi.
    - system_name: nazwa systemu do wyświetlenia na wykresie.
    - min_n: minimalna długość podciągu.
    - max_n: maksymalna długość podciągu.
    - step: krok długości podciągów.
    
    Zwraca:
    - H: wykładnik Hursta.
    """
    n_values, rs_values = calculate_rs_analysis(time_series, min_n, max_n, step)
    H, intercept = plot_rs_analysis(n_values, rs_values, system_name)
    return H

def analyze_signal_with_hurst(t, signal, system_name='System', dE=None, T=None):
    """
    Analizuje sygnał, wykonuje rekonstrukcję przestrzeni fazowej oraz analizę R/S w celu wyznaczenia wykładnika Hursta.
    
    Parametry:
    - t: wektor czasu.
    - signal: sygnał czasowy.
    - system_name: nazwa systemu do oznaczenia wykresów.
    - dE: wymiar przestrzeni fazowej (opcjonalnie).
    - T: opóźnienie czasowe (opcjonalnie).
    
    Zwraca:
    - Y: rekonstruowana przestrzeń fazowa.
    - H: wykładnik Hursta.
    """
    print(f'\nAnaliza dla: {system_name}')
    
    # 1. Rekonstrukcja przestrzeni fazowej
    if dE is not None and T is not None:
        print(f'Podany wymiar przestrzeni fazowej: dE = {dE}')
        print(f'Podane opóźnienie czasowe: T = {T}')
        Y = reconstruct_phase_space(signal, dE, T)
    else:
        raise ValueError("Musisz podać zarówno dE, jak i T do rekonstrukcji przestrzeni fazowej.")
    
    # 2. Wizualizacja przestrzeni fazowej
    if dE == 1:
        plt.figure(figsize=(10,5))
        plt.plot(Y, np.zeros_like(Y), marker='.', linestyle='None')
        plt.xlabel('X(t)')
        plt.title(f'Przestrzeń Fazowa dla {system_name} (dE={dE}, T={T})')
        plt.show()
    elif dE == 2:
        plt.figure(figsize=(10,5))
        plt.plot(Y[:,0], Y[:,1], marker='.', linestyle='None')
        plt.xlabel('X(t)')
        plt.ylabel(f'X(t+{T})')
        plt.title(f'Przestrzeń Fazowa dla {system_name} (dE={dE}, T={T})')
        plt.grid(True)
        plt.show()
    elif dE >=3:
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Y[:,0], Y[:,1], Y[:,2], lw=0.5, linestyle='None', marker='.', markersize=1)
        ax.set_xlabel('X(t)')
        ax.set_ylabel(f'X(t+{T})')
        ax.set_zlabel(f'X(t+{2*T})')
        ax.set_title(f'Przestrzeń Fazowa 3D dla {system_name} (dE={dE}, T={T})')
        plt.show()
    
    # 3. Analiza R/S i wyznaczenie wykładnika Hursta
    H = perform_hurst_analysis(signal, system_name)
    
    return Y, H

# Funkcje Generujące Dane

def generate_periodic_signal(frequency=5, sampling_rate=100, duration=10):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

def generate_random_signal(N=10000):
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
    return np.arange(N) * dt, xs

def generate_henon(a=1.4, b=0.3, N=10000):
    x = np.empty(N)
    y = np.empty(N)
    x0, y0 = 0, 0
    x[0], y[0] = x0, y0
    for i in range(1, N):
        x[i] = 1 - a * x[i-1]**2 + y[i-1]
        y[i] = b * x[i-1]
    return np.arange(N), x

# Przykładowe Użycie

if __name__ == "__main__":
    # Generowanie i analiza sygnału periodycznego
    t_periodic, signal_periodic = generate_periodic_signal(frequency=5, sampling_rate=100, duration=10)
    Y_periodic, H_periodic = analyze_signal_with_hurst(
        t_periodic, signal_periodic, system_name='Sygnał Periodyczny', dE=2, T=15
    )
    
    # Wizualizacja oryginalnego sygnału periodycznego
    plt.figure(figsize=(10,4))
    plt.plot(t_periodic, signal_periodic)
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.title('Sygnał Periodyczny')
    plt.grid(True)
    plt.show()
    
    # Generowanie i analiza sygnału losowego
    t_random, signal_random = generate_random_signal(N=10000)
    Y_random, H_random = analyze_signal_with_hurst(
        t_random, signal_random, system_name='Sygnał Losowy', dE=2, T=10
    )
    
    # Wizualizacja oryginalnego sygnału losowego
    plt.figure(figsize=(10,4))
    plt.plot(t_random, signal_random, linestyle='None', marker='.', markersize=1)
    plt.xlabel('Próbki')
    plt.ylabel('Amplituda')
    plt.title('Sygnał Losowy')
    plt.grid(True)
    plt.show()
    
    # Generowanie i analiza układu Lorenza
    t_lorenz, signal_lorenz = generate_lorenz_system(initial_state=(1.0, 1.0, 1.0), sigma=10.0, rho=28.0, beta=8/3, dt=0.01, N=10000)
    Y_lorenz, H_lorenz = analyze_signal_with_hurst(
        t_lorenz, signal_lorenz, system_name='Układ Lorenza', dE=3, T=10
    )
    
    # Wizualizacja oryginalnego sygnału układu Lorenza
    plt.figure(figsize=(10,4))
    plt.plot(t_lorenz, signal_lorenz)
    plt.xlabel('Czas')
    plt.ylabel('X')
    plt.title('Układ Lorenza - Zmienna X')
    plt.grid(True)
    plt.show()
    
    # Generowanie i analiza Mapy Hénona
    t_henon, signal_henon = generate_henon(a=1.4, b=0.3, N=10000)
    Y_henon, H_henon = analyze_signal_with_hurst(
        t_henon, signal_henon, system_name='Mapa Hénona', dE=2, T=1
    )
    
    # Wizualizacja oryginalnego sygnału Mapy Hénona
    plt.figure(figsize=(10,4))
    plt.plot(t_henon, signal_henon, linestyle='None', marker='.', markersize=1)
    plt.xlabel('Próbki')
    plt.ylabel('Amplituda')
    plt.title('Mapa Hénona - Zmienna X')
    plt.grid(True)
    plt.show()