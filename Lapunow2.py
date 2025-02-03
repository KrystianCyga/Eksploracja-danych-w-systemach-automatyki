import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import nolds  # Biblioteka do analizy wykładników Lyapunova

# Funkcje generujące sygnały
def generate_periodic_signal(frequency=5, sampling_rate=100, duration=10):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

def generate_random_signal(N=1000, sampling_rate=100):
    np.random.seed(0)  # Dla powtarzalności
    signal = np.random.randn(N)
    t = np.arange(N) / sampling_rate
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
    t = np.linspace(0, N*dt, N, endpoint=False)
    return t, xs, ys, zs

def generate_henon(length, a=1.4, b=0.3):
    x = np.zeros(length)
    y = np.zeros(length)
    x[0], y[0] = 0.0, 0.0
    for i in range(1, length):
        x[i] = 1 - a * x[i-1]**2 + y[i-1]
        y[i] = b * x[i-1]
    t = np.arange(length)  # Czas dyskretny
    return t, x, y

# Funkcja do tworzenia przestrzeni fazowej (delay embedding)
def create_embedding(data, dimension, delay=1):
    N = len(data)
    if dimension * delay > N:
        raise ValueError("Zbyt mała długość danych dla zadanej przestrzeni fazowej.")
    embedded = np.array([data[i:N - (dimension - 1) * delay + i:delay] for i in range(dimension)]).T
    return embedded

# Funkcja do obliczania całki korelacyjnej C(epsilon)
def compute_C_epsilon(embedded_data, epsilons):
    N = len(embedded_data)
    # Oblicz macierz odległości
    distance_matrix = squareform(pdist(embedded_data, metric='euclidean'))
    # Zlicz pary (i,j) gdzie odległość <= epsilon
    C_eps = []
    for eps in epsilons:
        count = np.sum(distance_matrix <= eps) - N  # odejmujemy N, bo odległość punktu do samego siebie to 0
        C_eps.append(count / (N**2))
    return np.array(C_eps)

# Ulepszona funkcja do estymacji D2 poprzez regresję liniową na log-log z automatycznym wyborem przedziału
def estimate_D2(epsilons, C_eps):
    # Filtrujemy epsilon i C(epsilon) > 0
    mask = (C_eps > 0)
    log_eps = np.log(epsilons[mask])
    log_C = np.log(C_eps[mask])
    
    if len(log_eps) < 2:
        return np.nan, None  # Zbyt mało punktów do regresji
    
    # Wyszukaj przedział, który ma najwyższą korelację
    best_r2 = -np.inf
    best_slope = np.nan
    best_intercept = np.nan
    best_start = 0
    best_end = 0
    
    # Próbujemy różnych przedziałów i wybieramy ten z najwyższą korelacją
    # Minimum długość przedziału: 10 punktów
    for start in range(0, len(log_eps)-10):
        for end in range(start+10, len(log_eps)+1):
            X = log_eps[start:end].reshape(-1, 1)
            y = log_C[start:end]
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            if r2 > best_r2:
                best_r2 = r2
                best_slope = model.coef_[0]
                best_intercept = model.intercept_
                best_start = start
                best_end = end
    
    # Możemy zdecydować, czy r2 jest wystarczająco wysoki
    if best_r2 < 0.8:
        print(f"Ostrzeżenie: niska jakość dopasowania (r2={best_r2:.2f})")
    
    return best_slope, (log_eps[best_start:best_end], best_slope * log_eps[best_start:best_end] + best_intercept)

# Funkcja do obliczania Mutual Information (MI) – identyfikacja opóźnienia tau
def calculate_mutual_information(signal, max_lag=100, bins=64):
    """
    Oblicza Mutual Information dla różnych opóźnień.
    
    Parameters:
    - signal: np.array, dane sygnałowe
    - max_lag: int, maksymalne opóźnienie do rozważenia
    - bins: int, liczba binów do histogramu
    
    Returns:
    - mi_values: list, wartości MI dla każdego opóźnienia
    """
    mi_values = []
    for lag in range(1, max_lag + 1):
        x = signal[:-lag]
        y = signal[lag:]
        c_xy = np.histogram2d(x, y, bins=bins)[0]
        # Normalizacja
        c_xy = c_xy / np.sum(c_xy)
        p_x = np.sum(c_xy, axis=1)
        p_y = np.sum(c_xy, axis=0)
        # Obliczanie entropii
        h_x = -np.sum(p_x * np.log(p_x + 1e-12))
        h_y = -np.sum(p_y * np.log(p_y + 1e-12))
        h_xy = -np.sum(c_xy * np.log(c_xy + 1e-12))
        mi = h_x + h_y - h_xy
        mi_values.append(mi)
    return mi_values

# FUNKCJA DO OBLICZANIA Λ_max DLA PRZESUWAJĄCYCH SIĘ OKIEN
def calculate_sliding_lambda_max(signal, fixed_dE, window_size, overlap, delay=1):
    """
    Oblicza Λ_max dla przesuwających się okien sygnału z ustalonym dE.
    
    Parameters:
    - signal: np.array, dane sygnałowe
    - fixed_dE: int, ustalony wymiar zanurzenia
    - window_size: int, rozmiar okna w próbkach
    - overlap: int, nakładanie się okien
    - delay: int, opóźnienie w przestrzeni fazowej
    
    Returns:
    - lambda_max_values: list, Λ_max dla każdego okna
    - window_centers: list, środek czasowy każdego okna
    """
    lambda_max_values = []
    window_centers = []
    
    step = window_size - overlap
    num_windows = (len(signal) - window_size) // step + 1
    
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = signal[start:end]
        
        # Tworzenie przestrzeni fazowej dla aktualnego okna
        try:
            embedded = create_embedding(window, fixed_dE, delay)  # Opcjonalne, jeśli potrzebne dla innych obliczeń
        except ValueError:
            lambda_max_values.append(np.nan)
            window_centers.append((start + end) / 2)
            continue
        
        # Obliczanie Λ_max dla danego okna z użyciem surowych danych
        try:
            lambda_max = nolds.lyap_r(window, emb_dim=fixed_dE, lag=delay, min_tsep=10)
            lambda_max_values.append(lambda_max)
        except Exception as e:
            print(f"Błąd obliczeń Λ_max w oknie {i}: {e}")
            lambda_max_values.append(np.nan)
        
        window_centers.append((start + end) / 2)  # Środek okna w próbkach
    
    return lambda_max_values, window_centers

# FUNKCJA GŁÓWNA DLA ANALIZY
def analyze_signal(signal_name, t, signal, fixed_dE=3, window_size=500, overlap=250, delay=1):
    """
    Analizuje sygnał z ustalonym wymiarem zanurzenia i rysuje Λ_max w czasie.
    
    Parameters:
    - signal_name: str, nazwa sygnału
    - t: np.array, os czasu (może być None)
    - signal: np.array, dane sygnałowe
    - fixed_dE: int, ustalony wymiar zanurzenia (domyślnie 3)
    - window_size: int, rozmiar okna w próbkach
    - overlap: int, nakładanie się okien
    - delay: int, opóźnienie w przestrzeni fazowej
    """
    print(f"\nAnaliza sygnału: {signal_name}")
    
    # Obliczanie Λ_max dla okien z ustalonym dE
    lambda_max_values, window_centers = calculate_sliding_lambda_max(
        signal, fixed_dE, window_size, overlap, delay
    )
    
    # Konwersja środków okien na czas (przy założeniu samplerate=100)
    time_centers = np.array(window_centers) / 100 if t is None else t[window_centers]
    
    # Wykres Λ_max w funkcji czasu/próbek
    plt.figure(figsize=(10, 5))
    if t is None:
        plt.plot(window_centers, lambda_max_values, 'b.-', label=f'Λ_max (dE={fixed_dE})')
        plt.xlabel('Numer próbki')
    else:
        plt.plot(time_centers, lambda_max_values, 'b.-', label=f'Λ_max (dE={fixed_dE})')
        plt.xlabel('Czas [s]')
    plt.ylabel('Λ_max')
    plt.title(f'Największy wykładnik Lyapunova dla sygnału {signal_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

# PRZYKŁADOWE UŻYCIE Z USTALONYM dE=3
signals = {
    'Sygnał okresowy': {'signal': generate_periodic_signal()[1], 't': None},
    'System Lorenz (x)': {'signal': generate_lorenz_system()[1], 't': None},
    'Mapa Henona (x)': {'signal': generate_henon(10000)[1], 't': None}
}

for name, data in signals.items():
    analyze_signal(
        signal_name=name,
        t=data['t'],
        signal=data['signal'],
        fixed_dE=3,  # Ustaw wymiar zanurzenia wybrany przez użytkownika!
        window_size=500,
        overlap=250
    )

# Funkcja główna do obliczeń
def main():
    # Generowanie sygnałów
    t_periodic, signal_periodic = generate_periodic_signal(frequency=5, sampling_rate=100, duration=10)
    t_random, signal_random = generate_random_signal(N=10000, sampling_rate=100)
    t_lorenz, xs_lorenz, ys_lorenz, zs_lorenz = generate_lorenz_system(N=10000)
    t_henon, x_henon, y_henon = generate_henon(length=10000)
    
    # Definicja parametrów dla przesuwającego się okna
    fixed_dE = 3  # ustalony wymiar zanurzenia
    window_size = 500  # przykładowa wielkość okna
    overlap = 250      # przykładowe nakładanie się okien
    
    # Analiza dla każdego sygnału z ustalonym dE
    signals = {
        'Sygnał okresowy': {'signal': generate_periodic_signal()[1], 't': np.linspace(0, 10, 1000, endpoint=False)},
        'System Lorenz (x)': {'signal': generate_lorenz_system()[1], 't': np.linspace(0, 100, 10000, endpoint=False)},
        'Mapa Henona (x)': {'signal': generate_henon(10000)[1], 't': np.arange(10000) / 100}  # zakładając sampling_rate=100
    }
    
    for name, data in signals.items():
        analyze_signal(
            signal_name=name,
            t=data['t'],
            signal=data['signal'],
            fixed_dE=fixed_dE,  # Ustalony wymiar zanurzenia
            window_size=window_size,
            overlap=overlap,
            delay=1  # Zakładane opóźnienie
        )
    
    # Wyświetlenie podsumowania
    print("\nPodsumowanie D2, K2 i Lambda_max dla każdego sygnału:")
    for name, metrics in results.items():
        D2_vals = metrics['D2']
        K2_vals = metrics['K2']
        Lyapunov_exponents = metrics['Lyapunov_exponents']
        D2_str = ', '.join([f"{dE}: {D2:.4f}" if not np.isnan(D2) else f"{dE}: NaN" 
                            for dE, D2 in zip(range(1, 9), D2_vals)])
        if any(not np.isnan(k2) for k2 in K2_vals):
            K2_str = ', '.join([f"{dE}: {k2:.4f}" if not np.isnan(k2) else f"{dE}: NaN" 
                                for dE, k2 in zip(range(1, 9), K2_vals)])
        else:
            K2_str = 'Brak danych'
        if any(not np.isnan(lam) for lam in Lyapunov_exponents):
            Lyap_str = ', '.join([f"{dE}: {lam:.4f}" if not np.isnan(lam) else f"{dE}: NaN" 
                                  for dE, lam in zip(range(1, 9), Lyapunov_exponents)])
        else:
            Lyap_str = 'Brak danych'
        print(f"\n{name}:")
        print(f"  D2: {D2_str}")
        print(f"  K2: {K2_str}")
        print(f"  Lambda_max: {Lyap_str}")

if __name__ == "__main__":
    main()