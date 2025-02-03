import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

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

# Funkcja główna do analizy sygnału i wyznaczania K2
def analyze_signal(signal_name, t, signal, max_dimension=8, delay=1, num_epsilons=50, log_eps_k2=None):
    """
    Parameters:
    - signal_name: str, nazwa sygnału
    - t: nieużywane, placeholder
    - signal: np.array, dane sygnałowe
    - max_dimension: int, maksymalny wymiar zanurzenia
    - delay: int, opóźnienie
    - num_epsilons: int, liczba wartości epsilon
    - log_eps_k2: float or None, log(eps) dla którego wyznaczana jest K2
                   Jeśli None, K2 nie jest wyznaczane
    """
    print(f"\nAnaliza sygnału: {signal_name}")
    D2_values = []
    K2_values = []
    dimensions = range(1, max_dimension + 1)
    
    # Tworzenie jednego wspólnego wykresu dla log(C(epsilon)) vs log(epsilon)
    plt.figure(figsize=(10, 6))
    
    for dE in dimensions:
        try:
            embedded = create_embedding(signal, dE, delay)
        except ValueError as e:
            print(f"Wymiar zanurzenia {dE} przekracza dostępne dane. Przechodzę do następnego wymiaru.")
            D2_values.append(np.nan)
            K2_values.append(np.nan)
            continue
        
        # Określenie zakresu epsilon z zapewnieniem, że eps_min > 0
        distances = pdist(embedded, 'euclidean')
        eps_min = np.percentile(distances, 0.1)
        eps_max = np.percentile(distances, 99.9)
        
        # Zapewnienie, że eps_min jest większy od zera
        if eps_min <= 0:
            positive_distances = distances[distances > 0]
            if len(positive_distances) == 0:
                eps_min = 1e-10
            else:
                eps_min = np.min(positive_distances)
            print(f"Ustawiono eps_min na {eps_min} dla wymiaru zanurzenia {dE}")
        
        # Generowanie logarytmicznie rozłożonych epsilonów
        epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), num_epsilons)
        
        C_eps = compute_C_epsilon(embedded, epsilons)
        
        # Estymacja D2
        D2, fit_info = estimate_D2(epsilons, C_eps)
        D2_values.append(D2)
        print(f"Wymiar zanurzenia {dE}: D2 = {D2:.4f}")
        
        # Wykres log(C(epsilon)) vs log(epsilon) na wspólnym wykresie
        plt.plot(np.log(epsilons), np.log(C_eps), marker='o', linestyle='-', label=f'dE={dE}')
        
        
        
        # Wyznaczanie K2, jeśli log_eps_k2 jest podany
        if log_eps_k2 is not None:
            # Interpolacja C(eps) dla zadanej epsilon
            eps_target = np.exp(log_eps_k2)
            if eps_target < eps_min or eps_target > eps_max:
                print(f"Warning: epsilon={eps_target} dla dE={dE} poza zakresem [{eps_min}, {eps_max}]. K2 nie jest wyznaczane.")
                K2 = np.nan
            else:
                # Interpolacja C(eps)
                interp_func = interp1d(epsilons, C_eps, kind='linear', bounds_error=False, fill_value="extrapolate")
                C_eps_target = interp_func(eps_target)
                if C_eps_target <= 0:
                    print(f"Warning: C(eps) <= 0 dla epsilon={eps_target} dE={dE}. K2 nie jest wyznaczane.")
                    K2 = np.nan
                else:
                    log_C_eps = np.log(C_eps_target)
                    K2 = (log_C_eps - D2 * log_eps_k2) / dE
            K2_values.append(K2)
            if not np.isnan(K2):
                print(f"Wymiar zanurzenia {dE}: K2 = {K2:.4f}")
            else:
                print(f"Wymiar zanurzenia {dE}: K2 = NaN")
        else:
            K2_values.append(np.nan)
    
    plt.xlabel('log(epsilon)')
    plt.ylabel('log(C(epsilon))')
    plt.title(f'log(C(epsilon)) vs log(epsilon) dla sygnału {signal_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Wykres D2 vs dE
    plt.figure(figsize=(8, 6))
    plt.plot(dimensions, D2_values, 'bo-', label='D2')
    plt.xlabel('Wymiar zanurzenia dE')
    plt.ylabel('D2')
    plt.title(f'Zależność D2 od wymiaru zanurzenia dla sygnału {signal_name}')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    # Wykres K2 vs dE, jeśli K2 jest obliczana
    if not all(np.isnan(K2) for K2 in K2_values):
        plt.figure(figsize=(8, 6))
        plt.plot(dimensions, [-k if not np.isnan(k) else np.nan for k in K2_values], 'ro-', label='K2')
        plt.xlabel('Wymiar zanurzenia dE')
        plt.ylabel('K2')
        plt.title(f'Zależność entropii korelacyjnej K2 od wymiaru zanurzenia dla sygnału {signal_name}')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return D2_values, K2_values

# Generowanie i analiza sygnałów
def main():
    # Generowanie sygnałów
    t_periodic, signal_periodic = generate_periodic_signal(frequency=5, sampling_rate=100, duration=10)
    t_random, signal_random = generate_random_signal(N=1000, sampling_rate=100)
    t_lorenz, xs_lorenz, ys_lorenz, zs_lorenz = generate_lorenz_system(N=10000)
    t_henon, x_henon, y_henon = generate_henon(length=10000)
    
    # Organizacja sygnałów w słowniku z informacją o log_eps_k2
    signals = {
        'Sygnał okresowy': {'signal': signal_periodic, 'log_eps_k2': -1},
        'Sygnał losowy': {'signal': signal_random, 'log_eps_k2': -1},
        'System Lorenz (x)': {'signal': xs_lorenz, 'log_eps_k2': -1},
        'Mapa Henona (x)': {'signal': x_henon, 'log_eps_k2': -1}
    }
    
    # Analiza każdego sygnału
    results = {}
    for name, data in signals.items():
        signal = data['signal']
        log_eps_k2 = data['log_eps_k2']
        D2_vals, K2_vals = analyze_signal(
            signal_name=name, 
            t=None, 
            signal=signal, 
            max_dimension=8, 
            delay=1, 
            num_epsilons=50, 
            log_eps_k2=log_eps_k2
        )
        results[name] = {'D2': D2_vals, 'K2': K2_vals}
    
    # Wyświetlenie podsumowania
    print("\nPodsumowanie D2 i K2 dla każdego sygnału:")
    for name, metrics in results.items():
        D2_vals = metrics['D2']
        K2_vals = metrics['K2']
        D2_str = ', '.join([f"{dE}: {D2:.4f}" if not np.isnan(D2) else f"{dE}: NaN" 
                            for dE, D2 in zip(range(1, 9), D2_vals)])
        if any(not np.isnan(k2) for k2 in K2_vals):
            K2_str = ', '.join([f"{dE}: {K2:.4f}" if not np.isnan(K2) else f"{dE}: NaN" 
                                for dE, K2 in zip(range(1, 9), K2_vals)])
        else:
            K2_str = 'Brak danych'
        print(f"\n{name}:")
        print(f"  D2: {D2_str}")
        print(f"  K2: {K2_str}")

if __name__ == "__main__":
    main()