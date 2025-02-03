# Eksploracja-danych-w-systemach-automatyki
Program rozwiązuje i wizualizuje następujące zagadnienia dla sygnału losowego, periodycznego, układu Hursta i Lorenza.

1. Rekonstrukcja przestrzeni fazowej
x(t) jest przekształcany w reprezentację w przestrzeni wielowymiarowej przy użyciu metody zanurzenia.
Metoda korelacji liniowej – określa optymalne opóźnienie na podstawie autokorelacji sygnału.
Metoda wzajemnej informacji (MI)
Adaptacyjna MI – dobiera T na podstawie pierwszego minimum funkcji wzajemnej informacji.
Metoda KDE (Kernel Density Estimation) – wykorzystuje jądrowe oszacowanie gęstości prawdopodobieństwa do precyzyjniejszego wyboru T.
Wyznaczanie wymiaru zanurzenia 
Metoda całki korelacyjnej – dobiera minimalny wymiar rekonstrukcji poprzez analizę zachowania całki korelacyjnej w zależności od d.
Wykładnik Hursta
Mierzy chropowatość sygnału i określa, czy jest on losowy, periodyczny czy wykazuje cechy pamięci długoterminowej (np. procesy fraktalne).

2. Analiza fraktalna
Obliczane są parametry opisujące strukturę fraktalną sygnału:
Wymiar korelacyjny 𝐷2
Mierzy złożoność struktury dynamicznej układu na podstawie funkcji korelacyjnej.
Entropia korelacyjna 𝐾2
Określa stopień nieprzewidywalności systemu – im większa entropia, tym bardziej chaotyczne zachowanie.

3. Wykładniki Lapunowa
Obliczane są największe wykładniki Lapunowa, które określają stabilność trajektorii dynamicznej:
Dla wartości dodatnich – system wykazuje chaos deterministyczny.
Dla wartości ujemnych – system stabilizuje się do punktu stałego lub cyklu granicznego.
Dla wartości bliskich zeru – system periodyczny.

Analizowane przebiegi
Program analizuje różne typy dynamiki:
Przebieg periodyczny (np. sinusoidalny) – analiza jego regularności.
Przebieg losowy (np. szum biały) – brak struktury deterministycznej.
Układ Lorenza – klasyczny przykład chaosu deterministycznego.
Układ Hénona – dwuwymiarowy model układu dynamicznego z chaosem.
