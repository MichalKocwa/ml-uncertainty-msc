# Projekt: porównanie metod estymacji niepewności (praca magisterska)

**Przed rozpoczęciem jakiegokolwiek zadania przeczytaj `docs/experiment_brief.md`.**
Ten plik zawiera pełną specyfikację eksperymentów i jest nadrzędny wobec doraźnych
ustaleń w rozmowie.

Repozytorium realizuje warstwę eksperymentalną pracy porównującej pięć metod estymacji
niepewności w regresji: Gaussian processes, Bayes by Backprop, Monte Carlo dropout,
Laplace approximation, deep ensembles, plus deterministyczny baseline MAP.

## Zasady twarde

- **Nie wymyślać wyników.** Nie wpisywać liczb, których nie policzył uruchomiony kod.
  Puste komórki i `TODO` są dopuszczalne, zmyślone wartości nie.
- **Nie edytować plików `.tex`.** Potrzebne zmiany w tekście pracy zgłaszać w raporcie.
- **Nie dodawać zależności** spoza tabeli w sekcji 1.1 briefu (wyjątek: `numpyro` dla E4).
- **Nie zastępować bibliotek własną implementacją** algorytmu bez pytania. Własny kod
  ogranicza się do klas opakowujących do wspólnego interfejsu.
- **Bez notebooków.** Kod w `src/` i `experiments/` jako moduły i skrypty CLI.
- **Determinizm.** Każdy wynik odtwarzalny z podanego seeda.
- Angielski w kodzie i komentarzach, pisownia brytyjska (`normalise`, `behaviour`).
- nie commituj nic 

## Protokół pracy

Praca podzielona jest na etapy (sekcja 12 briefu). **Realizować jeden etap na sesję.**
Na końcu etapu zatrzymać się i zdać raport: co działa, jakie liczby wyszły, co budzi
wątpliwości. Nie przechodzić do kolejnego etapu bez potwierdzenia.

W briefie są oznaczone **punkty decyzyjne** (np. model szumu w Etapie 2). Przy nich
zatrzymać się i zapytać — nie wybierać samodzielnie, nawet jeśli jedna opcja wydaje się
oczywista. Te decyzje mają konsekwencje dla tekstu pracy, nie tylko dla kodu.

Przed pisaniem kodu w nowym etapie: przedstawić plan i poczekać na akceptację.

## Układ repo

```
src/            biblioteka: methods/, data.py, metrics.py, config.py, style.py
experiments/    skrypty CLI, jeden na eksperyment (e0_*, e1_*, ...)
results/        CSV z wynikami, dopisywane nie nadpisywane
figures/        wygenerowane rysunki
docs/           experiment_brief.md
tests/          testy jednostkowe
```

## Komendy

```bash
pytest tests/ -q                          # testy
python experiments/e1_synthetic.py --quick  # szybka wersja dowolnego eksperymentu
```

## Bibliografia

Nie dopisywać niczego do plików `.bib`. Klucze niezweryfikowane, oznaczone w briefie,
weryfikuje autor.
