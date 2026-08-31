# Brief dla Claude Code — eksperymenty porównawcze metod estymacji niepewności

> **Jak tego użyć:** wklej cały ten plik do Claude Code jako pierwszą wiadomość w sesji.
> Zapisz go też w repo jako `docs/experiment_brief.md` — będzie punktem odniesienia
> w kolejnych sesjach i materiałem do rozdziału „Implementation details".

---

## 0. Kontekst

Repozytorium jest częścią pracy magisterskiej porównującej pięć metod estymacji niepewności
w regresji. Rozdziały teoretyczne są napisane; brakuje warstwy eksperymentalnej.

Porównywane metody (rozdział 3 pracy):

1. **Gaussian processes** (GP) — dokładny posterior, punkt odniesienia
2. **Bayesian neural networks / Bayes by Backprop** (BBB) — VI, rodzina gaussowska mean-field
3. **Monte Carlo dropout** (MCD) — VI, rodzina bernoullowska
4. **Laplace approximation** (LA) — lokalny gaussian wokół MAP, post-hoc
5. **Deep ensembles** (DE) — podejście niebayesowskie

Plus **deterministic baseline** (MAP z głowicą log-wariancji) — nie jest szóstą metodą,
tylko dolnym punktem odniesienia dla niepewności epistemicznej.

**Teza, którą warstwa eksperymentalna ma zoperacjonalizować:** wszystkie te metody
przybliżają tę samą całkę predykcyjną $p(y^*|x^*,\mathcal{D}) = \int p(y^*|x^*,\theta)\,p(\theta|\mathcal{D})\,d\theta$
i różnią się **wyłącznie** sposobem przybliżenia posteriora. Jeżeli w eksperymencie
różnią się czymkolwiek innym (architekturą, priorem, modelem szumu, budżetem), porównanie
mierzy co innego niż deklaruje.

---

## 1. Zasady twarde

- **Bez notebooków.** Cały kod w `src/` i `experiments/` jako moduły i skrypty CLI.
  Notebooki (`.ipynb`) nie trafiają do repo (blokada w `.gitignore` jest celowa).
- **Bez wymyślania wyników.** Jeżeli eksperyment nie został uruchomiony, nie wolno
  wpisywać liczb do tabel ani do dokumentacji. Puste komórki są dopuszczalne, zmyślone nie.
- **Determinizm.** Każdy wynik musi być odtwarzalny z podanego seeda. Ustawić
  `torch.manual_seed`, `np.random.seed`, `torch.use_deterministic_algorithms(True)`
  tam, gdzie nie łamie to wydajności.
- **Wyniki tylko w CSV.** Jeden narastający plik na eksperyment, o stałym schemacie
  (sekcja 8). Bez pickli, bez JSON-ów z zagnieżdżonymi strukturami.
- **Poprawić `.gitignore`:** wpis `*.csv` blokuje `results/`. Dodać `!results/*.csv`.
- Angielski w kodzie, nazwach plików, komentarzach i docstringach. Pisownia brytyjska
  w tekście dokumentacji (`normalise`, `behaviour`) — spójnie z pracą.

### 1.1 Biblioteki zamiast własnych implementacji

**Zasada nadrzędna: minimalizować liczbę decyzji bez źródła w literaturze.** Rozdział 4
pracy musi opisać każdą taką decyzję i obronić ją na obronie. Wierna implementacja
opublikowanego algorytmu wymaga jednego zdania („zgodnie z Blundell i in. 2015") i nie
jest problemem. Problemem są wybory, których nie ma w żadnym źródle.

Stąd podział:

| Metoda | Źródło | Uzasadnienie |
|---|---|---|
| Gaussian process | `scikit-learn` | wnioskowanie dokładne, brak decyzji projektowych |
| Bayes by Backprop | `bayesian-torch` (Intel Labs) | `LinearReparameterization` = Blundell i in. 2015 |
| Monte Carlo dropout | `torch.nn.Dropout` | metoda *jest* zwykłym dropoutem; biblioteka nic nie doda |
| Laplace | `laplace-torch` | biblioteka autorów pracy cytowanej w 3.4 |
| Deep ensembles | pętla po `M` treningach | metoda *jest* pętlą |
| Baseline MAP | `torch.nn` | sieć bazowa |

Własny kod ogranicza się do klas opakowujących do wspólnego interfejsu z sekcji 3.

**Biblioteka nie zwalnia z dokumentowania.** Każdy pozostawiony domyślny argument jest
nadal decyzją — tylko niejawną. Rozdział 4 musi wymienić wersje bibliotek i komplet
użytych argumentów, łącznie z domyślnymi. „Biblioteka tak robi" nie jest odpowiedzią.

**Nie dodawać zależności spoza tej tabeli bez pytania**, poza `numpyro` dla E4.

---

## 2. Co zostaje z obecnego repo, co przepisujemy

| Plik | Decyzja |
|---|---|
| `src/data.py` | **Zostaje**, rozbudować (sekcja 5). Poprawić błędny komentarz przy `StandardScaler` — standaryzacja daje średnią 0, nie 1. |
| `src/metrics.py` | **Zostaje**, rozbudować (sekcja 7). |
| `src/models.py` | **Przepisać** na pakiet `src/methods/` ze wspólnym interfejsem. Usunąć nieużywany import `BayesianRidge`. |
| `src/plotting.py` | **Przepisać** — funkcje ad-hoc zastąpić modułem generującym konkretne rysunki pracy pod ustalonymi nazwami plików (sekcja 10). |

Nie ma potrzeby zaczynać od zera, ale nie ma też obowiązku zachowywania obecnych sygnatur.

---

## 3. Wspólny interfejs — najważniejsza część zadania

Wszystkie metody muszą realizować jeden protokół. Bez tego porównanie rozjedzie się
na poziomie kodu i każda tabela będzie wymagała ręcznego sklejania.

```python
# src/methods/base.py
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class Prediction:
    """Predykcja z rozbiciem wariancji zgodnym z prawem wariancji całkowitej."""
    mean: np.ndarray            # (n,)
    var_aleatoric: np.ndarray   # (n,)  E_q[sigma^2_theta(x)]
    var_epistemic: np.ndarray   # (n,)  Var_q[mu_theta(x)]
    samples: np.ndarray | None = None   # (T, n) — średnie z poszczególnych przebiegów,
                                        # None dla metod bez próbkowania (GP, LA-linearised)

    @property
    def var_total(self) -> np.ndarray:
        return self.var_aleatoric + self.var_epistemic

    @property
    def std_total(self) -> np.ndarray:
        return np.sqrt(self.var_total)


class UncertaintyMethod(Protocol):
    name: str          # "gp", "bbb", "mcd", "laplace", "ensemble", "map"
    n_parameters: int  # do tabeli kosztów

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> "UncertaintyMethod": ...
    def predict(self, X: np.ndarray) -> Prediction: ...
```

Rejestr metod: `src/methods/__init__.py` z `METHODS: dict[str, Callable[[Config], UncertaintyMethod]]`,
żeby runner mógł iterować po nazwach z linii poleceń.

**Krytyczne:** `var_aleatoric` i `var_epistemic` muszą być raportowane osobno dla
**każdej** metody. To jest główny wynik pracy, nie dodatek. Metoda, która nie potrafi
ich rozdzielić, musi to zgłosić jawnie (np. `var_epistemic = zeros` dla `map`),
a nie zwracać wariancję łączną w jednym polu.

---

## 4. Specyfikacja poszczególnych metod

### Wspólny backbone

```
MLP: input → [Linear(h) → ReLU → (dropout)] × 1 → Linear(1)
     + jeden globalny, uczony parametr log_sigma2 (skalar)
```

Domyślnie: **`L = 1` warstwa ukryta, `h = 50` jednostek, `activation = ReLU`**,
`log_sigma2` klamrowany do `[-6, 6]`. Trening: Adam, `lr = 1e-2`, `weight_decay = 0`,
Gaussian NLL **plus jawna kara priora** (sekcja 4.6). Liczba epok — patrz sekcja 5.2.

**Model szumu: homoskedastyczny, jednolity we wszystkich metodach.** Jeden globalny
parametr `log_sigma2` uczony wspólnie ze stratą, o statusie identycznym z `noise_level`
w `WhiteKernel` GP: stała estymowana z danych. `var_aleatoric = exp(log_sigma2)`,
identyczna dla wszystkich punktów.

Decyzja podjęta w Etapie 2 po ustaleniu, że `laplace-torch` twardo odrzuca wektorowe
`sigma_noise` (`baselaplace.py`: *„Only homoscedastic output noise supported"*) oraz że
klasa `KronSubnetLaplace` nie istnieje — co czyniło Rysunek 3.9 niewykonalnym
w trybie subnetwork niezależnie od modelu szumu.

Uzasadnienie do rozdziału 4: **jednolitość, nie wygoda.** Argument z rozdziału 3.3
(l. 177–184) przetrwał zmianę bez modyfikacji — dotyczy symetrii modelu szumu między
metodami, a nie kierunku, w którym się ją osiąga. Dodatkowa korzyść: znika wyjątek GP
opisywany w `rozdzial31.tex` l. 155, więc wszystkie sześć metod dzieli jeden model szumu.

**`log_sigma2` nie podlega karze priora.** Kara liczona wyłącznie po parametrach MLP.
Objęcie nią `log_sigma2` ciągnęłoby `sigma^2` w stronę 1 i tworzyło asymetrię wobec GP,
gdzie `noise_level` nie jest regularyzowany. Błąd cichy — przechodzi testy i daje sensowne
liczby — więc wymaga osobnego testu (sekcja 4.7).

Ta sama architektura, ten sam optymalizator i ta sama liczba epok dla BBB, MCD, LA,
DE i baseline'u. GP nie ma architektury — pozostaje wyjątkiem strukturalnym, ale już
nie w modelu szumu.

**Wariant heteroskedastyczny** (dwie głowice, `eq:mc-dropout-heteroscedastic`) zostaje
poza porównaniem głównym, jako osobna ilustracja do Rysunku 2.1. Wystarczy tam sam
baseline, bez Laplace'a.

### 4.1 Gaussian process

`sklearn.gaussian_process.GaussianProcessRegressor`, kernel `ConstantKernel × RBF + WhiteKernel`,
`n_restarts_optimizer=5`.

- `var_epistemic` = wariancja z `return_std=True` **bez** członu szumu
- `var_aleatoric` = `noise_level` z dopasowanego `WhiteKernel`, stałe po całej przestrzeni
- **Uwaga:** `normalize_y=True` normalizuje target wewnętrznie, a sieci nie. Ustawić
  `normalize_y=False` i standaryzować `y` na zewnątrz, jednakowo dla wszystkich metod
  (sekcja 5.3). Inaczej NLL nie jest porównywalny między GP a sieciami.
- **Bez podpróbkowania.** GP trenowany na pełnym zbiorze treningowym, tak samo jak
  pozostałe metody. Dobór zbiorów zapewnia wykonalność (sekcja 5.2).

**Polityka wobec największego zbioru.** Na zbiorach, gdzie dokładny GP przestaje być
wykonalny w ramach pełnego protokołu, GP **nie jest uruchamiany**, a komórka w tabeli
zawiera `—` z przypisem, nigdy pustkę ani wartość z innego `N`.

Trzy rzeczy muszą temu towarzyszyć, inaczej pominięcie jest gołosłowne:

1. **Zmierzony koszt zamiast twierdzenia.** Eksperyment E0 (sekcja 9) mierzy czas
   dopasowania i szczytowe zużycie pamięci dokładnego GP dla rosnącego `N`.
   **Wykonane — wyniki w sekcji 11 przy P14.** Uzasadnienie opiera się na trzech
   liczbach z własnego pomiaru: nachyleniu `~N^2,3`, mnożniku 4,2–8,7× od strojenia
   hiperparametrów, oraz dziewięciu macierzach `N × N` w pamięci.
2. **Precyzyjne sformułowanie.** Przy `N ≈ 8600` dokładny GP jest *wykonalny*, ale
   przy 20 podziałach i wielokrotnych restartach optymalizatora koszt rośnie do rzędu
   godzin na zbiór. Napisać „koszt nieproporcjonalny do wartości informacyjnej",
   a nie „metoda się nie skaluje" — to drugie jest nieprawdą przy tym `N`.
3. **Konsekwencja w tekście.** Każde stwierdzenie porównawcze o GP w rozdziale 5 musi
   być wtedy warunkowane zakresem `N`, w którym GP wystąpił.

*Opcjonalnie, jako diagnostyka:* wiersz `gp_subsampled` trenowany na 2000 punktach,
oznaczony osobno. Wolno go czytać wyłącznie jako „ile GP osiąga przy ograniczonych
danych", nigdy jako wiersz porównawczy obok metod trenowanych na pełnym zbiorze.

Rzadkie przybliżenia GP (*inducing points*) pozostają wykluczone — rozdział 3.1 uzasadnia
to tym, że zastąpiłyby dokładny posterior przybliżonym, czyli usunęły własność
kwalifikującą GP jako punkt odniesienia. Nie wprowadzać ich jako obejścia kosztu.

### 4.2 Bayes by Backprop

**`bayesian-torch` (Intel Labs).** Warstwa `LinearReparameterization` realizuje wprost
Blundella i in. 2015, `get_kl_loss` zwraca człon KL. Instalacja: `pip install bayesian-torch`.
Nie używać `blitz-bayesian-pytorch` — słabiej utrzymywany.

- Prior: `N(0, gamma^2 I)` przez `prior_mu = 0`, `prior_sigma = gamma` —
  **ta sama wartość `gamma` co w Laplace i HMC** (sekcja 4.6)
- Nie włączać MOPED (inicjalizacja z wytrenowanej sieci deterministycznej) ani Flipout
  w konfiguracji podstawowej — obie zmieniają metodę względem opisu w rozdziale 3.2.
  Flipout może wejść wyłącznie jako świadoma ablacja, oznaczona osobno.
- Ważenie KL po minibatchach: `pi_i = 1/M` (jednorodne)
- Predykcja: `T` przebiegów, estymator zgodny z `eq:bbb-variance-estimator`,
  dzielnik `T-1` w członie epistemicznym
- Zapisać w dokumentacji wersję biblioteki i komplet argumentów, w tym `posterior_rho_init`

**Do sprawdzenia w Etapie 2:** czy `dnn_to_bnn()` poprawnie obsługuje sieć z dwiema
głowicami wyjściowymi. Jeśli nie — składać model z warstw `LinearReparameterization`
ręcznie, co nadal nie jest własną implementacją algorytmu.

### 4.3 Monte Carlo dropout

- Dropout **przed każdą warstwą z wagami**, zgodnie z gal2016 — nie tylko przed ostatnią
- Własny moduł `AlwaysOnDropout`, żeby nie zależeć od `model.train()` przy predykcji
- Model szumu homoskedastyczny, wspólny z pozostałymi metodami (sekcja 4). Wariant
  heteroskedastyczny wg kendall2017 pozostaje opisany w rozdziale 3.3, ale nie jest
  uruchamiany w porównaniu głównym.
- `p` (dropout rate) jako hiperparametr eksperymentu, domyślnie `0.1`
- `T` domyślnie 100 — **wartość prowizoryczna**, uzasadniana dopiero przez E6a.
  Oznaczyć w kodzie.
- Kara priora musi być niezerowa — bez niej odpowiedniość z VI nie zachodzi

### 4.4 Laplace approximation

**`laplace-torch >= 0.2.3`** — biblioteka z pracy, którą rozdział 3.4 cytuje.
Punkt decyzyjny z Etapu 2 **rozstrzygnięty**: tryb pełnosieciowy, model homoskedastyczny.

- `subset_of_weights="all"` — **nie** `subnetwork`. Tryb subnetwork działa dla głowicy
  średniej, ale `KronSubnetLaplace` nie istnieje, więc Rysunek 3.9 byłby niewykonalny.
- `hessian_structure` ∈ `{full, kron, diag}` — wszystkie trzy, do Rysunku 3.9.
  Przy 1×50 sieci mają rząd 200–550 parametrów, więc `full` jest trywialne, a `kron`
  wchodzi jako pytanie o wierność faktoryzacji, nie o wykonalność.
- `sigma_noise` **skalarne** — dokładnie to, czego biblioteka oczekuje.
- Predykcja **linearised**, nie próbkowana (daxberger2021), zgodnie z `eq:laplace-predictive`
- `prior_precision` strojony przez marglik; wariant nieregularyzowany jako osobna
  konfiguracja do ablacji E6c
- Zapisać w dokumentacji **wersję biblioteki i komplet użytych argumentów**, łącznie
  z domyślnymi.

**Test P8 — na dokładną równość**, nie `approx`. Predykcja linearised nie modyfikuje
średniej MAP, więc identyczność jest bitowa. Różnica choćby na ostatnim bicie oznacza
dodatkowy forward pass lub inną ścieżkę numeryczną i wymaga wyjaśnienia, nie tolerancji.

**Do sprawdzenia i udokumentowania:** czy marglik stroi również `sigma_noise`. Jeśli tak,
Laplace i MAP mają identyczną średnią, ale **różny człon aleatoryczny** — co przy PICP
wygląda jak różnica metod, a jest różnicą procedury strojenia szumu. Nie jest to błąd,
ale dotyczy dekompozycji z sekcji 7.4 i musi być opisane w rozdziale 4.

### 4.5 Deep ensembles

- `M = 5` członków, różne inicjalizacje i różna kolejność batchy (`lakshminarayanan2017`)
- **Bez adversarial training** — uzasadnić w dokumentacji: autorzy raportują to jako
  opcjonalne wzmocnienie, a włączenie go dodałoby metodzie składnik nieobecny
  w pozostałych czterech
- Estymator wariancji identyczny jak dla BBB (`eq:bbb-variance-estimator`),
  z `T = M`
- **Budżet:** ensemble dostaje `M`-krotność kosztu pozostałych metod. To jest świadomy
  wybór (*equal architecture*, nie *equal compute*), więc koszt musi być raportowany
  jako osobna metryka (sekcja 7.4), inaczej porównanie jest przechylone.

### 4.6 Zgodność priorów — punkt, który łatwo przeoczyć

BBB, LA i HMC muszą używać **tego samego priora** `N(0, gamma^2 I)` z tą samą `gamma`.

**Rekomendacja: prior jawnie w stracie, `weight_decay = 0` wszędzie.**

Dla MAP i MC dropout dodawać do straty człon `||theta||^2 / (2 * gamma^2 * N)` i ustawiać
`weight_decay=0` w optymalizatorze. Wtedy prior jest zdefiniowany jednym wzorem, identycznym
dla wszystkich metod, i **wybór optymalizatora przestaje mieć jakikolwiek wpływ na prior**.
Jest to też bliższe implementacji referencyjnej Gala, która używa regularyzatora Keras
dodającego karę wprost do straty.

Jeżeli mimo to używany jest `weight_decay` w optymalizatorze, obowiązuje:

```
weight_decay = 1 / (gamma^2 * N)
```

**a nie `1/(2 * gamma^2 * N)`.** PyTorch dodaje `weight_decay * theta` bezpośrednio do
gradientu, a nie karę do straty; pochodna `||theta||^2/(2 gamma^2 N)` wynosi
`theta/(gamma^2 N)`, więc dwójki się skracają. Zweryfikowane empirycznie testem
`test_weight_decay_matches_explicit_prior_penalty`.

**Adam vs AdamW.** W `torch.optim.Adam` `weight_decay` wchodzi do gradientu przed
skalowaniem adaptacyjnym, więc efektywna siła kary zależy od historii gradientu per parametr
— prior przestaje być jednorodny i stały w czasie. `AdamW` odsprzęga zanik wagi, ale nadal
nie odpowiada dokładnie estymacji MAP. Rekomendacja jawnej kary w stracie usuwa ten problem
całkowicie i dlatego jest preferowana.

**Trzy parametryzacje jednego priora — źródło cichych asymetrii:**

| Miejsce | Parametr | Postać |
|---|---|---|
| `bayesian-torch` | `prior_sigma` | odchylenie standardowe |
| `laplace-torch` | `prior_precision` | precyzja, czyli `1/gamma^2` |
| jawna kara w stracie | współczynnik | `1/(2 gamma^2 N)` |
| `torch.optim` (jeśli używany) | `weight_decay` | `1/(gamma^2 N)` |
| `numpyro` (E4) | skala w `dist.Normal` | odchylenie standardowe |

Jedna funkcja w `src/config.py` przyjmująca `gamma` i zwracająca komplet, plus dwa testy:
porównanie log-gęstości tego samego wektora wag (z asercjami przypiętymi do `gamma`, nie
tylko wzajemnymi) oraz test empiryczny jednego kroku SGD. Bez nich asymetria jest niewidoczna
i przechodzi prosto do wyników.

MCD ma inny implikowany prior z natury (bernoullowski) i GP ma prior nad funkcjami —
to są ograniczenia strukturalne, nie do naprawienia. Odnotować w dokumentacji.

---

### 4.7 Testy chroniące spójność, nie tylko poprawność

Trzy błędy w tej architekturze są **ciche**: przechodzą zwykłe testy i dają sensowne
liczby, a różnicę widać dopiero w tabeli wyników, gdzie wygląda jak własność metody.
Każdy wymaga osobnego testu.

**1. Kara priora obejmująca `log_sigma2`.** Ciągnie `sigma^2` w stronę 1 i tworzy
asymetrię wobec GP, gdzie `noise_level` nie jest regularyzowany. Test: dwa treningi
na tych samych danych, różniące się wyłącznie `gamma` (np. `1.0` i `0.01`). Wagi MLP
mają się wyraźnie różnić, dopasowane `log_sigma2` ma pozostać praktycznie takie samo.

**2. Inicjalizacja przed ustawieniem seeda.** Konstruowanie modelu przed `set_seed()`
uzależnia wagi od globalnego stanu RNG. Test: dwa `fit`+`predict` z tym samym seedem
dają identyczny wynik. Rozwiązanie strukturalne: `model_factory` wywoływane **po**
`set_seed()` wewnątrz pętli treningowej.

**3. Rozjazd konwencji priora między metodami.** Sekcja 4.6 — jedna funkcja, dwa testy
(log-gęstość z asercjami przypiętymi do `gamma` oraz empiryczny krok SGD).

Do tego minimum dla każdej metody: poprawne kształty, nieujemne oba człony wariancji,
`map` z dokładnie zerowym członem epistemicznym, stały człon aleatoryczny po `x`
(test wprost sprawdzający homoskedastyczność, nie zakładający jej), oraz asymptotyka GP
— dla punktu bardzo daleko od danych `var_epistemic` dąży do `ConstantKernel`,
a `var_total` do `ConstantKernel + noise_level`.

## 5. Dane

### 5.1 Syntetyczne (1D, jedyne miejsce z pełnym ground truth)

Trzy warianty, wszystkie z **znaną** funkcją `f(x)` i **znanym** `sigma(x)`:

| Wariant | `f(x)` | Szum | Trening | Ewaluacja | Cel |
|---|---|---|---|---|---|
| `sin_homo` | `sin(x)` | `sigma = 0.1` stałe | `[0, 6]`, N=250 | `[-2, 10]`, ≥1000 pkt | ekstrapolacja |
| `sin_hetero` | `sin(x)` | `sigma(x) = 0.05 + 0.15·x/6` | `[0, 6]`, N=250 | `[-2, 10]` | heteroskedastyczność |
| `sin_gap` | `sin(x)` | `sigma = 0.1` | `[0,2] ∪ [4,6]`, 125+125 | `[-2, 10]` | *in-between uncertainty* |

**`N = 250`, nie 50 — decyzja wymuszona, nie estetyczna.** Sieć 1×50 ma 151 parametrów.
Przy `N = 50` macierz GGN ma rząd ≤ 50, więc ponad sto kierunków w przestrzeni parametrów
nie ma żadnej informacji z danych i pozostaje ograniczone wyłącznie priorem. Skutkiem są
wąskie piki wariancji epistemicznej Laplace'a (`full`, `kron`) **wewnątrz zakresu
treningowego**, o wysokości 10–50× przekraczającej wartość na granicy ekstrapolacji.
Zweryfikowane na sześciu seedach: przy `N = 50` maksimum wypadało wewnątrz danych
w 4–5 przypadkach na 6; przy `N = 250` — w 1 na 6, i o wysokości nieodróżnialnej
od normalnego wzrostu przy granicy.

`N = 250` odpowiada też reżimowi `N > p` obowiązującemu we wszystkich zbiorach UCI
(277–8611), więc eksperyment syntetyczny nie różni się od benchmarków w wymiarze,
który nie jest przedmiotem badania.

Wariant `sin_gap` jest najbardziej wartościowy diagnostycznie — luka w środku zakresu
treningowego jest testem, którego metody mean-field zwykle nie przechodzą.

**`sin_hetero` po przejściu na model homoskedastyczny** służy już tylko do jednego celu:
pokazania, czego jednolity model szumu **nie** potrafi. Wszystkie metody odpowiedzą tam
stałym `sigma`, więc porównanie z `sigma_true(x)` (poziom L0) mierzy błąd modelu, nie
różnice między metodami. Zachować jako świadome ograniczenie do opisania w rozdziale 4,
oraz jako źródło Rysunku 2.1 — ten generować osobnym skryptem z siecią dwugłowicową,
poza porównaniem głównym.

### 5.2 Benchmarki UCI — protokół porównywalny z literaturą

**Cel nadrzędny tej sekcji:** tabela główna ma być bezpośrednio zestawialna z liczbami
opublikowanymi dla MC dropout i PBP. To wymaga trafienia w protokół co do szczegółu,
nie tylko w metrykę.

#### Dane i podziały — z repozytorium referencyjnego, nie z UCI

**Nie ładować z `ucimlrepo` i nie dzielić samodzielnie przez `train_test_split`.**

Repozytorium `yaringal/DropoutUncertaintyExps` zawiera katalog `UCI_Datasets` z danymi
oraz plikami indeksów podziałów, identycznymi z tymi z kodu Hernándeza-Lobato. README
ostrzega wprost: ze względu na mały rozmiar zbiorów samodzielne dzielenie danych
najprawdopodobniej da wyniki **nieporównywalne** z raportowanymi.

Konsekwencja: `load_uci()` przyjmuje **numer podziału `0..19`**, nie seed, i czyta
gotowe indeksy z plików. Seed steruje wyłącznie inicjalizacją sieci.

Rozwiązuje to jednocześnie cztery problemy, które inaczej trzeba by rozstrzygać
arbitralnie: wybór targetu w `energy` (Y1 czy Y2), wariant `wine` (red czy red+white),
zależność `yacht` od pojedynczego URL-a w archiwum UCI, oraz zaokrąglenie przy
`test_size=0.1` w `kin8nm`.

**Przed skopiowaniem danych sprawdzić LICENSE tego repozytorium** i zgłosić autorowi.
Nie vendorować przed zgodą.

#### Protokół

| Element | Wartość |
|---|---|
| Podziały | **z plików indeksów**, 20 sztuk, 90/10 |
| Architektura | 1 warstwa ukryta × 50 jednostek ReLU |
| Liczba epok | patrz uwaga niżej |
| Standaryzacja | `X` **i** `y`, dopasowana na treningu, odwracana przed metrykami |
| Metryki | RMSE oraz **average test log-likelihood** |
| Raportowanie | średnia po podziałach ± **standard error** (nie odchylenie standardowe) |

**Uwaga o liczbie epok — rozstrzygnięta.** `n_epochs.txt` w repo referencyjnym zawiera 40,
ale `experiment.py` liczy `n_epochs = int(n_epochs.txt * epochs_multiplier)`, a nazwy plików
wyników zawierają `100_xepochs` dla **wszystkich sześciu** zbiorów. Opublikowane liczby
pochodzą więc z **4000 epok**, nie 40 ani 400. Zapis „10x" w README repo jest opisem prozą
i nie odpowiada danym w katalogu `results/`.

Do tego dochodzi grid search po `dropout rate` i `tau` na wewnętrznym podziale 80-20,
wykonywany **osobno w każdym z 20 foldów** (hiperparametry odrzucane między foldami —
poprawka z 2018 roku, po wykryciu kontaminacji zbioru testowego).

Pełne odtworzenie tego protokołu to 4000 epok × 20 podziałów × 6 zbiorów × siatka
hiperparametrów. **Rozwiązanie: rozdzielić dwa cele.**

- **Tabela główna** — jednolity, tańszy protokół dla wszystkich sześciu metod. Liczba epok
  do ustalenia empirycznie (kiedy strata walidacyjna przestaje spadać), stałe hiperparametry.
  Wewnętrznie spójna, ale nieporównywalna wprost z liczbami Gala.
- **Osobny przebieg walidacyjny (P13)** — wyłącznie MC dropout, wyłącznie na 2–3 najmniejszych
  zbiorach (`yacht`, `energy`, `concrete`), z pełnym odtworzeniem protokołu: 4000 epok
  i grid search per fold. Służy **potwierdzeniu poprawności implementacji**, nie porównaniu
  metod, więc nie musi obejmować wszystkiego.

Ten podział daje P13 jako ostry test przy ułamku kosztu. Opisać go w rozdziale 4 jako
dwa osobne eksperymenty o różnych celach, nie jako niespójność.

#### Zbiory

Dobrane tak, żeby **dokładny GP mieścił się na pełnym zbiorze treningowym**:

| Zbiór | N | N_train | d | Dokładny GP |
|---|---|---|---|---|
| `yacht` | 308 | 277 | 6 | trywialny |
| `energy` | 768 | 691 | 8 | trywialny |
| `concrete` | 1030 | 927 | 8 | trywialny |
| `wine_quality_red` | 1599 | 1439 | 11 | bez problemu |
| `kin8nm` | 8192 | ~7373 | 8 | graniczny |
| `power_plant` | 9568 | 8611 | 4 | koszt nieproporcjonalny → `—` |

**`boston` pominięty świadomie.** Występuje w każdej opublikowanej tabeli, ale został
usunięty ze `scikit-learn` ze względu na cechę zakodowaną rasowo. Napisać w rozdziale 4
jedno zdanie uzasadniające pominięcie i wskazujące, że zawęża to zestaw wierszy
referencyjnych.

**Bez podpróbkowania `max_train`** — wyłącznie dla opcjonalnego wiersza `gp_subsampled`
i dla trybu `--quick`.

### 5.3 Preprocessing

- `X`: `StandardScaler` dopasowany **na zbiorze treningowym**
- `y`: **też standaryzowany** (obecnie nie jest) — dopasowany na treningu
- Metryki liczone **po odwróceniu transformacji `y`**, w jednostkach oryginalnych.
  Bez tego NLL i MPIW nie są porównywalne między zbiorami ani z literaturą.
- Odwrócenie dotyczy **całego rozkładu predykcyjnego**, nie tylko średniej:
  `sigma_original = sigma_standardised * scaler.scale_`. Zapomnienie o tym daje poprawny
  RMSE i błędny NLL — najczęstszy cichy błąd w tym protokole.

### 5.4 Gap split na danych rzeczywistych

Dla 2 zbiorów UCI: wybrać cechę o największej wariancji, usunąć z treningu obserwacje
z przedziału `[q33, q66]` tej cechy, ewaluować osobno na `in-range` i `in-gap`.
Nie ma tu ground truth, ale jest testowalne uporządkowanie: niepewność epistemiczna
w luce powinna być wyższa niż w obszarze gęstym.

---

## 6. Warstwa referencyjna (ground truth)

To jest część, która odróżnia rzetelne porównanie od tabelki z liczbami. Trzy poziomy:

### L0 — dokładna niepewność aleatoryczna (dane syntetyczne)

`sigma(x)` jest znane z konstrukcji. Liczymy błąd estymaty:
`RMSE(sigma_hat(x), sigma_true(x))` po siatce ewaluacyjnej. Metody homoskedastyczne
(GP w wersji standardowej) będą tu wypadać źle na `sin_hetero` — i o to chodzi.

### L1 — dokładny posterior GP (dane syntetyczne)

Na `sin_homo` dane są generowane zgodnie z założeniami GP z jądrem RBF, więc posterior
GP jest **dokładny z dokładnością do hiperparametrów jądra**. Służy jako odniesienie
dla kształtu niepewności epistemicznej w 1D.

### L2 — HMC jako posterior wzorcowy (kluczowe)

Na **małej** sieci (1 warstwa ukryta × 50 jednostek, ~200 parametrów) i małym zbiorze
(N=50 syntetyczne, N=200 podzbiór jednego UCI) posterior da się próbkować przez
**NUTS** i traktować jako gold standard.

- Biblioteka: `numpyro` (szybsze niż `pyro`), 4 łańcuchy × 2000 próbek, warm-up 1000
- Diagnostyka **obowiązkowa**: `r_hat < 1.01`, ESS > 400 na parametr; jeśli nie zbiega,
  raportować to zamiast udawać, że wynik jest wzorcem
- **Ta sama architektura i ten sam prior** co pozostałe metody w tym eksperymencie
- Metryki porównania: per-punkt `|sigma_epi_method − sigma_epi_HMC|`,
  korelacja rangowa Spearmana między nimi, oraz stosunek średnich szerokości pasm

**Uwaga do tekstu pracy:** rozdział 2.6 zawiera zdanie *„MCMC methods were omitted in this
work"*. Jeśli HMC wchodzi jako referencja, to zdanie wymaga korekty — HMC nie jest wtedy
metodą porównywaną, tylko punktem odniesienia, dokładnie w tej samej roli co GP.
Zgłosić autorowi, nie zmieniać tekstu samodzielnie.

### L3 — brak ground truth, ale testowalne uporządkowanie

Gap split i OOD: nie znamy prawdziwej niepewności, ale wiemy, jak powinna się zachować.
Raportować stosunek `mean(sigma_epi | in-gap) / mean(sigma_epi | in-range)`.
Wartość ≈ 1 oznacza, że metoda nie wykrywa luki.

---

## 7. Metryki

Wszystkie w `src/metrics.py`, wszystkie liczone w jednostkach oryginalnych.

### 7.1 Dokładność
`RMSE`, `MAE`

### 7.2 Jakość rozkładu predykcyjnego
- **`LL` — average test log-likelihood, w konwencji literatury: wyżej = lepiej,
  wartości ujemne.** To jest kolumna zestawialna z opublikowanymi tabelami. Liczyć jako
  `-NLL` i raportować obie, ale w tabeli głównej pokazywać `LL`. Pomylenie znaku przy
  przepisywaniu cudzych liczb to klasyczny błąd — dodać test jednostkowy sprawdzający,
  że `LL = -NLL` co do znaku i że dla dobrze dopasowanego modelu `LL > LL_baseline`.
- `NLL` (gaussowski) — już jest, do użytku wewnętrznego
- `CRPS` — forma zamknięta dla gaussiana:
  `CRPS = sigma * (z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))`, `z = (y - mu)/sigma`.
  Proper scoring rule, mniej wrażliwy na ogony niż NLL — warto mieć obok.
- `PICP@95`, `MPIW@95` — już są
- `interval_score@95` (Winkler) — kara za szerokość plus kara za nietrafienie.
  Rozwiązuje problem MPIW: samo MPIW jest bez sensu, jeśli pokrycia się różnią.

### 7.3 Kalibracja (obecnie brakuje)
- Krzywa kalibracyjna kwantyli: dla `alpha ∈ {0.05, 0.10, ..., 0.95}`
  `empirical(alpha) = mean(y <= mu + sigma * Phi^{-1}(alpha))`
- `ECE_reg = mean_alpha |empirical(alpha) − alpha|` — jedna liczba do tabeli
- Zapisywać całą krzywę do osobnego CSV, żeby dało się narysować
- **`QICE`** (*quantile interval coverage error*) — bardzo bliski powyższemu, ale ma
  ustaloną nazwę w tej samej linii badań i został wprowadzony właśnie dlatego, że NLL
  zakłada gaussowską gęstość warunkową, co na danych rzeczywistych nie musi zachodzić.
  Sprawdzić definicję u Han i in. (2022) i użyć jej nazwy zamiast wymyślać własną —
  łatwiej się broni przed recenzentem. **Klucz bibliograficzny do zweryfikowania.**

> **Podział kolumn w tabeli głównej — do zaznaczenia w rozdziale 5.**
> Porównywalne z literaturą: **RMSE** i **LL**. Tylko te dwie występują w opublikowanych
> tabelach. PICP, MPIW, interval score, CRPS, ECE/QICE to wkład własny bez wierszy
> referencyjnych. Nazwać ten podział jawnie: dwie kolumny walidują implementację,
> reszta rozszerza porównanie.

### 7.4 Rozbicie niepewności — liczone zawsze, raportowane wybiórczo

**Zasada:** te metryki są **liczone i zapisywane do CSV w każdym eksperymencie**, ale
**do tabeli głównej w rozdziale 5 nie trafiają**. Powód nie jest techniczny, tylko
interpretacyjny: na UCI nie ma ground truth dla żadnego z dwóch członów, więc kolumna
`mean_var_epistemic` jest liczbą, o której nie da się powiedzieć, czy niska wartość
oznacza „lepiej", czy „metoda zapada się do punktu". W literaturze porównawczej
(Hernández-Lobato & Adams, Gal, Lakshminarayanan, Ovadia, Daxberger) tych kolumn nie ma.

Zapisywać mimo to, bo koszt jest zerowy, a bez zapisu nie da się ich odzyskać bez
powtórzenia treningu — i bo służą jako **wewnętrzna diagnostyka poprawności**:
`epi_ratio ≈ 0` na wszystkich zbiorach to sygnał błędu implementacji, nie wynik.

Metryki:

- `mean_var_aleatoric`, `mean_var_epistemic`
- `epi_ratio = mean_var_epistemic / mean_var_total`
- `epi_extrap_ratio` = średnia `sigma_epi` poza zakresem treningowym / wewnątrz
- `epi_gap_ratio` = średnia `sigma_epi` w luce / w obszarze gęstym (gap split)

**Uwaga: `alea_gap_ratio` odpada.** Przy jednolitym modelu homoskedastycznym człon
aleatoryczny jest stały z konstrukcji, więc diagnostyka przecieku dekompozycji nie ma
zastosowania. To jest **korzyść uboczna** tej decyzji, warta jednego zdania w rozdziale 4:
mechanizm, w którym rzadkie dane pozwalają obniżyć stratę przez zawyżenie `sigma^2(x)`
(z `eq:gaussian-nll`), nie może wystąpić, gdy `sigma` nie zależy od `x`. Człon aleatoryczny
nie ma jak wchłonąć epistemicznego.

Oczekiwane zachowanie: `epi_gap_ratio > 1`. Wartość bliska 1 oznacza, że metoda nie
wykrywa luki — i jest to wynik do zaraportowania, nie usterka do wystrojenia.

**Gdzie te liczby trafiają do pracy:** wyłącznie w sekcji analitycznej rozdziału 5,
na podstawie E1 (gdzie `sigma(x)` jest znane dokładnie), E3 (gap split) i E4 (odniesienie
do HMC — tam porównywany jest **tylko** człon epistemiczny, bo tylko on odróżnia jakość
przybliżeń posteriora). Nigdy jako dodatkowe kolumny tabeli UCI.

### 7.5 Koszt
`train_time_s`, `predict_time_ms_per_1k`, `n_parameters`, `peak_memory_mb`

---

## 8. Format wyników

Jeden plik na eksperyment, `results/{experiment_id}.csv`, długi format, stały schemat:

```
experiment_id, dataset, method, config_id, split_index, init_seed,
n_train, n_test, split_type,          # "random" | "gap_in" | "gap_out" | "extrapolation"
rmse, mae, ll, nll, crps, picp95, mpiw95, interval_score95, ece_reg, qice,
mean_var_aleatoric, mean_var_epistemic, epi_ratio,
train_time_s, predict_time_ms_per_1k, n_parameters,
timestamp, git_commit
```

**CSV przechowuje wszystko, tabela w pracy jest podzbiorem.** To są dwie różne decyzje
i nie należy ich mylić: zapis jest tani i nieodwracalny w drugą stronę, więc do CSV idzie
komplet metryk z sekcji 7. Wybór kolumn do rozdziału 5 jest decyzją redakcyjną,
podejmowaną osobno dla każdej tabeli — patrz zasada z sekcji 7.4. Nie usuwać kolumn
ze schematu CSV „bo nie będą raportowane".

Osobno:
- `results/calibration_curves.csv` — `(experiment_id, dataset, method, split_seed, alpha, empirical)`
- `results/predictions_1d/{dataset}_{method}.csv` — `(x, mean, std_alea, std_epi, y_true)`
  dla danych syntetycznych, żeby rysunki dały się przerysować bez powtarzania treningu
- `results/hmc_reference/` — próbki i diagnostyka zbieżności

Dopisywanie do CSV, nie nadpisywanie. Kolumna `git_commit` z `git rev-parse --short HEAD`.

---

## 9. Eksperymenty

| ID | Nazwa | Zakres | Wynik |
|---|---|---|---|
| **E0** | GP scaling | dokładny GP, `N ∈ {250, 500, 1000, 2000, 4000, 8000}`, 3 powtórzenia | Zmierzone uzasadnienie pominięcia GP na największym zbiorze |
| **E1** | Synthetic 1D | 3 warianty × 6 metod | Rysunki na wspólnych osiach + tabela L0 |
| **E2** | UCI benchmarks | 6 zbiorów × 6 metod × 20 podziałów, protokół z sekcji 5.2 | Tabela główna + walidacja implementacji |
| **E3** | Gap split | `sin_gap` + 2 UCI × 6 metod | `epi_gap_ratio` |
| **E4** | HMC reference | `sin_homo` + 1 mały UCI × 5 metod vs HMC | Odległość od posteriora wzorcowego |
| **E5** | Depth ablation | głębokość ∈ {1, 2, 4} × wszystkie metody × 2 zbiory | Wpływ topologii |
| **E6** | Ablacje metod | patrz niżej | Rysunki 3.7 i 3.9 |

**E6 rozbija się na cztery tanie ablacje:**

- `E6a` — liczba przebiegów `T ∈ {2,5,10,20,50,100,200,500}` dla BBB i MCD,
  po 10 losowań masek na punkt → **Rysunek 3.7** (uzasadnia wybór `T`, obecnie arbitralny)
- `E6b` — rozmiar ensemble `M ∈ {2,3,5,10}` → gdzie się nasyca
- `E6c` — struktura kowariancji Laplace: `full`/`kron`/`diag` × `prior_precision`
  {nieregularyzowany, strojony} → **Rysunek 3.9**
- `E6d` — funkcja aktywacji ReLU vs TanH dla MCD na `sin_homo` → test przewidywania
  gal2016 o nieograniczonym wzroście niepewności dla ReLU

Każdy eksperyment: osobny skrypt `experiments/e{N}_{nazwa}.py` z `argparse`,
plus `--quick` uruchamiające zredukowaną wersję (2 podziały, mniej epok) do testów.

---

## 10. Rysunki

Nazwy plików **muszą zgadzać się z tym, co jest już w `.tex`** — katalogi mają literówkę
(`rodzial2_rys`, nie `rozdzial2_rys`), zachować ją albo zgłosić autorowi do poprawy
w obu miejscach naraz.

| Plik | Zawartość | Źródło |
|---|---|---|
| `rodzial2_rys/img2_1.png` | heteroskedastyczna aleatoryczna, pasmo ±2σ | **osobny skrypt, sieć dwugłowicowa** |
| `rodzial2_rys/img2_2.png` | próbki funkcji z posteriora, rozchodzące się poza `[0,6]` | E1 |
| `rodzial2_rys/img2_3.png` | dwa zagnieżdżone pasma: aleatoryczne i całkowite | E1 |
| `rodzial3_rys/img3_1.png` | próbki z GP prior dla `ell ∈ {0.3, 1.0, 3.0}` | skrypt osobny |
| `rodzial3_rys/img3_2.png` | posterior GP na `sin_homo` | E1 |
| `rodzial3_rys/img3_5.png` | posterior BBB | E1 |
| `rodzial3_rys/img3_6.png` | posterior MCD | E1 |
| `rodzial3_rys/img3_7.png` | stabilność względem `T` | E6a |
| `rodzial3_rys/img3_8.png` | posterior Laplace | E1 |
| `rodzial3_rys/img3_9.png` | struktura kowariancji Laplace, 3 panele | E6c |
| `rodzial3_rys/img3_10.png` | **posterior deep ensembles** | E1 |
| `figures/e1_map.png` | baseline MAP, poza katalogami pracy | E1 |

**`img3_10` i rozdział 3.5.** Sekcja o deep ensembles nie jest jeszcze napisana, stąd brak
rysunku w istniejącym `.tex`. `img3_10` to pierwszy wolny numer po 3.4. Baseline MAP nie
ma własnej sekcji w rozdziale 3, więc jego panel trafia poza katalogi pracy — służy ocenie
wzrokowej i ewentualnie rozdziałowi 5.

**`img3_3` jest wolny** — luka w numeracji rozdziału 3. Sprawdzić, czy nie ma po nim śladu
w `.tex` (podpisu, odwołania, zakomentowanego bloku).

**Wymóg krytyczny:** rysunki 3.2, 3.5, 3.6, 3.8 i 3.10 muszą mieć **identyczne osie
i identyczną skalę** — to jest główny argument wizualny rozdziału 3. Zaimplementować jako
jedną funkcję z wymuszonym `xlim`/`ylim` z konfiguracji (`src/style.py`, nazwana stała),
nie ustawianą per rysunek. Wszystkie na `seed=0`, udokumentowanym w kodzie. Siatka
ewaluacyjna minimum 1000 punktów — igła Laplace'a ma szerokość ~0.02, więc przy rzadszej
siatce trafienie w nią zależałoby od rozmieszczenia punktów.

**`figures/redraw.py`** — skrypt czytający **wyłącznie** z `results/predictions_1d/`,
bez importowania metod i bez treningu. Argumenty: lista metod, wariant, tryb (osobne
panele / jeden wykres z nałożonymi pasmami / siatka). Pozwala złożyć dowolną kombinację
rysunków bez dotykania pipeline'u eksperymentów.

Pliki w `results/predictions_1d/` muszą zawierać komplet potrzebny do przerysowania:
`x, mean, std_aleatoric, std_epistemic, y_true` oraz dane treningowe (`x_train`, `y_train`)
— te ostatnie w osobnym pliku na wariant, jeśli tak wygodniej.

Rysunki `img2_3_1.png` (schemat przestrzeni hipotez) i `img3_4.png` (waga jako rozkład)
są koncepcyjne — **nie generować z danych**, autor rysuje je osobno.

Format: PNG 300 dpi, `matplotlib` bez seaborn, jedna paleta zdefiniowana w `src/style.py`,
stały kolor na metodę we wszystkich rysunkach.

---

## 11. Weryfikacja względem literatury

Praca ma być porównaniem, a nie replikacją, więc **rozbieżność z literaturą jest wynikiem,
a nie błędem do ukrycia**. Ale rozbieżność bez wyjaśnienia zwykle oznacza błąd implementacji,
dlatego każde przewidywanie z literatury ma być sprawdzane automatycznie.

Wygenerować `results/expectations_check.csv` ze schematem
`(id, prediction, metric, expected_relation, observed, verdict)`, gdzie `verdict ∈
{confirmed, refuted, inconclusive}`.

Przewidywania do sprawdzenia:

| # | Przewidywanie | Metryka | Źródło (**zweryfikować przed cytowaniem**) |
|---|---|---|---|
| P1 | Baseline deterministyczny ma PICP wyraźnie poniżej 0.95 | `picp95[map] < 0.95` | teza o overconfidence, rozdz. 2.1.2 |
| P2 | BBB ma węższe pasma niż GP i pokrycie poniżej nominalnego | `mpiw[bbb] < mpiw[gp]`, `picp[bbb] < 0.95` | blundell2015, arbel2023 |
| P3 | MCD jest przeuwiarygodniony na interpolacji, GP przeszacowuje | `picp[mcd] < 0.95 < picp[gp]` | gal2016appendix |
| P4 | MCD z ReLU: niepewność rośnie bez ograniczenia poza danymi; z TanH — ograniczona | `epi_extrap_ratio[relu] >> [tanh]` | gal2016 |
| P5 | Laplace pełny bez regularyzacji przeszacowuje niepewność przy N=50 i tysiącach parametrów | `mpiw[la_full_unreg] >> mpiw[la_full_tuned]` | ritter2018 |
| P6 | Laplace diagonalny zachowuje się podobnie do dropoutu; KFAC daje wyższą niepewność OOD | `epi_extrap_ratio[kron] > [diag]` | ritter2018 |
| P7 | Laplace strojony: pokrycie **powyżej** nominalnego, szersze pasma, wolniejszy wzrost niż GP | `picp[la] >= 0.95`, `epi_extrap_ratio[la] < [gp]` | ritter2018, daxberger2021 |
| P8 | Laplace zachowuje dokładność sieci MAP dokładnie | `rmse[la] == rmse[map]` (do 1e-10) | daxberger2021 |
| P9 | Deep ensembles: najlepszy lub bliski najlepszemu NLL wśród przybliżeń; lepiej skalibrowany niż MCD | `ece[de] < ece[mcd]` | lakshminarayanan2017 |
| P10 | Metody mean-field (BBB, MCD) nie podnoszą niepewności w luce; GP i DE podnoszą | `epi_gap_ratio[bbb], [mcd] ≈ 1` | foong2019 — **koniecznie zweryfikować** |
| P11 | Uporządkowanie kosztów: LA ≈ MAP < MCD < BBB < DE(×M); GP rośnie jak `O(N^3)` | `train_time_s`, `n_parameters` | daxberger2021 + pomiar własny |
| P12 | Przewaga GP maleje wraz z wymiarem wejścia | `rmse[gp] − rmse[de]` rosnące z `d` | rozdz. 3.1.6 |
| P13 | **Własna implementacja odtwarza opublikowane liczby** dla MCD, BBB i DE w granicach odchylenia standardowego | `\|rmse_own − rmse_published\| < sd_published` dla RMSE i LL | gal2016, blundell2015, lakshminarayanan2017 |
| P14 | Czas dopasowania dokładnego GP rośnie superkwadratowo | **ROZSTRZYGNIĘTE (E0):** nachylenie 2,31 (bez restartów) i 2,23 (pełna konfiguracja), nie 3 | E0, pomiar własny |

**P14 — wynik i korekta sformułowania.** Pierwotnie zakładano nachylenie bliskie 3, zgodnie
z asymptotyką `O(N^3)` rozkładu Choleskiego. Zmierzone wartości to 2,31 i 2,23 — spójne
ze sobą, co potwierdza sensowność rozdzielenia kosztu algebraicznego od strojenia
hiperparametrów, ale wyraźnie poniżej 3.

Prawdopodobna przyczyna: BLAS zrównolegla rozkład Choleskiego, a przy małych `N` nie wysyca
dostępnych rdzeni, więc mierzony jest reżim przejściowy, nie asymptotyczny. **Nie naginać
wyniku do 3.** W rozdziale 4 zapisać: teoretyczne `O(N^3)`, zmierzone `~N^2,3` na jednej
maszynie.

**Mocniejszy argument z E0 to wariancja czasu, nie nachylenie.** Identyczna konfiguracja
przy `N = 8000`: raz nie ukończyła ani jednego powtórzenia w ponad 2 godzinach, raz
ukończyła w 42 minutach. Przy 20 podziałach na największych zbiorach UCI rozrzut kumuluje
się w sposób nieprzewidywalny — to jest uzasadnienie pominięcia GP na `power_plant`,
mocniejsze niż jakakolwiek asymptotyka.

**Pamięć:** stosunek zmierzonego szczytu do teoretycznej macierzy jądra zbiega do **9,000**
aż po `N = 8000` (512 MB teoretyczne wobec 4608 MB rzeczywistych). `sklearn` trzyma
jednocześnie około dziewięciu macierzy `N × N`. Konkret, którego nie podaje żadna
z prac referencyjnych.

**P13 jest najważniejsze i wykonuje się jako porównanie sparowane.**

Katalog `results/` w repo referencyjnym zawiera **wartości per podział**, po 20 liczb na
zbiór, osobno dla RMSE i LL. Ponieważ używamy **tych samych plików indeksów**, wartość dla
podziału `i` dotyczy dokładnie tego samego zbioru testowego u nas i u Gala.

Konsekwencja: zamiast porównywać dwie średnie ze słupkami błędu, liczymy **rozkład różnic**
`own[i] − published[i]` po 20 podziałach. Wariancja między podziałami — na małych zbiorach
UCI duża — znika z porównania, więc test jest o rząd wielkości ostrzejszy.

`results/literature_comparison.csv`: `(dataset, method, metric, split_index, own_value,
published_value, difference)`. Podsumowanie osobno: średnia różnica, jej standard error,
udział podziałów z różnicą tego samego znaku.

**Właściwe pliki: `test_MC_rmse_*`, nie `test_rmse_*`.** Ten drugi zawiera wynik pojedynczego
deterministycznego przebiegu i nie odpowiada tabeli w README (dla `concrete`: 5.45 zamiast
4.82). Tabela referencyjna dotyczy MC dropout z próbkowaniem.

**Nie ufać zaokrąglonym komórkom README.** Dla `energy`/RMSE podana wartość `± 0.06`
odpowiada odchyleniu standardowemu (0.0605), podczas gdy wszystkie pozostałe komórki są
spójne ze standard error. Przyczyna nierozstrzygnięta. Kolejny powód, żeby liczyć wszystko
z plików per podział, a README traktować wyłącznie jako kontrolę zgodności średnich.

Wiersze dla BBB i deep ensembles wpisuje **autor ręcznie po odczytaniu z prac**.
Claude Code ich nie wypełnia i nie zgaduje.

Jeżeli P13 zawiedzie, sprawdzić w tej kolejności: czy użyto podziałów z plików indeksów,
czy porównanie idzie względem `test_MC_rmse`, liczba epok i procedura strojenia (sekcja 5.2),
architektura (1×50), standaryzacja `y` i jej odwrócenie dla `sigma`, znak `LL`, konwencja
priora (sekcja 4.6). Jeżeli różnica idzie w tę samą stronę na wszystkich podziałach
i wszystkich zbiorach, to jest błąd protokołu, nie własność metody.

**Żaden z tych kluczy bibliograficznych nie jest jeszcze zweryfikowany poza tymi,
które występują w `.tex`.** `lakshminarayanan2017`, `foong2019`,
`hernandez-lobato2015` — sprawdzić dokładne dane bibliograficzne przed wpisaniem
do `.bib`. Nie dopisywać niczego do bibliografii samodzielnie.

Jeśli przewidywanie zostanie obalone: **nie stroić hiperparametrów, żeby wyszło zgodnie
z literaturą.** Zamiast tego zapisać obserwację i sprawdzić trzy najczęstsze przyczyny:
niezgodność priorów (sekcja 4.6), inna skala targetu, za mała liczba podziałów.

---

## 11. Precyzja numeryczna

**`float64` we wszystkich metodach i we wszystkich eksperymentach.** Sieci mają 151–550
parametrów, największy zbiór 8611 punktów — koszt pomijalny, a znika cała klasa pytań
o to, czy obserwowany efekt jest numeryczny.

Powód konkretny: macierz precyzji posteriora Laplace'a ma współczynnik uwarunkowania
rzędu `10^6` niezależnie od precyzji — to własność problemu, nie artefakt. Przy `float32`
zostaje wtedy około jednej cyfry znaczącej i wartości wariancji nie są wiarygodne.

**`bayesian-torch` nie przyjmuje `dtype`.** `LinearReparameterization.__init__` tworzy
tensory według **globalnego domyślnego dtype**, a `dnn_to_bnn` go nie przekazuje.
Jedyny punkt kontroli to `torch.set_default_dtype(torch.float64)` wywołane **przed**
konstrukcją modelu — umieszczone w `set_seed()`, bo to jedyna funkcja wołana przed
budową modelu w każdej metodzie.

Nie używać `.double()` po konstrukcji: inicjalizacja w `float32` z późniejszym rzutowaniem
pobiera inny ciąg z generatora niż budowa wprost w `float64`, więc trening z tym samym
seedem zbiega do innej sieci. Błąd wykryty przy diagnostyce Laplace'a; wart odnotowania,
bo jest cichy i daje pozornie sensowne wyniki.

## 12. Kolejność pracy i punkty kontrolne

Nie realizować całości jednym ciągiem. Po każdym etapie **zatrzymać się i zdać raport**:
co działa, jakie liczby wyszły, co budzi wątpliwości.

> **Stan na dziś: Etapy 1–3 zamknięte.** Szkielet, dane z podziałami literaturowymi,
> sześć metod w rejestrze, E1 policzone, rysunki wygenerowane. Bieżący etap: 4.

- **Etap 1 — szkielet.** ZAMKNIĘTY. `Prediction`, `UncertaintyMethod`, rejestr metod, `config.py`
  z mapowaniem prior ↔ weight decay, poprawki w `data.py` (standaryzacja `y`, warianty
  syntetyczne, gap split), rozbudowane `metrics.py` z testami jednostkowymi na
  syntetycznych przypadkach o znanej odpowiedzi. **Bez metod.** → raport
- **Etap 2 — metody.** Sześć klas realizujących protokół + testy: każda musi zwracać
  poprawne kształty, nieujemne wariancje, a `map` zerowy człon epistemiczny.
  Test P8 (Laplace vs MAP) jako assert. → raport

  > **PUNKT DECYZYJNY — ROZSTRZYGNIĘTY.** Sprawdzono empirycznie na `laplace-torch==0.2.3`:
  > subnetwork Laplace na głowicy średniej działa, ale wektorowe `sigma_noise` jest twardo
  > odrzucane (`baselaplace.py`: *„Only homoscedastic output noise supported"*), a klasa
  > `KronSubnetLaplace` nie istnieje. Model heteroskedastyczny odpada.
  >
  > **Decyzja: jednolity model homoskedastyczny** we wszystkich metodach sieciowych
  > (sekcja 4). Wymaga edycji w `rozdzial21.tex` (l. 42), `rozdzial22.tex` (l. 80–91, 125),
  > `rozdzial31.tex` (l. 155), `rozdzial32.tex` (l. 186–210), `rozdzial33.tex` (l. 156–199)
  > i `rozdzial34.tex` (l. 231–245). **Edycje wykonuje autor — nie tykać plików `.tex`.**
- **Etap 3 — E1 i rysunki.** Uruchomić na danych syntetycznych, wygenerować komplet
  rysunków na wspólnych osiach. To jest moment na ocenę wzrokową — jeśli któraś metoda
  wygląda źle, jest błąd, a nie wynik. → raport z rysunkami
- **Etap 4 — E0 i E2.** Najpierw E0 (tani, daje uzasadnienie polityki wobec GP), potem
  pełna tabela UCI: `--quick` na 2 podziałach, dopiero po weryfikacji pełne 20.
  Wygenerować `literature_comparison.csv`, wypełniając wiersze MC dropout z tabeli
  w sekcji 11 i zostawiając puste dla BBB i deep ensembles. → raport

  > **PUNKT DECYZYJNY — liczba epok w tabeli głównej.** Protokół referencyjny (4000 epok
  > + grid search per fold) jest odtwarzany wyłącznie w osobnym przebiegu walidacyjnym
  > dla MC dropout na 2–3 najmniejszych zbiorach, zgodnie z sekcją 5.2. Dla tabeli głównej
  > wyznaczyć liczbę epok empirycznie — punkt, w którym strata walidacyjna przestaje
  > spadać na najtrudniejszym zbiorze — i przedstawić autorowi propozycję z wykresem,
  > zanim ruszy pełne E2. Nie wybierać samodzielnie.
- **Etap 5 — E3, E5, E6.** → raport
- **Etap 6 — E4 (HMC).** Osobno, bo wymaga `numpyro` i najwięcej czasu. → raport
- **Etap 7 — `expectations_check.csv`** i podsumowanie rozbieżności. → raport

Na koniec każdego etapu: `results/` zsynchronizowane, README z instrukcją odtworzenia
każdego eksperymentu jedną komendą.

---

## 13. Czego nie robić

- Nie stroić hiperparametrów pod metrykę raportowaną — jeśli strojenie jest potrzebne,
  robić je na osobnym zbiorze walidacyjnym i udokumentować procedurę identyczną
  dla wszystkich metod.
- Nie dodawać metod spoza listy pięciu (SWAG, SGLD, ewidencyjne sieci) bez pytania —
  rozdział 3 ich nie opisuje, więc w wynikach nie mają czego szukać.
- Nie zmieniać architektury per metoda „żeby lepiej działało".
- Nie raportować MPIW bez PICP obok.
- Nie commitować `.ipynb`, wag modeli ani wykresów PDF.
- Nie edytować plików `.tex` pracy — zgłaszać potrzebne zmiany w raporcie.
- Nie zastępować biblioteki z tabeli w sekcji 1.1 własną implementacją bez pytania,
  nawet jeśli własna wydaje się prostsza. Każda własna implementacja algorytmu wnosi
  decyzje, które trzeba obronić w rozdziale 4.
- Nie dodawać zależności spoza sekcji 1.1 (wyjątek: `numpyro` dla E4).
