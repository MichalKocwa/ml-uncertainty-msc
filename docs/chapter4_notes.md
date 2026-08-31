# Errata do rozdziałów 2–3 oraz materiał do rozdziału 4

> Notatka robocza. Powstaje równolegle z implementacją, żeby uzasadnienia decyzji
> nie musiały być odtwarzane z pamięci przy pisaniu rozdziału 4.
>
> Zasada: **każda decyzja bez źródła w literaturze wymaga w rozdziale 4 jawnego akapitu
> „X was adopted because Y"**. Poniższa lista jest kompletnym zestawem takich decyzji
> na dzień ostatniej aktualizacji.

---

# CZĘŚĆ A — Errata: zmiany wymagane w napisanym tekście

Wszystkie wynikają z jednej decyzji: **przejścia na jednolity model homoskedastyczny**
(uzasadnienie w części B, punkt D1).

## A.1 Tabela zmian

| Plik | Linie | Charakter zmiany | Sprawdzone |
|---|---|---|---|
| `rozdzial21.tex` | 42 | zdanie „The heteroscedastic model … is the one adopted in the experiments of this work" — odwrócić kierunek | grep |
| `rozdzial22.tex` | 80–91, 125 | wiarygodność zapisana dla dwóch głowic — przeformułować na jedno wyjście + globalna wariancja | grep |
| `rozdzial31.tex` | 155 | uzasadnienie, dlaczego GP jest wyjątkiem — **ten fragment można skrócić lub usunąć**, wyjątek znika | grep |
| `rozdzial32.tex` | 186–210 | estymator BBB z `σ²_w(x*)` z drugiej głowicy | grep |
| `rozdzial33.tex` | 156–199 | wariant Kendalla i Gala + zdanie „adopted in the present work" | przeczytane |
| `rozdzial33.tex` | 221 | podpis Rys. 3.6 odwołuje się do `eq:mc-dropout-heteroscedastic` | przeczytane |
| `rozdzial34.tex` | 231–245 | to samo dla Laplace'a | grep |
| `rozdzial22.tex` lub 2.6 | ? | zdanie „MCMC methods were omitted in this work" — **tylko jeśli wejdzie E4 (HMC)** | do sprawdzenia |

Pozycje oznaczone „grep" wymagają przeczytania kontekstu przed edycją — znam ich
istnienie z wyszukiwania, nie z lektury.

## A.2 Fragment kluczowy — `rozdzial33.tex`, l. 177–184

Obecny argument brzmi (parafraza): wariant homoskedastyczny odpada, bo uczyniłby człon
aleatoryczny jednej metody dopasowaną stałą, a drugiej wyuczoną funkcją, więc różnica
w rozdziale 5 dałaby się przypisać modelowi szumu, a nie przybliżeniu posteriora.

**Ten argument przetrwał zmianę bez szwanku.** Dotyczy *symetrii* modelu szumu między
metodami, nie *heteroskedastyczności*. Jednolicie homoskedastyczny setup spełnia go
tak samo. Zmienia się kierunek unifikacji, nie zasada.

Do przeredagowania: zdanie o przyjęciu wariantu Kendalla i Gala. Równanie
`eq:mc-dropout-heteroscedastic` **zostaje** — jest częścią opisu metody w literaturze,
przestaje być tylko wariantem uruchamianym.

## A.3 Co zyskuje tekst

Nie tylko strat. Warto odnotować przy redakcji:

- **Wyjątek GP znika.** `rozdzial31.tex` l. 155 musi obecnie tłumaczyć, dlaczego GP stoi
  w opozycji do sformułowania z 2.1. Przy jednolitej homoskedastyczności wszystkie sześć
  metod dzieli jeden model szumu i ten akapit staje się zbędny.
- **Znika ryzyko przecieku dekompozycji.** Mechanizm z `eq:gaussian-nll`, w którym rzadkie
  dane pozwalają obniżyć stratę przez zawyżenie `σ²(x)`, nie może wystąpić, gdy `σ`
  nie zależy od `x`. Człon aleatoryczny nie ma jak wchłonąć epistemicznego. Jedno zdanie
  w rozdziale 4, mocny argument.

## A.4 Rysunki wymagające osobnego traktowania

`img2_1.png` (heteroskedastyczna niepewność aleatoryczna) nie da się wygenerować
z modelu porównania głównego. Osobny skrypt z siecią dwugłowicową, poza pipeline'em
eksperymentów. Zaznaczyć w podpisie, że ilustruje sformułowanie ogólne z 2.1,
a nie konfigurację użytą w rozdziale 5.

---

# CZĘŚĆ B — Materiał do rozdziału 4

Układ odpowiada proponowanej strukturze rozdziału. Dla każdej sekcji: co trzeba napisać
i jakie uzasadnienie jest już ustalone.

## 4.1 Experimental design

### Co ma się tu znaleźć
Sedno rozdziału: **co jest zmienną eksperymentu, a co stałą**. Bez tego czytelnik nie ma
jak sprawdzić, czy porównanie jest uczciwe.

### Tabela do wstawienia wprost

| Element | Status | Uwaga |
|---|---|---|
| Zbiór danych, podziały, seedy | stałe | podziały z plików indeksów, identyczne dla wszystkich metod |
| Preprocessing (standaryzacja X i y) | stałe | |
| Architektura backbone'u | stałe | 1×50 TanH (D7/D7b — było ReLU, patrz D7b), poza Study III |
| Model szumu | stałe | homoskedastyczny, jeden globalny `log σ²` |
| Prior `N(0, γ²I)` | stałe | ta sama `γ` we wszystkich metodach |
| Optymalizator, learning rate, liczba epok | stałe | |
| Liczba próbek predykcyjnych `T` | stałe | poza deep ensembles, gdzie `T = M` |
| **Sposób aproksymacji posteriora** | **zmienna** | jedyna |

### Trzy miejsca, gdzie symetria się łamie — każde wymaga akapitu

**D-A. Gaussian process nie ma architektury.** Wnioskowanie dokładne, brak parametrów
w sensie sieciowym. Rola: *reference point*, nie równorzędny uczestnik. Argument już
istnieje w 3.1.

**D-B. Deep ensembles dostaje `M`-krotny budżet.** Wybór *equal architecture*, nie
*equal compute*. Uzasadnienie: równanie budżetu wymagałoby zmiany szerokości sieci,
co złamałoby zgodność z protokołem referencyjnym. Konsekwencja: **koszt musi być
raportowany jako osobna metryka**, inaczej porównanie jest przechylone.

**D-C. Liczba próbek `T`.** BBB i MC dropout mogą losować `T = 100`; ensemble ma `T = M = 5`.
Nie zrównywane. Uzasadnienie przez E6a, który pokazuje, gdzie estymator wariancji
się nasyca.

**D-D. Koszt GP (E0) — zmierzony, nie zakładany.** Dokładny GP (`GaussianProcessRegressor`,
`n_restarts_optimizer=5`, protokół jak w E1/E2), dane: proces generujący `sin_homo`
(`x ~ linspace(0,6,N)`, `sigma=0.1`), `N ∈ {250, 500, 1000, 2000, 4000, 8000}`, 3 powtórzenia
(seed=repeat) dla `N ≤ 4000`.

*Zużycie pamięci — dwie kolumny, `tracemalloc` (stdlib) zamiast `psutil` (nie ma go w
tabeli 1.1, decyzja użytkownika tej sesji):*

| N | kernel_matrix_mb (teoretyczne, `8N²/1e6`) | peak_memory_mb (`tracemalloc`) | stosunek |
|---|---|---|---|
| 250 | 0,5 | 4,5–4,6 | ~9,1–9,2 |
| 500 | 2,0 | 18,04 | ~9,02 |
| 1000 | 8,0 | 72,05 | ~9,006 |
| 2000 | 32,0 | 288,07 | ~9,002 |
| 4000 | 128,0 | 1152,10 | ~9,0007 |
| 8000 | 512,0 | 4608,23 | 9,0004 |

Stosunek zbiega do 9,0 przy rosnącym `N` — konkretny wniosek zamiast ogólnika o `O(N²)`
pamięci: `GaussianProcessRegressor` trzyma jednocześnie w pamięci równowartość **dziewięciu**
macierzy `N×N` (Gram, jej odwrotność/faktoryzacja Choleskiego, bufory pośrednie
optymalizatora hiperparametrów per restart itd.) — nie jedną.

*Czas dopasowania — problem z pierwszej wersji pomiaru i poprawka.* Pierwszy pomiar
(`fit_time_s`, pełna konfiguracja `n_restarts_optimizer=5`) pokazał ~2× rozrzut między
powtórzeniami przy identycznym `N` (np. `N=4000`: 530s / 602s / 295s) — nie szum, tylko
zmienna liczba kroków L-BFGS do zbieżności, różna dla każdego z 6 restartów (1 startowy +
5 losowych). To osłabiało P14 (nachylenie `log(t)` względem `log(N)` powinno być bliskie 3):
metryka mieszała koszt algebraiczny z nieprzewidywalną liczbą iteracji optymalizatora.

Poprawka (`experiments/e0_gp_scaling.py --norestart`, `n_restarts_optimizer=0` — jedna
próba optymalizacji zamiast sześciu): izoluje koszt algebraiczny, osobny plik
`results/e0_gp_scaling_norestart.csv`, nie miesza się z `fit_time_s`.

| N | fit_time_norestart_s (3 powt., średnia) | fit_time_s (pełna config, 3 powt., średnia) | mnożnik pełna/norestart |
|---|---|---|---|
| 250 | 0,128 | 1,123 | ~8,7 |
| 500 | 0,551 | 3,336 | ~6,3 |
| 1000 | 2,852 | 14,292 | ~5,0 |
| 2000 | 13,935 | 75,533 | ~5,4 |
| 4000 | 69,317 | 475,498 | ~6,9 |
| 8000 | 381,0 (średnia 3 powt.: 383,9 / 382,6 / 376,7) | 2533,7 (1 powtórzenie) | 6,65 |

Mnożnik nie ma czystego trendu z `N` (waha się ~4,2–9,6 na pojedyncze powtórzenie,
średnio ~5–8,7× per `N`, `N=8000` na 6,65×) — zgodne z tym, że to jest artefakt liczby
restartów (6 prób), nie funkcja `N`. **Nachylenie `log(fit_time_norestart_s)` względem
`log(N)`, cała siatka `N=250..8000` (18 punktów) = 2,314** — **nieoczekiwane wobec P14
(oczekiwane blisko 3)**. Nachylenie z pełnej konfiguracji (`fit_time_s`, 16 punktów
łącznie z pojedynczym powtórzeniem `N=8000`) = 2,231 — zgodne co do rzędu wielkości z
wersją `norestart`, co samo w sobie potwierdza, że separacja obu kosztów (D-D wyżej)
była słuszna: obie metryki zgadzają się co do nachylenia, mimo że różnią się wielkością
o stały (choć hałaśliwy) mnożnik restartów. Lokalne nachylenia między kolejnymi `N`
(2,1–2,5) są spójne z wartością całościową na każdym segmencie, nie tylko na jednym —
nie jest to artefakt małych `N`. Możliwe wyjaśnienie (niesprawdzone, do odnotowania
jako obserwacja, nie wyjaśnienie przyjęte bez dowodu): BLAS/LAPACK-owa faktoryzacja
Choleskiego korzysta z wielowątkowości i efektów pamięci podręcznej silniej przy
większych `N`, co częściowo kompensuje teoretyczny wzrost sześcienny w mierzonym czasie
ściennym. **Do rozdziału 4: zgłosić nachylenie ~2,2–2,3 jako zmierzone, nie naginać
opisu do „bliskie 3" wbrew danym — P14 w tej formie nie potwierdza się w pełni.**

`N=8000`, pełna konfiguracja (`n_restarts_optimizer=5`): **pierwsza próba (3 powtórzenia)
nie ukończyła nawet jednego powtórzenia w >2h, proces przerwany** — sam ten fakt jest
częścią dowodu na koszt GP przy dużym `N`, niezależnie od drugiej próby. Druga próba
(1 powtórzenie zamiast 3, limit 3h, decyzja użytkownika): **ukończona w 2533,7s (42,2
min)** — znacząco szybciej niż pierwsza próba nawet zdążyła, bo trafiła na "łatwiejszy"
zestaw losowych restartów (patrz mnożnik 6,65× — w środku obserwowanego zakresu
4,2–8,7×, nic anomalnego w tym konkretnym powtórzeniu). Wniosek do rozdziału 4/5: sam
czas jednego (udanego) dopasowania GP przy `N=8000` to ~42 min, ale **wariancja między
powtórzeniami identycznej konfiguracji jest na tyle duża, że jedno powtórzenie może się
nie skończyć w rozsądnym czasie w ogóle** — to, nie sam nachylenie ~O(N^2,3), jest
mocniejszym praktycznym argumentem za wykluczeniem GP z E2 (20 podziałów × wielokrotne
restarty na największych zbiorach UCI, `N≈8600`, brief sekcja 4.1).

**P14 rozstrzygnięte (decyzja użytkownika, brief zaktualizowany).** Rozdział 4 zapisuje
teoretyczne `O(N^3)` (Cholesky) obok zmierzonego ~`N^2,3` na tej konkretnej maszynie —
bez naciągania opisu do „bliskie 3". Kolejność argumentacji za pominięciem GP na
`power_plant` (N=9568, `n_train=8611` — największy zbiór UCI, `src/data.py::UCI_SPEC`):
**najpierw wariancja czasu dopasowania** (>2h bez ukończenia vs 42 min przy identycznej
konfiguracji, `N=8000`) **jako główne uzasadnienie, dopiero potem nachylenie** ~N^2,3
jako uzupełnienie. Nie odwracać tej kolejności w tekście — nachylenie samo w sobie (2,3,
nie 3) jest słabszym argumentem niż nieprzewidywalność czasu wykonania.

## 4.2 Datasets

### Decyzje do uzasadnienia

**D2. Podziały z repozytorium referencyjnego, nie generowane samodzielnie.**
`yaringal/DropoutUncertaintyExps`, commit `6eb4497`, pliki `index_train_{0..19}.txt`.
README ostrzega, że ze względu na mały rozmiar zbiorów samodzielne dzielenie daje wyniki
nieporównywalne z opublikowanymi. Uzasadnienie: bez tego walidacja implementacji (P13)
jest niewykonalna.

Efekt uboczny: rozstrzyga cztery pytania, które inaczej byłyby arbitralne — target `Y1`
w `energy`, wariant red-only w `wine`, źródło `yacht`, zaokrąglenie w `kin8nm`.

**D3. Dobór zbiorów pod wykonalność dokładnego GP.**
`yacht`, `energy`, `concrete`, `wine_quality_red` — pełne. `kin8nm`, `power_plant` —
graniczne, dołożone po to, żeby pokazać granicę stosowalności GP.
Konsekwencja do nazwania wprost: **zbiory są z małego końca kanonicznej dziesiątki**,
więc wnioski o skalowalności są ograniczone.

**D4. Pominięcie `boston`.** Usunięty ze `scikit-learn` ze względu na cechę zakodowaną
rasowo. Konsekwencja: zawęża zestaw dostępnych wierszy referencyjnych.

**D5. GP nieuruchamiany na `power_plant`.** Komórka `—` z przypisem.
Sformułowanie: **„koszt nieproporcjonalny do wartości informacyjnej"**, nie
„metoda się nie skaluje" — przy `N ≈ 8600` dokładny GP jest wykonalny, problem
jest w iloczynie 20 podziałów × restartów optymalizatora. Poparte pomiarem z E0.
Każde stwierdzenie porównawcze o GP musi być warunkowane zakresem `N`.

**D6. Licencja danych.** CC BY-NC 4.0. Dane nie są wendorowane do repo — pobierane
skryptem z przypiętego commita, weryfikowane sumami SHA-256. Atrybucja w `docs/datasets.md`.

## 4.3 Common experimental protocol

**D7. Architektura 1×50, aktywacja TanH (nie ReLU).** Szerokość/głębokość (1×50) —
nie wybór swobodny, architektura protokołu Hernándeza-Lobato i Adamsa, wymagana dla
porównywalności; głębsze sieci wyłącznie w Study III, gdzie głębokość jest zmienną
badaną. Funkcja aktywacji jest jednak odseparowana od tej decyzji i **ustawiona na
TanH, nie ReLU** — to osobna decyzja tej sesji, patrz D7b: ReLU był domyślny (zgodnie
z H-L) do czasu, aż zdiagnozowany mechanizm (nieciągłość jakobianu ReLU w linearyzowanej
wariancji Laplace'a, część E) i sweep porównawczy ReLU/TanH pokazały, że TanH usuwa ten
mechanizm u źródła bez pogorszenia dopasowania in-range dla większości metod. ReLU
zostaje dostępny jako `activation="relu"` — ablacja E6d oraz wymóg P13 (patrz D7b).

**D7b. TanH jako aktywacja domyślna zamiast ReLU (2026-08-25) — decyzja, nie
wartość domyślna przyjęta bez sprawdzenia.** Wyjściowo (D7, brief) sieci używały ReLU
za protokołem H-L. Po zdiagnozowaniu, że skok `var_epistemic` Laplace'a pokrywa się co
do 6. miejsca po przecinku z załamaniami aktywacji ReLU (część E) — czyli jest realną
własnością linearyzowanego Laplace'a wokół sieci ReLU, nie błędem — sprawdzono hipotezę,
że sieć o ciągłym jakobianie (TanH) usuwa ten mechanizm u źródła, a nie tylko go wygładza.
Sweep: `γ=1,0` (D11b), 6 metod × 3 seedy × 3 zbiory (`sin_homo`, `sin_hetero`, `sin_gap`),
ReLU vs TanH, pełne przeliczenie E1 dla obu.

*Skok Laplace'a — max |Δvar_epistemic| na siatce (średnia / max po 3 seedach):*

| zbiór | ReLU | TanH | redukcja |
|---|---|---|---|
| sin_homo | 1,283 / 3,641 | 0,0130 / 0,0154 | ~99× |
| sin_hetero | 6,827 / 8,501 | 0,0137 / 0,0151 | ~500× |
| sin_gap | 0,885 / 1,576 | 0,0121 / 0,0137 | ~73× |

Wartość resztkowa pod TanH (~0,01–0,015) to szum gęstości siatki, nie struktura skoku —
mechanizm zniknął, nie został tylko zmniejszony.

*Asymetria BBB — MPIW przy x=-2 / x=8 (średnia po seedach), stosunek L/P:*

| zbiór | ReLU (L/P, stosunek) | TanH (L/P, stosunek) |
|---|---|---|
| sin_homo | 4,736 / 0,405 (11,7×) | 0,463 / 0,468 (0,99×) |
| sin_hetero | 5,115 / 0,529 (9,7×) | 0,576 / 0,580 (0,99×) |
| sin_gap | 4,703 / 0,419 (11,2×) | 0,455 / 0,461 (0,99×) |

*Zakresy pasm predykcyjnych vs `Y_RANGE=[-5,5]`:* pod ReLU/`γ=1,0` dwie metody
przebijają zakres (laplace `sin_hetero`: [-6,69; 4,76]; bbb `sin_hetero`: [-5,84; 1,45]).
Pod TanH najgorszy przypadek (laplace, `sin_homo`) to [-4,24; 2,12] — mieści się.

*Dryf średniej poza danymi (nachylenie, x∈[-2;-1,5] i x∈[7;8], średnia po zbiorach):*
TanH redukuje nachylenie o ok. 50% dla map/ensemble/laplace i o ok. 70–80% dla bbb/mcd,
ale **nie eliminuje go całkowicie** — przy `γ=1,0` argumenty tanh na granicach zakresu
(x=±2, x=8) nie są jeszcze w pełni nasycone (`tanh'(2)≈0,07`, nie 0). Hipoteza „średnia
przestanie odjeżdżać liniowo" — potwierdzona częściowo, nie w pełni.

*RMSE/LL/PICP95/MPIW95 — kierunek zmian pod TanH:*
- **laplace, ensemble, map** — poprawa na wszystkich czterech metrykach ekstrapolacji
  (np. laplace `sin_homo` extrap.: LL -0,57→-0,15, PICP95 0,85→0,96).
- **bbb** — pogorszenie in-range (LL spada o ~0,15–0,2 nat na wszystkich trzech
  zbiorach) ORAZ załamanie kalibracji ekstrapolacji: PICP95 spada z 0,64–0,74 (ReLU) do
  0,35–0,49 (TanH).
- **mcd** — załamanie kalibracji ekstrapolacji: PICP95 spada z 0,51–0,83 (ReLU) do
  0,35–0,38 (TanH); RMSE też się pogarsza (np. `sin_homo`: 0,71→0,87).
- **gp** — bez zmian (brak pojęcia aktywacji).

Spadek PICP dla bbb/mcd nie jest odczytywany jako regresja losowa, tylko jako
ujawnienie się znanych własności tych metod z literatury: BBB (mean-field) strukturalnie
niedoszacowuje wariancję posteriora (P2), MCD jest znany z bycia przeuwiarygodnionym w
ekstrapolacji przy standardowym `dropout_p` (P3). Pod ReLU obie metody wyglądały na
lepiej skalibrowane niż w rzeczywistości, bo kawałkami liniowa ekstrapolacja ReLU sama z
siebie poszerzała pasmo średniej, maskując zbyt wąski komponent epistemiczny. TanH usuwa
tę maskującą szerokość i ujawnia rzeczywistą kalibrację tych dwóch metod.

**Decyzja: `activation="tanh"` jako `DEFAULT_ACTIVATION`** (`src/methods/backbone.py`).
E6d (ablacja aktywacji, brief sekcja 9) **odwraca się**: ReLU jest teraz wariantem
ablacyjnym względem TanH jako bazy, nie odwrotnie — do jawnego zapisania w tekście
rozdziału o E6d, żeby nie było niespójności z brzmieniem briefu (który zakładał ReLU
jako punkt wyjścia).

**P13 — świadoma różnica, nie przeoczenie.** Przebieg walidacyjny MC dropout
(odtworzenie protokołu Gala 2016 na 2–3 najmniejszych zbiorach UCI, D12) zostaje na
`activation="relu"`, jawnie przekazywanym w tym jednym wywołaniu — bo P13 sprawdza
zgodność z opublikowanymi liczbami Gala, które są liczone na sieci ReLU. Używanie TanH
tam zepsułoby porównanie z literaturą, nie naprawiło go. Tabela główna (pozostałe
eksperymenty) używa `DEFAULT_ACTIVATION="tanh"`.

**D7c. Mechanizm skoku Laplace'a (D7b) jest opisany w literaturze — nie nasze własne
odkrycie, tylko potwierdzona zgodność z nią.** Foong, Li, Hernandez-Lobato, Turner,
"In-Between Uncertainty in Bayesian Neural Networks", ICML 2019 Workshop on Uncertainty
and Robustness in Deep Learning, arXiv 1906.11537 (`foong2019`, zweryfikowany — O1) —
przypis 3 i Dodatek D: linearised Laplace dla sieci ReLU cierpi z powodu nieciągłego
gradientu `g(x)`, bo ReLU jest niegładkie; ich Rysunek 6 pokazuje to zjawisko na tej
samej konfiguracji co nasza (1 warstwa ukryta, 50 jednostek). Ich rozwiązaniem jest
TanH — dokładnie decyzja D7b, podjęta tu niezależnie, przed odnalezieniem tego źródła.
Mechanizm (nieciągłość jakobianu ReLU w `f_var = J(x)ᵀΣJ(x)`, część E) nie zależy od
geometrii danych — przenosi się bez zastrzeżeń, w odróżnieniu od D14c poniżej. Do
rozdziału 4: cytować jako uzasadnienie literaturowe decyzji D7b, nie prezentować jako
własne odkrycie mechanizmu (empiryczna weryfikacja — zgodność lokalizacji skoku z
załamaniami ReLU co do 6. miejsca po przecinku, część E — pozostaje naszym wkładem;
sam mechanizm jest znany).

**D7d. Rozważone i odrzucone: TanH tylko dla Laplace'a, ReLU dla reszty.** Skoro
mechanizm D7c dotyczy wyłącznie Laplace'a (nieciągłość jakobianu w `f_var=J^TΣJ`),
kusząca alternatywa to zostawić ReLU jako domyślną aktywację wszędzie poza Laplace'em,
zamiast zmieniać ją globalnie (D7b). **Odrzucone.** Aktywacja jest częścią modelu
sieciowego (definiuje, jaką funkcję i jaki posterior nad wagami sieć w ogóle
reprezentuje), nie częścią procedury wnioskowania nad ustalonym modelem — różna
aktywacja per metoda oznacza więc różne posteriory porównywane, ta sama kategoria
błędu co różne `γ` per metoda (D11). `foong2019` sami to rozróżnienie traktują
poważnie: ich Tabela 1 ma osobne wiersze `MFVI-ReLU` i `MFVI-TanH`, właśnie żeby dało
się oddzielić efekt aktywacji od efektu metody aproksymacji — nie miksują ich per
metoda wygodnie. Nasze rozwiązanie: TanH dla wszystkich pięciu metod sieciowych
(`DEFAULT_ACTIVATION`, D7b), ReLU dostępne jako jawna ablacja E6d dla dowolnej
metody, nigdy jako ukryty domyślny wyjątek dla jednej z nich.

**D14c. Płaskie/wąskie pasmo BBB poza danymi — częściowo wyjaśnione przez `foong2019`,
NIE w pełni.** Dodatek B tej pracy dowodzi, że wariancja predykcyjna mean-field VI jest
funkcją wypukłą `x` — zastosowanie: niepewność MIĘDZY dwoma oddzielonymi skupiskami
danych (ich `sin_gap`-owa geometria). Twierdzenie to **przewiduje wzrost** wariancji na
brzegach skupiska, więc dla dwustronnej ekstrapolacji poza jedną wyspą danych `[0,6]`
(nasza geometria `sin_homo`/`sin_hetero`) przewidywałoby to samo — **nie tłumaczy** więc
obserwowanego u nas wąskiego/płaskiego pasma BBB w ekstrapolacji, bo przewiduje coś
przeciwnego do tego, co próbujemy wyjaśnić. Mechanizm niezależny od geometrii, który
faktycznie pasuje: **overpruning** (Trippe & Turner 2017, cytowany w `foong2019`) — MFVI
przycina jednostki ukryte, ustawiając ich wariancję posteriora blisko zera, żeby
zminimalizować koszt KL bez utraty dopasowania danych; przycięta jednostka nie wnosi
wariancji NIGDZIE (ani w luce, ani na brzegach ekstrapolacji), niezależnie od geometrii
danych. Zgodne z wcześniejszym pomiarem sigma posteriora BBB (mediana 13–17× poniżej
priora, część wag blisko zera — patrz D14b/historia sesji, do zweryfikowania ponownie,
O10). **Wniosek: nie zamykać tematu jako "znane ograniczenie mean-field" bez dalszego
sprawdzenia dla naszej geometrii — twierdzenie o wypukłości z `foong2019` przyda się
dopiero przy `sin_gap` i E3, gdzie geometria odpowiada ich configuracji (dwa skupiska,
luka pomiędzy).** Test wariantów setupu — D14d.

**D14d. Test wariantów (A: obecny domyślny; B: pełny setup Foong przy NASZYM reżimie
treningu — 2000 epok/lr=0,01/batch=128, nie ich 20000/0,001/pełny wsad, patrz metodologia
niżej; C: tylko `sigma_o` stałe; D: tylko 32 próbki ELBO), `sin_homo`, `tanh`, seed=0
(faza eksploracji — 1 seed, patrz zasada metodologiczna poniżej).**

| Wariant | overpruning (frac<1% priora) | RMSE (extrap.) | LL | PICP95 | MPIW(x=8)/MPIW(x=3) |
|---|---|---|---|---|---|
| A (bazowy) | 43,0% | 0,400 | -5,42 | 0,265 | 1,028 |
| B (pełny Foong) | 1,3% | 0,993 | -3,43 | 0,283 | 1,111 |
| C (`sigma_o` stałe) | 13,9% | 0,528 | **-1,13** | **0,710** | 1,014 |
| D (32 próbki ELBO) | **0,66%** | 0,766 | -6,36 | 0,193 | 1,208 |
| GP (referencja, kształt "działający") | — | 0,343 | -0,48 | 0,878 | **5,485** |

**Wynik: żaden wariant nie daje rosnącego pasma.** Stosunek MPIW(8)/MPIW(3) mieści się
w 1,01–1,21 we wszystkich czterech konfiguracjach, w tym w wariancie D, gdzie
overpruning jest praktycznie wyzerowany (0,66%) — czyli usunięcie overpruningu NIE
przywraca wzrostu pasma. GP w tej samej siatce/metryce daje 5,49 — dowód, że metryka
działa poprawnie, a płaskość jest własnością BBB w tej geometrii, nie artefaktem
pomiaru. **Do rozdziału 4/5: to jest WŁASNA obserwacja, NIE opierać jej na twierdzeniu
o wypukłości z `foong2019` — to twierdzenie dotyczy geometrii luki między skupiskami
(ich `sin_gap`), nie jednostronnej/dwustronnej ekstrapolacji poza jedną wyspą danych.
Twierdzenie testujemy dopiero na `sin_gap`/E3, gdzie geometria faktycznie odpowiada
`foong2019`.** Nie uruchomiono potwierdzenia na 3 seedach dla wariantów A–D — decyzja
metodologiczna, nie oszczędność: wybór finalnej konfiguracji BBB (patrz D14e, D-sigma-E1)
padł z innych powodów niż ten stosunek MPIW, więc nie ma czego potwierdzać tym testem.

**KOREKTA konwencji metryki (sesja 2026-08-26) — liczby w tabeli wyżej ZOSTAJĄ, ale nie
są porównywalne z nową tabelą D14i.** Stosunek w kolumnie `MPIW(x=8)/MPIW(x=3)` jest
liczony z JEDNEGO punktu odniesienia, `x=3`. Dla samego BBB (i dla wiersza referencyjnego
GP) ta konkretna wartość nie jest skażona — profil BBB jest w tym miejscu gładki, a
zapadnięcie przy `x≈3,2` opisane w D14i dotyczy MCD, nie BBB, więc **wniosek D14d
pozostaje w mocy: żaden z wariantów A–D nie daje rosnącego pasma**. Ale sama konwencja
„jeden punkt w danych" jest krucha (D14i pokazuje, jak dokładnie ugryzła MCD), więc od
tej sesji punktem odniesienia jest **mediana `std_epi` po całym nośniku treningowym**
(`src.metrics.epistemic_growth`). Wartości 1,01–1,21 z tej tabeli i wartości z D14i są
liczone dwiema różnymi konwencjami i **nie wolno ich zestawiać w jednym zdaniu w
rozdziale 4** — przy cytowaniu D14d podać, że to metryka jednopunktowa. Przeliczenie
wariantów A–D na nową konwencję nie zostało uruchomione (wymagałoby ponownego treningu
czterech wariantów BBB); nie jest potrzebne, bo decyzja D14e i tak zapadła z innych
powodów.

**Metodologia sweepu ustalona tą sesją, na stałe:**
1. **Faza eksploracji = jeden seed** (domyślnie `--seeds 0`). Trzy seedy dopiero PO
   wyborze wariantu, żeby potwierdzić, że wybór nie jest przypadkiem — nie w trakcie
   porównywania wariantów. (Wymuszone empirycznie: pierwsza wersja tego sweepu, na 3
   seedach × dosłownym setupie Foong z 20000 epokami, mieliła 90 minut, z czego 85 nie
   wnosiło nic do decyzji.)
2. **Nie kopiować liczby epok z pracy źródłowej — kopiować punkt zbieżności.** Foong
   trenuje 20000 epok przy `lr=0,001`; nasz `lr=0,01` (10×) zbiega ~10× szybciej. Do
   porównania wariantów: 2000 epok / `lr=0,01` / `batch_size=128` (reżim D18) dla
   WSZYSTKICH wariantów, nie dosłowne `20000/0,001/pełny wsad` z pracy. Dopiero jeśli
   wariant wygrywa, sprawdzić osobno, czy dłuższy trening przy ich `lr` coś zmienia —
   nie zakładać tego z góry.
3. **Liczyć tylko metody, których wariant faktycznie dotyczy.** Wariant D (próbki ELBO)
   dotyczy wyłącznie BBB — GP/MAP/MCD/ensemble nie są przeliczane, są brane z wariantu A.
   Wariant C (`sigma_o` stałe) dotyczy wszystkich pięciu metod sieciowych, ale nie GP.

**D14e. `elbo_samples=32` dla BBB — decyzja, ale WYŁĄCZNIE dla E1, nie stała
domyślna klasy.** D14d pokazał, że pojedyncza próbka gradientu (`elbo_samples=1`,
dotychczasowa wartość) daje 43% wag z wariancją posteriora zapadniętą poniżej 1%
priora — to nie kosmetyczna różnica, tylko artefakt estymatora ELBO z jednej próbki,
niereprezentatywny dla mean-field VI jako metody. Przy 32 próbkach: 0,66% (patrz D14d).
**Koszt zmierzony, nie zgadywany:** przy `N=250` (skala E1) koszt jest nieistotny
(~90s/fit zamiast ~17s przy `elbo_samples=1`, ten sam rząd wielkości). Przy `N=8611`
(skala E2, największy zbiór UCI) koszt eksploduje: ~9,5 min/fit (`elbo_samples=1`) →
~96,3 min/fit (`elbo_samples=32`) — dla protokołu 20 podziałów to ~32h na SAM ten
jeden zbiór. Sprawdzone pośrednie wartości: `elbo_samples=8` → 5,96% overpruningu,
~28,5 min/fit (~9,5h/20 podziałów); `elbo_samples=16` → 2,65%, ~46,7 min/fit
(~15,6h/20 podziałów). **Decyzja: `elbo_samples=32` przyjęte dla E1 (poprzez
`experiments/e1_synthetic.py::BBB_ELBO_SAMPLES_E1`, nie jako nowa wartość domyślna
`BBBMethod.__init__`, która zostaje na `1`), wartość dla E2 odłożona do wspólnej
decyzji z liczbą epok tabeli głównej (O4/O8) — nie kopiować `32` do E2 bez ponownego
rozważenia kosztu.**

**D14e-uwaga. Liczba 43% z D14d ma NIEUSTALONĄ KONWENCJĘ — nie zestawiać jej
z nowymi pomiarami bez zastrzeżenia (2026-08-27).** Kod, który policzył wartości
„overpruning (frac<1% priora)" w tabeli D14d, nie istnieje w repozytorium, więc nie da
się odtworzyć, czy ułamek liczony był po samych macierzach wag, czy po wagach i biasach
łącznie. Metryka została odtworzona z opisu jako
`src.methods.bbb.frac_posterior_var_below_prior` i **zwraca obie konwencje naraz**
właśnie po to, żeby przyszła weryfikacja była możliwa; identyfikacja wymagałaby
przeliczenia wariantu A z D14d tą funkcją (`sin_homo`, N=250, 2000 epok,
`fixed_sigma2=0,01`, `elbo_samples=1`) i nie została uruchomiona.

**NIE CYTOWAĆ liczb 43% / 0,66% z D14d w pracy (decyzja autora, 2026-08-28).** Są
nieodtwarzalne z bieżącego kodu i nie wolno na nich opierać uzasadnienia `elbo_samples`
w rozdziale 4. Dwa niezależne powody:

1. **Nieustalona konwencja progu.** Kod, który je policzył, nie istnieje w repozytorium.
   Żadna z dwóch kandydujących konwencji nie odtwarza 0,66% na obecnej konfiguracji E1
   (tabela niżej): `wariancja < 1% priora` daje 100,00%, `sigma < 1% priora` daje 0,00%.
2. **Pomiar sprzed decyzji D-sigma-E1.** Wariant D z D14d miał `sigma_o` UCZONE, podczas
   gdy obecne E1 ma `fixed_sigma2=0,01`, więc porównanie nie jest równoważne nawet przy
   ustalonej konwencji.

Zastępczy pomiar z bieżącego kodu, w konfiguracji D14d (`sin_homo`, `sigma_o` uczone,
2000 epok, `K ∈ {1, 8, 32}`, seedy 0–2, obie konwencje) — patrz tabela w tej sekcji
i `results/bbb_posterior_diagnostic.csv`. **Do rozdziału 4 idą liczby stamtąd, nie
z D14d.**


**ZASTĘPCZY POMIAR Z BIEŻĄCEGO KODU — `elbo_samples` faktycznie steruje tym, jak daleko
posterior odchodzi od inicjalizacji (2026-08-28).** Konfiguracja D14d odtworzona dzisiejszym
kodem: `sin_homo`, `sigma_o` UCZONE, 2000 epok, `K ∈ {1, 8, 32}`, seedy 0–2, średnie po
seedach (`experiments/bbb_posterior_diagnostic.py --e1-sigma learned`,
`results/bbb_posterior_diagnostic.csv`). Wartość początkowa każdej `sigma_post`: 0,0486.

| K | mediana `sigma_post` | max | % wag powyżej startu | `wariancja<1% priora` | `sigma<1% priora` | mediana `std_epi` | KL | suma NLL |
|---|---|---|---|---|---|---|---|---|
| 1 | 0,0091 | 0,0214 | **0%** | 100,00% | **54,7%** | 0,0220 | 16,94 | −399,6 |
| 8 | 0,0273 | 0,1534 | 13% | 98,33% | 7,0% | 0,0602 | 12,10 | −412,7 |
| 32 | 0,0513 | 0,2729 | **55%** | 89,00% | **1,0%** | 0,1482 | 9,20 | −287,0 |

**Wniosek, tym razem z liczb odtwarzalnych:** przy `K=1` ŻADNA waga nie przekracza
wartości początkowej `sigma_post` — cały rozkład leży poniżej punktu startu (mediana
0,0091 wobec 0,0486, czyli 5× niżej), a ponad połowa wag ma `sigma_post` poniżej 1%
odchylenia priora. Przy `K=32` ponad połowa wag jest powyżej punktu startu, a udział wag
zapadniętych spada do 1%. **Mediana `std_epistemic` rośnie 6,7×** (0,022 → 0,148),
a KL spada z 16,9 do 9,2 — posterior odsuwa się od punktu startu i zbliża do priora.
To potwierdza tezę autora, że `elbo_samples` nie jest obojętnym szczegółem estymatora,
tylko parametrem decydującym o tym, czy mean-field VI w ogóle wyucza wariancje.

**Kierunek i rząd wielkości zgadzają się z D14d (43% → 0,66%) w konwencji
`sigma < 1% priora`** (54,7% → 1,0%), co jest najsilniejszą dostępną przesłanką, że taka
konwencja została tam użyta — ale to nadal przesłanka, nie dowód, więc do rozdziału 4
idą liczby z tej tabeli.

**CENA K=32 — zdanie do ROZDZIAŁU 5, nie tylko notatka (autor, 2026-08-28).** Przy `K=32` suma NLL na zbiorze treningowym rośnie z **−399,6 do −287,0**: niezapadnięty posterior kosztuje dopasowanie danych. To nie jest efekt uboczny do wystrojenia, tylko jawny kompromis mean-field VI — więcej próbek ELBO utrzymuje wariancje wariacyjne przy życiu (mediana `std_epi` 6,7× wyżej), a płaci się za to gorszym dopasowaniem średniej. Podać oba końce tego kompromisu razem, nigdy samego zysku.


**Pomiar rozstrzygający część sprawy (2026-08-28, `experiments/bbb_posterior_diagnostic.py`,
`results/bbb_posterior_diagnostic.csv`).** Sprawdzone obie kandydujące konwencje progu na
tej samej sieci, K=32, seed 0:

| konfiguracja | `wariancja < 1% priora` (wagi) | `sigma < 1% priora` (wagi) |
|---|---|---|
| `yacht`, E2, 2000 epok | 86,57% | 2,00% (7/350) |
| `concrete`, E2, 1000 epok | 84,22% | 1,56% (7/450) |
| `sin_homo`, E1, 2000 epok | **100,00%** | **0,00% (0/151)** |

**Żadna z nich nie odtwarza 0,66% z D14d na obecnej konfiguracji E1.** Wartość 0,66%
odpowiada dokładnie 1/151, czyli jednemu parametrowi sieci `1x50` przy `d=1` — co jest
sugestywne, ale nie ustalone. Porównanie nie jest też równoważne: wariant D z D14d był
mierzony PRZED decyzją D-sigma-E1, czyli z `sigma_o` uczonym, a obecne E1 ma
`fixed_sigma2=0,01`. Domknięcie wymagałoby przeliczenia `sin_homo` z uczonym `sigma_o`
(~3 min) — nie uruchomione, bo żadna bieżąca decyzja od tego nie zależy.

**Nazwa metryki celowo NIE zawiera słowa „overpruning".** Przy
`posterior_rho_init = -3,0` i `gamma = 1` nieuczona sieć ma `sigma_post = log1p(exp(-3))
= 0,0486`, czyli wariancję **0,236% priora** — już poniżej progu 1%. Sprawdzone wprost
na dopasowaniu `epochs=0`: metryka zwraca dokładnie `1,0` dla obu konwencji. Metryka nie
odróżnia więc „posterior się zapadł" (mechanizm Trippe & Turnera) od „posterior nie
ruszył z inicjalizacji", a przy krótkich treningach dominuje ten drugi przypadek.
Jednoznaczna jest tylko wartość wyraźnie poniżej 1,0, i tylko różnice przy tej samej
liczbie epok. Zmierzone przy epokach z O4 (`experiments/bbb_elbo_samples_cost.py`,
`results/bbb_elbo_samples_cost.csv`): 87–100% na `yacht` i 96–100% na `power_plant`,
czyli poziom, przy którym ta metryka nic nie rozstrzyga.

**D-sigma-E1. `sigma_o` (szum aleatoryczny) ustalone na prawdziwą wartość — WYŁĄCZNIE
w E1, wyłącznie tam gdzie istnieje pojedyncza prawdziwa wartość.** `sin_homo` i
`sin_gap` mają stały `sigma=0,1` z konstrukcji (`src/data.py::_sigma_homo`) —
`fixed_sigma2=0,01` (WARIANCJA, `sigma**2`, nie `sigma` — patrz błąd niżej) dla
wszystkich pięciu metod sieciowych na tych dwóch zbiorach
(`experiments/e1_synthetic.py::KNOWN_HOMOSCEDASTIC_SIGMA`).

**Błąd wykryty i naprawiony (2026-08-26): pierwsze przeliczenie E1 pod tą decyzją
miało `KNOWN_HOMOSCEDASTIC_SIGMA = {"sin_homo": 0,1, "sin_gap": 0,1}` — podstawiono
`sigma` tam, gdzie `fixed_sigma2` (log-WARIANCJA w `HomoscedasticMLP`) potrzebowała
`sigma**2`.** Skutek: `mean_var_aleatoric=0,10000` zamiast `0,01000` (10× za dużo) dla
wszystkich pięciu metod sieciowych na `sin_homo`/`sin_gap` — pasmo szerokości samego
szumu, `PICP95≈1,000` bo pasmo było ~3× za szerokie. Przeszedł wszystkie 60 ówczesnych
testów, bo żaden nie sprawdzał konkretnej liczby względem znanej prawdy, tylko
kształty/znaki/determinizm. Naprawione, dodane dwa testy regresyjne
(`tests/test_methods.py::test_fixed_sigma2_is_variance_not_std`,
`::test_e1_known_homoscedastic_sigma_is_variance_not_std`) — jeden na poziomie
mechanizmu (`HomoscedasticMLP`), jeden na poziomie konkretnej wartości w
`e1_synthetic.py`, żeby złapać obie warstwy tego samego błędu. Sprawdzone przy okazji:
`sin_hetero` (gdzie `sigma_o` jest uczone, nie ustalone) nie ma analogicznej pomyłki —
`map`'s `mean_var_aleatoric=0,01693` zgadza się z `mean(sigma_hetero(x)**2)=0,01752`
policzonym niezależnie z `src/data.py`, w granicach szumu zbieżności.

**`sin_hetero` NIE jest
objęty tą decyzją** — jego prawdziwe `sigma(x)=0,05+0,15x/6` zmienia się z `x`, a nasz
backbone jest z definicji homoskedastyczny (jeden globalny `log_sigma2`), więc nie ma
pojedynczej "prawdziwej wartości", do której dałoby się go ustalić bez arbitralnego
wyboru (np. średniej) — `sigma_o` zostaje tam uczone, tak jak w E2 i tak jak dotychczas
(zgodne z D1: `sin_hetero` i tak służy wyłącznie do pokazania, czego model
homoskedastyczny nie potrafi, nie do porównania metod na równych prawach). Uzasadnienie
z pomiaru (D14d, wariant C, BBB): PICP95 0,265→0,710, LL -5,42→-1,13 — rząd wielkości
różnicy w jakości niepewności, nie subtelny efekt. **Na E2 (dane rzeczywiste UCI,
prawdziwy szum nieznany) `sigma_o` zostaje uczone dla wszystkich metod bez wyjątku —
świadoma różnica protokołu E1 vs E2, nie przeoczenie.**

**D14f. Rysunki E1, wersja wyłącznie epistemiczna (`img3_{method}_epistemic.png`,
`src/plotting.py::make_single_method_figure_epistemic`).** Diagnostyka estymatora
(D14d/E dalej) potwierdziła: `Var[mu_t]` liczone ręcznie z `samples` zgadza się z
`Prediction.var_epistemic` co do 6. miejsca po przecinku dla bbb/mcd/ensemble — nie
ma błędu w estymatorze. Ale surowy człon epistemiczny ROŚNIE u wszystkich pięciu metod —
**liczby poniżej poprawione 2026-08-26, patrz D14i; poprzednia wersja tego akapitu
podawała `mcd ~0,040→0,156 (~3,9×)`, gdzie `0,040` to wartość odstająca z lokalnego
zapadnięcia profilu MCD przy `x≈3,2`**. Wersja aktualna, mediana po nośniku treningowym →
`x=8` (`sin_homo`, seed=0): gp 0,0136→0,525 (38,5×), laplace 0,0169→0,883 (52,4×),
ensemble 0,0204→0,0545 (2,67×), mcd 0,120→0,156 (**1,31×**), bbb 0,0877→0,130 (1,48×) —
tonie to jednak na standardowym rysunku `±2σ_total`, bo `sigma_o` jest w E1 ustalone
identycznie (`0,1`) dla wszystkich metod (D-sigma-E1): przy epistemicznym starcie
rzędu `0,02-0,09` (bbb/mcd/ensemble), człon aleatoryczny (`0,1`) dominuje wizualnie
pasmo całkowite i maskuje wzrost. Skoro `sigma_o` jest tą samą stałą u wszystkich
metod, różnica między metodami na rysunku `±2σ_total` jest już WYŁĄCZNIE członem
epistemicznym — pokazanie go osobno (te same osie `X_RANGE`/`Y_RANGE`, ta sama skala,
brief section 10) nic nie ukrywa, tylko usuwa wspólną stałą. **Obie wersje idą do
pracy — pokazują różne rzeczy: `±2σ_total` pokazuje, co obserwator faktycznie widzi
jako pasmo predykcyjne; `±2σ_epistemic` pokazuje, czym metody realnie się różnią.**

**D14g. Wniosek do rozdziału 5: metody próbkujące (bbb/mcd/ensemble) mają
niepewność epistemiczną w obszarze danych tego samego rzędu co szum, metody
analityczne (gp/laplace) o rząd wielkości mniejszą.**

**LICZBY POPRAWIONE 2026-08-26 (poprzednio brane z `x=3`, patrz D14i).** Wartość
odniesienia to teraz mediana `std_epi` po nośniku treningowym, `sin_homo`, seed=0:
bbb `0,0877`, mcd `0,120`, ensemble `0,0204`; gp `0,0136`, laplace `0,0169`.
Poprzednia wersja podawała dla MCD `≈0,040` — wartość z lokalnego zapadnięcia profilu,
3× za niską.

**Teza D14g wychodzi z korekty MOCNIEJSZA, nie słabsza, i to w obu połowach:**
bbb (`0,0877`) i mcd (`0,120`) leżą przy `sigma_o=0,1`, a mcd wręcz JE PRZEKRACZA —
czyli w gęstych danych sam człon epistemiczny MCD jest większy niż cały szum
obserwacyjny, którego model ma się nauczyć. gp i laplace pozostają o rząd wielkości
niżej. Ensemble (`0,0204`) należy tu do grupy „analitycznej", nie do „próbkującej" —
oryginalne sformułowanie „metody próbkujące" jako blok trzech metod jest w tej połowie
tezy **nieścisłe i trzeba je w rozdziale 5 rozdzielić**: to bbb i mcd mają nadmiarową
niepewność w danych, ensemble nie.

**Druga połowa tezy (za pewne poza danymi) po korekcie jest OSTRZEJSZA dla MCD i
wymaga wyodrębnienia ensemble.** Stosunki wzrostu (mediana → `x=8`): gp 38,5×,
laplace 52,4×, ensemble 2,67×, bbb 1,48×, **mcd 1,31× (nie 3,9×, jak podawała
poprzednia wersja)**. Po stronie lewej (`x=-2`) bbb daje **0,99×**, czyli nie rośnie
w ogóle — jego niepewność przy dwóch jednostkach poza danymi jest równa medianie
w danych. Ensemble po lewej: 4,85×.

**Interpretacja (utrzymana, z poprawionym podziałem metod): bbb i mcd są jednocześnie
ZA MAŁO pewne w obszarze danych i ZA BARDZO pewne poza nimi** — zgodne z P2
(mean-field VI niedoszacowuje wariancję posteriora w sensie strukturalnym, nie tylko na
brzegach) i P3 (MCD bywa przeuwiarygodniony). Nie tylko „pasmo płaskie poza danymi"
(D14d), ale „pasmo niepotrzebnie szerokie w danych, za wąskie poza nimi" jednocześnie —
dwustronny, nie jednostronny, problem kalibracji kształtu. **Ensemble jest osobnym
przypadkiem: kształt ma poprawny (2,7–4,9×, minimum w danych, wzrost na brzegach),
tylko amplitudę ~5× poniżej `sigma_o`** — czyli jego problemem jest to, że pasmo jest
za ciche, żeby je było widać, a nie to, że ma zły kształt. To rozróżnienie jest tezą
do rozdziału 5, nie szczegółem technicznym.

**D14h. Ablacja `M` dla deep ensembles — E6b, ZAMKNIĘTA tym pomiarem, nie trzeba
powtarzać w Etapie 5. `M` NIE zmienione na stałe — `DeepEnsembleMethod`'s default
zostaje `M=5`.** `sin_homo`, konfiguracja E1 (`fixed_sigma2=0,01`), `M ∈ {5, 10, 20}`,
3 seedy (0-2).

**Wniosek wiodący: `M` steruje POWTARZALNOŚCIĄ oszacowania niepewności, nie jego
wartością ani jakością predykcji.** Stosunek std_epi(x=8)/std_epi(x=3) (miara wzrostu
pasma epistemicznego w ekstrapolacji, D14d/D14g) NIE zależy istotnie od `M` w zakresie
5-20 co do ŚREDNIEJ (2,65 / 3,09 / 3,62 dla `M`=5/10/20 — różnice mniejsze niż rozrzut
międzyseedowy), ale **odchylenie standardowe między seedami spada wyraźnie z rosnącym
`M`: 1,035 (`M`=5) → 1,035 (`M`=10) → 0,454 (`M`=20)** — ta sama średnia, o połowę
mniejszy rozrzut przy `M=20`. `M` nie zmienia WARTOŚCI, którą estymator mierzy, tylko
PRECYZJĘ, z jaką ją mierzy — to jest własność estymatora (uśrednianie po większej
liczbie niezależnie wytrenowanych członków redukuje wariancję samej estymaty
wariancji), a nie przypadek jednego pomiaru. Konsekwencja praktyczna: podwojenie `M`
kosztuje dwukrotnie więcej czasu treningu (mierzone: `17,6s→35,4s→71,5s`, liniowo w
`M`) za dokładniejszy, nie inny, wynik — decyzja o `M` w innym eksperymencie
(np. E2, gdzie koszt liczy się per zbiór × podział) powinna ważyć koszt względem
POTRZEBNEJ PRECYZJI oszacowania niepewności, nie względem jego wartości średniej.

*Dane źródłowe, seed=0 (krok 1 tego pomiaru):*

| M | std_epi(x=-2) | std_epi(x=3) | std_epi(x=8) | stosunek(8/3) | czas treningu |
|---|---|---|---|---|---|
| 5 | 0,0991 | 0,0204 | 0,0545 | 2,675 | 17,6s |
| 10 | 0,0899 | 0,0188 | 0,0720 | 3,831 | 35,4s |
| 20 | 0,1010 | 0,0237 | 0,0848 | 3,584 | 71,5s |

*Potwierdzenie na seedach 0-2 (kryterium ustalone z góry: `M=10` wyraźnie powyżej
`M=5` na WSZYSTKICH trzech seedach → przyjąć `M=10`; przedziały nakładają się → zostaje
`M=5`):*

| seed | M=5 stosunek(8/3) | M=10 stosunek(8/3) | M=20 stosunek(8/3) |
|---|---|---|---|
| 0 | 2,675 | 3,831 | 3,584 |
| 1 | 3,669 | 3,534 | 4,083 |
| 2 | 1,599 | 1,908 | 3,178 |
| **średnia ± std** | **2,648 ± 1,035** | **3,091 ± 1,035** | **3,615 ± 0,454** |

**Kryterium zastosowane, NIE spełnione: na seedzie 1, `M=10` (3,534) wypada NIŻEJ niż
`M=5` (3,669)** — nie jest to "wyraźnie powyżej na wszystkich trzech". Różnica średnich
`M=5→M=10` (~0,44) jest mniejsza niż rozrzut międzyseedowy (~1,04) — dokładnie sytuacja
przewidziana w kryterium jako "przedziały się nakładają". **Decyzja: `M=5` zostaje —
wartość z lakshminarayanan2017, potwierdzona empirycznie jako wystarczająca** (nie
"odziedziczona bez sprawdzenia", tak jak `posterior_rho_init=-3,0` w D14b: sprawdzona i
utrzymana, nie zmieniona domyślnie). RMSE/LL/PICP95 nie zależą od `M` w ogóle (patrz
tabela seed=0 — te trzy metryki są wręcz niewrażliwe na `M`, w odróżnieniu od
precyzji stosunku(8/3)). Czas treningu skaluje się liniowo z `M` — koszt `M=10`/`M=20`
nie jest uzasadniony tym pomiarem, ALE gdyby inny eksperyment potrzebował precyzyjniej
zmierzonego stosunku wzrostu pasma (np. rysunek porównawczy w rozdziale 5, gdzie rozrzut
międzyseedowy miałby zaciemniać przekaz), `M=20` jest tańszym sposobem na to niż więcej
seedów przy `M=5`.

**E6b zrealizowana tym pomiarem (3 seedy, kryterium ustalone z góry, wynik
jednoznaczny) — nie odtwarzać jako osobny eksperyment w Etapie 5, chyba że pojawi się
nowy powód wracać do `M`.**

**D14i. Metryka wzrostu pasma epistemicznego: MEDIANA po nośniku treningowym, nie
wartość w `x=3`. Poprzednia konwencja dała jedną liczbę, która była błędna, i jeden
wniosek, który się na niej opierał.**

Do tej sesji stosunek wzrostu liczyliśmy jako `std_epi(x=8) / std_epi(x=3)`, biorąc
`x=3` jako „środek danych". **Dla MCD `x=3` trafia w wąskie lokalne zapadnięcie
profilu.** Profil `std_epi(x)` MC dropoutu na `sin_homo` (seed=0) jest praktycznie
płaski, 0,13–0,15 na całym `[-2, 8]`, ale ma jeden dołek do `0,040` przy `x≈3,2` — i
punkt odniesienia w nim wylądował. Skutek: raportowane `3,9×` zamiast rzeczywistego
`1,31×`, i zbudowany na tym odczyt „niepewność MCD rośnie w ekstrapolacji prawie
czterokrotnie" (D14g, poprzednia wersja), którego profil nie potwierdza.

Dołek jest realną własnością wytrenowanej sieci, nie szumem estymatora — zostaje
widoczny na `figures/rodzial3_rys/img3_epistemic_profile_sin_homo.png` i nie jest
niczym „wygładzany". Chodzi wyłącznie o to, żeby **nie kotwiczyć na nim porównania**.
Nowa wartość odniesienia to mediana `std_epi` po całym nośniku treningowym (~600
punktów siatki dla `sin_homo`), której jedna wąska cecha profilu nie ruszy.
Implementacja: `src.metrics.epistemic_growth`, przeliczenie:
`python scripts/epistemic_growth.py --csv` → `results/epistemic_growth.csv`.
Dla `sin_gap` mediana liczona jest po `[0,2] ∪ [4,6]`, z wyłączeniem luki — definicja
masek jest jedna, w `src.data.range_masks`, wspólna z `results/e1_synthetic.csv`.

*Przeliczone, seed=0, wszystkie zapisane wyniki (mediana `std_epi` w danych → punkty
sondujące). `gap_ratio` = mediana w luce `(2,4)` / mediana w danych.*

| zbiór | metoda | mediana w danych | `x=-2` | `x=8` | stosunek(-2) | stosunek(8) | `gap_ratio` |
|---|---|---|---|---|---|---|---|
| sin_homo | gp | 0,0136 | 0,5249 | 0,5249 | **38,51** | **38,51** | — |
| sin_homo | laplace | 0,0169 | 0,8943 | 0,8833 | **53,05** | **52,40** | — |
| sin_homo | ensemble | 0,0204 | 0,0991 | 0,0545 | 4,85 | 2,67 | — |
| sin_homo | bbb | 0,0877 | 0,0865 | 0,1297 | **0,99** | 1,48 | — |
| sin_homo | mcd | 0,1196 | 0,1525 | 0,1562 | 1,27 | **1,31** | — |
| sin_gap | gp | 0,0136 | 0,5384 | 0,5384 | 39,57 | 39,57 | **3,27** |
| sin_gap | laplace | 0,0164 | 0,9053 | 1,0018 | 55,05 | 60,92 | **7,32** |
| sin_gap | ensemble | 0,0196 | 0,1256 | 0,1187 | 6,42 | 6,06 | 1,41 |
| sin_gap | bbb | 0,0805 | 0,0699 | 0,1195 | 0,87 | 1,48 | **1,06** |
| sin_gap | mcd | 0,1254 | 0,1344 | 0,1569 | 1,07 | 1,25 | **0,68** |
| sin_hetero | gp | 0,0178 | 0,5606 | 0,5606 | 31,56 | 31,56 | — |
| sin_hetero | laplace | 0,0214 | 1,0333 | 0,5286 | 48,19 | 24,65 | — |
| sin_hetero | ensemble | 0,0362 | 0,1029 | 0,1004 | 2,84 | 2,78 | — |
| sin_hetero | bbb | 0,1164 | 0,1115 | 0,1789 | 0,96 | 1,54 | — |
| sin_hetero | mcd | 0,1438 | 0,1760 | 0,1788 | 1,22 | 1,24 | — |

**Trzy rzeczy z tej tabeli, których stara metryka nie pokazywała:**
1. **BBB nie rośnie w lewo w ogóle** — stosunek `0,87–0,99` na wszystkich trzech
   zbiorach, czyli przy dwóch jednostkach poza danymi ma niepewność RÓWNĄ tej w
   danych albo mniejszą. To nie jest „słaby wzrost", to brak reakcji. Profil BBB
   jest w istocie rampą rosnącą w `x` (`corr(std_epi, x) = 0,87` na `sin_homo`), nie
   krzywą reagującą na gęstość danych — kierunek wzrostu jest przypadkiem geometrii,
   nie wnioskowaniem.
2. **MCD w luce `sin_gap` ma `gap_ratio = 0,68`, czyli niepewność NIŻSZĄ niż w
   danych.** To nie jest za mały wzrost — to odwrócony znak. Najmocniejszy pojedynczy
   wynik do rozdziału 5, i argument za tym, żeby rysunki rozdziału 3 robić na
   `sin_gap`, nie na `sin_homo` (na `sin_homo` różnica jest tylko ilościowa).
3. **Ensemble nie należy do tej samej kategorii co bbb/mcd** — `gap_ratio = 1,41`,
   stosunki 2,7–6,4×, minimum w danych. Kształt poprawny, amplituda ~5× poniżej
   `sigma_o=0,1`. Patrz D14g (poprawione) i O8.

**D14j. `foong2020` — twierdzenie o nieekspresywności jednowarstwowego MFVI i MC
dropoutu. Klucz do ZWERYFIKOWANIA przed `.bib` (nie dopisywać samodzielnie, zasada
twarda).**

Foong, Burt, Li, Turner, *On the Expressiveness of Approximate Inference in Bayesian
Neural Networks*, NeurIPS 2020, arXiv **1909.00719**. To PEŁNA wersja pracy, której
warsztatowy skrót (`foong2019`, arXiv 1906.11537) cytujemy w D7c/D14c — nie duplikat,
inne twierdzenia. Zweryfikowana przez autora 2026-08-26: istnieje, mówi to, co poniżej.

Dwa wyniki:
- **Dla sieci z JEDNĄ warstwą ukrytą (ReLU): mean-field Gaussian ORAZ Monte Carlo
  dropout nie mogą mieć istotnie podwyższonej niepewności pomiędzy dobrze
  rozdzielonymi obszarami niskiej niepewności.** Ograniczenie w przestrzeni funkcji —
  własność rodziny aproksymacji, nie kwestia zbieżności, hiperparametrów ani
  overpruningu.
- **Dla sieci głębszych (co najmniej dwie warstwy ukryte): wynik o uniwersalności —
  ISTNIEJĄ w tych samych rodzinach aproksymacje dające dowolnie elastyczne
  oszacowania niepewności.**

**Dlaczego to jest dla nas istotne:** nasz backbone to dokładnie `1 × 50`
(`src/methods/backbone.py`), czyli architektura leżąca w klasie objętej twierdzeniem
o niemożliwości — i to dla obu metod naraz, bbb i mcd. Wyjaśnia jednym mechanizmem to,
czego D14c/D14d nie potrafiły domknąć: dlaczego wariant D w D14d wyzerował overpruning
do 0,66%, a pasmo pozostało płaskie (usuwaliśmy nie tę przyczynę). Pasuje też do
`gap_ratio` z D14i: bbb `1,06`, mcd `0,68` — dwie metody objęte twierdzeniem nie
pokazują wzrostu w luce, trzy nieobjęte (gp, laplace, ensemble) pokazują.

**DWA ZASTRZEŻENIA, bez których nie wolno tego cytować:**
1. **Wynik o głębokich sieciach to dowód ISTNIENIA, nie gwarancja.** Mówi, że w tych
   rodzinach istnieje aproksymacja o dobrym kształcie niepewności — NIE, że VI ją
   znajdzie. Autorzy sami stwierdzają empirycznie, że patologie podobnej postaci mogą
   utrzymywać się przy VI w sieciach głębszych. **Druga warstwa ukryta nie jest
   zagwarantowaną naprawą i nie wolno jej tak przedstawiać** — ani w rozdziale 4, ani
   jako uzasadnienie zmiany backbone'u.
2. **Twierdzenie jest dla ReLU; my mamy TanH** (D7b). Aktywacja jest częścią modelu
   (D7d), więc przeniesienie twierdzenia na naszą konfigurację jest przesłanką, nie
   dowodem. Dla `sin_homo` dochodzi drugie rozminięcie: twierdzenie dotyczy niepewności
   MIĘDZY skupiskami, a `sin_homo` to ekstrapolacja poza jedną wyspę (to samo
   zastrzeżenie co w D14c). **Dla `sin_gap` geometria się zgadza** — tam twierdzenie
   stosuje się wprost, z zastrzeżeniem o aktywacji.

**Status decyzji o drugiej warstwie ukrytej: NIE, na razie** (autor, 2026-08-26).
Przy `2 × 50` sieć ma ~2700 parametrów wobec `N=250`, czyli `N/p ≈ 0,09` — daleko poza
wszystkim, co testowaliśmy, i wraca pytanie o uwarunkowanie GGN Laplace'a, które
wymusiło `N=250` i `float64` (D25, D26). Architektura jest wspólna dla wszystkich pięciu
metod sieciowych (sekcja 4), więc zmiana nie dotyczy samego BBB. Do decyzji po
domknięciu O8 (D14k) i sweepu `dropout_p` (D14l).

**D14k. O8 zamknięte pomiarowo: liczba epok RZECZYWIŚCIE steruje rozrzutem ensemble i
kierunek jest taki, jak przewidywał autor — ale nie da się tego wykorzystać. `epochs =
2000` zostaje.** `sin_homo` i `sin_gap`, `epochs ∈ {300, 600, 1000, 2000}`, seedy 0–2,
tylko ensemble, protokół E1. `experiments/ensemble_epochs_sweep.py` →
`results/ensemble_epochs_sweep.csv`.

*`sin_homo`, średnia ± std po seedach 0–2:*

| epoki | mediana `std_epi` | `std_epi(x=-2)` | stosunek(-2) | stosunek(8) | RMSE w danych | LL w danych | PICP w danych | RMSE ekstrap. | LL ekstrap. | PICP ekstrap. |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 0,0170 ± 0,0033 | **0,2011 ± 0,0613** | **11,82 ± 2,39** | **9,77 ± 1,74** | 0,1244 ± 0,0047 | 0,636 ± 0,043 | 0,896 ± 0,011 | **0,6403 ± 0,0276** | **−7,783 ± 1,121** | 0,347 ± 0,202 |
| 600 | 0,0217 ± 0,0094 | 0,1679 ± 0,0129 | 9,00 ± 4,50 | 4,85 ± 2,75 | 0,1066 ± 0,0015 | 0,817 ± 0,017 | 0,937 ± 0,012 | 0,3366 ± 0,0245 | −3,003 ± 0,668 | 0,465 ± 0,070 |
| 1000 | 0,0199 ± 0,0096 | 0,1430 ± 0,0177 | 8,78 ± 5,43 | 4,53 ± 1,84 | 0,1043 ± 0,0005 | 0,840 ± 0,005 | 0,942 ± 0,008 | 0,2458 ± 0,0300 | −1,010 ± 0,379 | 0,572 ± 0,047 |
| 2000 (domyślne) | 0,0231 ± 0,0055 | 0,1005 ± 0,0085 | 4,44 ± 0,62 | 2,64 ± 1,03 | **0,1017 ± 0,0022** | **0,866 ± 0,023** | **0,947 ± 0,010** | **0,2289 ± 0,0479** | **−0,755 ± 0,575** | **0,624 ± 0,066** |

*`sin_gap`, ten sam kierunek na wszystkich kolumnach:* stosunek(-2) 11,65 → 5,74;
`gap`-owa geometria niczego nie odwraca. RMSE ekstrap. 0,5488 → 0,2456, LL ekstrap.
−6,986 → −1,048, PICP ekstrap. 0,364 → 0,620 przy przejściu 300 → 2000 epok.

**Hipoteza O8 potwierdzona mechanicznie, odrzucona praktycznie.** Przy 300 epokach
pasmo epistemiczne na brzegu jest **dwukrotnie szersze** w wartościach bezwzględnych
(`std_epi(x=-2)` 0,2011 wobec 0,1005) i rośnie **11,8×** wobec 4,4× — czyli tak, przy
2000 epokach członkowie faktycznie zbiegają bliżej siebie i to zawęża pasmo, dokładnie
jak autor zgłaszał przy D7b. **Ale szersze pasmo nie jest lepszym pasmem:** przy 300
epokach PICP@95 w ekstrapolacji wynosi **0,347 wobec 0,624** przy 2000. Pasmo jest
dwa razy szersze i trafia dwa razy rzadziej — bo średnia jest niedotrenowana i ucieka
od prawdy szybciej (RMSE ekstrap. 0,64 wobec 0,23, LL −7,8 wobec −0,76), niż pasmo
nadąża się poszerzać. **Rozrzut między członkami przy 300 epokach nie jest rozrzutem
posteriora, tylko szumem niedokończonej optymalizacji** — te same 300 epok psują też
dopasowanie w danych (RMSE 0,124 wobec 0,102, PICP 0,896 wobec 0,947, czyli poniżej
nominału) i podnoszą rozrzut międzyseedowy prawie wszędzie (`±0,202` na PICP ekstrap.
wobec `±0,066`).

**Wniosek do rozdziału 5, i to jest wniosek negatywny wart zapisania:** amplitudy
pasma epistemicznego deep ensembles NIE da się podnieść liczbą epok, tak jak nie da się
jej podnieść liczbą członków (`M`, D14h). D14h pokazała, że `M` steruje PRECYZJĄ
estymaty, nie jej wartością; D14k pokazuje, że liczba epok steruje wartością, ale
wyłącznie kosztem jakości predykcji, w bilansie ujemnym. **Ensemble ma poprawny kształt
niepewności i za małą amplitudę (D14g, D14i), i żaden z dwóch parametrów samej metody
tego nie zmienia** — jeśli amplituda ma wzrosnąć bez utraty dopasowania, musi to przyjść
spoza zestawu pokręteł `lakshminarayanan2017` (randomised prior functions —
`osband2018`; NTKGP — `he2020`; oba NIEZWERYFIKOWANE i oba byłyby zmianą metody, nie
wariantem, patrz sekcja o zakresie pracy).

**Zastrzeżenie do `train_time_s` w `results/ensemble_epochs_sweep.csv`: NIE CYTOWAĆ.**
Ten przebieg dzielił CPU z równolegle liczonym sweepem `dropout_p` (D14l), więc czasy
są zawyżone o nieznaną i niejednorodną wartość. Kolumna zostaje w pliku, bo skrypt ją
zapisuje, ale nie jest pomiarem. Jeśli czas treningu w funkcji epok będzie potrzebny,
przeliczyć osobnym, wyłącznym przebiegiem (D23).

**D14l. Sweep `dropout_p` dla MC dropoutu — WYNIK NEGATYWNY, celowo. `dropout_p` steruje
WYŁĄCZNIE amplitudą niepewności MCD, nie jej kształtem, a podnoszenie go pogarsza
wszystko, co miało poprawić.** `sin_homo`, seedy 0–2, protokół E1 (`fixed_sigma2=0,01`,
tanh, 2000 epok). `experiments/mcd_dropout_sweep.py` →
`results/mcd_dropout_sweep.csv`, `results/mcd_dropout_profiles.csv`.

| `p` | mediana `std_epi` (POZIOM) | stosunek(-2) (KSZTAŁT) | stosunek(8) (KSZTAŁT) | PICP w danych | MPIW w danych | PICP ekstrap. | MPIW ekstrap. | LL w danych | RMSE w danych |
|---|---|---|---|---|---|---|---|---|---|
| 0,05 | 0,0983 ± 0,0036 | 1,39 ± 0,03 | 1,34 ± 0,20 | 0,942 ± 0,027 | 0,536 ± 0,010 | **0,305 ± 0,195** | 0,636 ± 0,005 | **0,575 ± 0,133** | **0,1373 ± 0,0168** |
| 0,10 (domyślne) | 0,1144 ± 0,0052 | 1,25 ± 0,08 | 1,23 ± 0,15 | 0,947 ± 0,022 | 0,574 ± 0,012 | 0,179 ± 0,110 | 0,678 ± 0,004 | 0,541 ± 0,065 | 0,1446 ± 0,0096 |
| 0,20 | 0,1331 ± 0,0040 | 1,15 ± 0,03 | 1,19 ± 0,04 | 0,964 ± 0,002 | 0,630 ± 0,007 | 0,090 ± 0,029 | 0,727 ± 0,005 | 0,485 ± 0,013 | 0,1554 ± 0,0025 |
| 0,30 | **0,1487 ± 0,0047** | **1,17 ± 0,04** | **1,18 ± 0,06** | 0,952 ± 0,016 | **0,677 ± 0,012** | **0,066 ± 0,029** | 0,772 ± 0,015 | 0,357 ± 0,042 | 0,1773 ± 0,0071 |

**Trzy odczyty, wszystkie negatywne:**

1. **Poziom rośnie, kształt nie.** Mediana `std_epi` rośnie monotonicznie 0,098 → 0,149
   (1,51×) przy `p` 0,05 → 0,30, ale stosunki wzrostu **maleją**: 1,39 → 1,17 przy
   `x=-2`, 1,34 → 1,18 przy `x=8`. Przy rozrzucie międzyseedowym 0,03–0,04 to spadek
   realny, nie szum. **Więcej dropoutu daje profil PŁASKIEJSZY, nie bardziej rosnący** —
   dokładnie odwrotnie, niż potrzeba. Potwierdzone niezależnie testem kształtu:
   korelacja Pearsona między profilami `log std_epi(x)` przy różnych `p` wynosi
   0,89–0,96 (logi, bo teza brzmi „ta sama krzywa przeskalowana", co na logach jest
   przesunięciem o stałą). To jedna krzywa podniesiona do góry, nie inna krzywa.

2. **Poprawa pokrycia w ekstrapolacji nie następuje — pokrycie SPADA.** PICP@95 poza
   danymi: 0,305 → 0,179 → 0,090 → **0,066**. Pasmo jest szersze (MPIW 0,636 → 0,772),
   a mimo to trafia rzadziej, bo silniejszy dropout mocniej reguluje także ŚREDNIĄ i ta
   ucieka od prawdy w ekstrapolacji szybciej, niż pasmo się poszerza. Nie da się kupić
   pokrycia poza danymi przez `p`.

3. **Koszt w danych jest realny i płacony od razu.** MPIW w danych rośnie o 26%
   (0,536 → 0,677) przy PICP już siedzącym na nominale (0,94–0,97 wszędzie), czyli
   pasmo poszerza się tam, gdzie było poprawne. RMSE w danych pogarsza się 0,137 →
   0,177 (+29%), LL 0,575 → 0,357.

**Do rozdziału 5:** MCD ma jedno pokrętło i to pokrętło jest od amplitudy. Nie ma
parametru, który zmieniałby kształt jego niepewności, bo w posteriorze dropoutowym nie
ma niczego, co byłoby funkcją położenia danych — maska Bernoulliego nad jednostkami
ukrytymi nie widzi `x` treningowego. Zgodne z `verdoja2020` (arXiv 2008.02627: estymata
epistemiczna MCD nie zależy od ilości danych treningowych ani od ich wariancji, a jest
ustawiana przez `dropout_p`) — **klucz do zweryfikowania przed `.bib`.** Zgodne też z
`foong2020` (D14j), które obejmuje MC dropout tym samym twierdzeniem co MFVI.
**`dropout_p = 0,1` zostaje** (D18, wartość z `gal2016`): sweep nie daje powodu jej
zmieniać — żadna inna wartość nie jest lepsza, tylko inaczej zła.

**D8. Model szumu homoskedastyczny** — patrz D1 poniżej, kluczowa decyzja.

**D9. Prior jako jawna kara w stracie, `weight_decay = 0`.**
Kara `||θ||² / (2γ²N)` dodawana do straty. Uzasadnienie: `weight_decay` w PyTorch
dodaje `wd·θ` do gradientu, a w `Adam` wchodzi przed skalowaniem adaptacyjnym, więc
efektywna siła kary zależy od historii gradientu per parametr — prior przestaje być
jednorodny. Jawna kara usuwa optymalizator z równania. Jest to też bliższe implementacji
referencyjnej Gala, używającej regularyzatora Keras.

*Odnotować:* wzór `weight_decay = 1/(γ²N)`, nie `1/(2γ²N)` — zweryfikowane empirycznie.

**D10. `log σ²` nie podlega karze priora.** Status identyczny z `noise_level`
w `WhiteKernel` GP: stała estymowana z danych, nieregularyzowana. Objęcie jej karą
ciągnęłoby `σ²` w stronę 1 i tworzyło asymetrię wobec GP.

**D11. Ta sama `γ` we wszystkich metodach.** Cztery różne parametryzacje jednego priora
(`bayesian-torch`, `laplace-torch`, jawna kara, `numpyro`) — jedna funkcja przeliczająca
i dwa testy zgodności. Bez tego porównywane byłyby różne modele bayesowskie,
a nie różne przybliżenia jednego posteriora.

**D11b. Sprawdzono `γ = 0,3` (`prior_precision ≈ 11,1`) jako alternatywę dla `γ = 1,0`
— odrzucono, `γ = 1,0` zostaje.** Wartość `1,0` weszła do briefu bez uzasadnienia poza
„ta sama we wszystkich metodach" — poprawne co do zasady D11, ale sama liczba nigdy nie
była sprawdzona. Zrewidowana (tymczasowo) po zdiagnozowaniu
skoków `var_epistemic` Laplace'a na granicach aktywacji ReLU (D-item o igłach, część E):
jednostka ukryta, której załamanie ma słabo ograniczony przez dane kierunek posteriora,
generuje realny, nieciągły skok wariancji linearyzowanej przy przekroczeniu progu —
mocniejszy prior powinien taki kierunek ścisnąć. Sprawdzone: `γ ∈ {1,0; 0,5; 0,3; 0,1}`
× 6 metod × 3 seedy, trzy kryteria naraz — czy skok wyraźnie maleje, czy RMSE/LL na
`in_range` się nie psuje, czy PICP@95 na `in_range` zostaje ≥0,90.

| `γ` | skok Laplace'a (max Δvar) | RMSE `in_range` (map/laplace/ensemble) | uwaga |
|---|---|---|---|
| 1,0 | 1,28 | ~0,10 | punkt startowy |
| 0,5 | 0,98 | ~0,10 | za mała redukcja (~24%) |
| **0,3** | **0,27** | **~0,10** | **spełnia wszystkie trzy kryteria** |
| 0,1 | 0,0025 | ~0,38 | dopasowanie rozwalone dla 4 z 5 metod sieciowych |

Przy `γ=0,1` kara `1/(2γ²N)` jest 100× silniejsza niż przy `γ=1,0` i dominuje stratę —
RMSE na `in_range` skacze z ~0,10 do ~0,38 dla map/mcd/ensemble/laplace (LL z ~+0,85 do
~-0,46). **BBB jedyny przechodzi przez `γ=0,1` względnie bez szkody** (RMSE=0,12,
LL=0,69) — jego prior wchodzi przez człon KL w ELBO, nie przez jawną karę skalowaną
`1/γ²`, więc ta sama `γ` degraduje różne metody w różnym tempie. Do odnotowania wprost
w rozdziale 4: „ta sama `γ`" (D11) nie znaczy „ta sama wrażliwość na `γ`" — struktura
priora różni się między metodami nawet przy identycznej wartości hiperparametru.

**Decyzja odwrócona, `γ = 1,0` (2026-08-25).** `γ = 0,3` rzeczywiście tłumi skok
Laplace'a (tabela wyżej) i nie psuje `in_range` dla żadnej metody — ale poprawia
wygląd jednego wykresu kosztem zasady D11 samej w sobie: `γ` jest częścią definicji
posteriora, a praca twierdzi, że sześć metod przybliża *ten sam* posterior. Różne `γ`
dobrane pod kątem tego, jak wypada dana metoda, oznacza już porównanie różnych modeli
bayesowskich, nie różnych przybliżeń jednego modelu. Dodatkowo efekt netto nie jest
jednoznacznie pozytywny: `γ = 0,3` pomaga ensemble (ciaśniejszy prior, subiektywnie
gładszy wykres), ale pogarsza dopasowanie MCD i Laplace'a względem `γ = 1,0` — nie ma
tu jednej wartości korzystnej dla wszystkich metod naraz, co samo w sobie jest argumentem
przeciw wybieraniu `γ` pod konkretną metodę. Wraca się więc do `γ = 1,0` — standardowego
priora jednostkowego, wybranego niezależnie od wyglądu wyników — jako **stałej
domyślnej** (`src/methods/backbone.py::DEFAULT_GAMMA`). Sam pomiar sweepu (tabela wyżej)
zostaje jako udokumentowany fakt empiryczny: skok Laplace'a *jest* wrażliwy na `γ`, tylko
że to nie jest wystarczający powód, by go zmieniać. Po powrocie do `γ = 1,0` skok
Laplace'a wraca do rzędu wielkości sprzed sweepu (`sin_homo`: max Δvar ≈ 0,11 na
gęstej siatce — patrz część E, wpis o mechanizmie ReLU-kink) i — nowość, niezauważona
przy `γ = 0,3` bo mniejsza — na `sin_hetero` osiąga pojedynczą igłę Δvar ≈ 4,78 przy
x≈4,39, co wypycha pasmo Laplace'a poza `Y_RANGE = [-5, 5]` (dolna granica pasma
≈ -5,36). Wymaga rewizji `Y_RANGE` albo osobnego komentarza w tekście o tej metodzie/
zbiorze — patrz część D (pytania otwarte).

**D29. `LOGVAR_CLAMP` poszerzone z `[-6, 6]` do `[-12, 6]` (2026-08-27) — klamra była
WIĄŻĄCA na 2 z 6 zbiorów UCI i ustawiała tam wspólną podłogę członu aleatorycznego.**
Pomiar: `experiments/logvar_clamp_diagnostic.py`, `map`, split 0, seedy 0–2, zapis co
epokę przez 2000 epok, dwie klamry.

| zbiór | dotyka −6 | epoka kontaktu | `sigma^2` wymuszone | wariancja resztowa @2000 | `log_sigma2` surowy @2000 |
|---|---|---|---|---|---|
| `yacht` | **3/3 seedy** | **342** | 0,002479 | **0,000692** | −6,052 |
| `energy` | **3/3 seedy** | **295** | 0,002479 | **0,001354** | −6,032 |
| `concrete` | nie | — | — | 0,020715 | −3,821 |
| `wine_quality_red` | nie | — | — | 0,252963 | −1,365 |
| `kin8nm` | nie | — | — | 0,062190 | −2,741 |
| `power_plant` | nie | — | — | 0,056105 | −2,878 |

Dla homoskedastycznego gaussowskiego NLL przy ustalonej średniej optimum `sigma^2` to
dokładnie wariancja resztowa, więc na `yacht` klamra trzymała `sigma^2` **3,6× za
wysoko** (0,00248 wobec 0,00069), a parametr surowy przeszedł poniżej granicy (−6,052) —
gradient stale go tam spychał. To nie jest optimum przy podłodze, tylko model dociskany
do podłogi.

**Dlaczego to ta sama wada strukturalna co D18 (`batch_size`) i O4 (liczba epok):** jedna
zadeklarowana stała, która na czterech zbiorach nie znaczy nic, a na dwóch ustawia człon
aleatoryczny wszystkim metodom naraz. Brief wprowadza ją jako zabezpieczenie numeryczne;
tam, gdzie jest wiążąca, przestaje nim być i staje się hiperparametrem modelu szumu.

**Klamra fabrykowała też plateau zbieżności.** Krzywa NLL walidacyjnego `yacht` jest
płaska od 500 epok wzwyż WYŁĄCZNIE dzięki klamrze; przy `[-12, 6]` ma minimum przy ~350
epokach i dalej się psuje (−1,00 przy 500 → −0,39 przy 1000 → −0,08 przy 2000).
Wyznaczona pierwotnie liczba 1000 epok dla `yacht` była więc wytworem klamry. Na
`energy` klamra też jest wiążąca, ale różnica NLL mieści się w rozrzucie międzyseedowym
(|Δ| ≤ 0,06). Na pozostałych czterech zbiorach różnica wynosi **dokładnie 0,0000 w
każdej epoce**.

**Koszt, nazwany wprost:** najlepszy NLL walidacyjny `yacht` pogarsza się z −1,41 do
−1,16, czyli o ~0,25 nata. Ta część wyniku pochodziła z podłogi, nie z metody.

**Asymetria między metodami — argument rozstrzygający.** Kontrola krzyżowa
(`--contact-check`, split 0, seed 0, 2000 epok, `results/logvar_clamp_contact_check.csv`):
stary próg byłby wiążący dla `map` (−6,82 / −6,40) i `bbb` (−6,35 / −6,01) na
`yacht`/`energy`, ale **nie** dla `mcd` (−3,36 / −3,65 — dropout utrzymuje `sigma^2`
wyżej). Podłoga nie działałaby więc jednakowo na wszystkie metody: przypinałaby `map`
i `bbb` do tej samej sztucznej wartości członu aleatorycznego, zostawiając `mcd` jego
własną. To dodaje oś różnicy między metodami, która nie jest różnicą w przybliżeniu
posteriora — dokładnie to, czego teza pracy zabrania.

**E1 nie wymaga przeliczenia.** Sprawdzone bezpośrednim porównaniem, nie założone: na
`sin_homo`/`sin_gap` `log_sigma2` jest ustalone (`fixed_sigma2=0,01` → −4,605, wewnątrz
obu klamr), a na `sin_hetero`, gdzie jest uczone, wychodzi −4,079 — też wewnątrz obu.
Predykcje pod `[-6,6]` i `[-12,6]` są bit-identyczne dla wszystkich trzech wariantów.

`exp(-12) = 6,1·10⁻⁶` leży poniżej najmniejszej zmierzonej wartości `sigma^2` na
wszystkich sześciu zbiorach (0,00053 na `yacht`), więc nowa granica jest nieaktywna
wszędzie tam, gdzie mierzono.

**D12. Dwa protokoły liczby epok.**
- Tabela główna: jednolity, tańszy protokół, liczba epok wyznaczona empirycznie
- Osobny przebieg walidacyjny (P13): pełne odtworzenie protokołu referencyjnego —
  4000 epok (40 × mnożnik 100 z nazw plików `results/`) i grid search per fold —
  wyłącznie dla MC dropout na 2–3 najmniejszych zbiorach

Opisać jako **dwa eksperymenty o różnych celach**, nie jako niespójność.

**D30. Liczba epok wyznaczana PER ZBIÓR, jako maksimum po metodach (O4/D12 ZAMKNIĘTE,
2026-08-28).** `experiments/uci_epochs_sweep.py` → `results/uci_epochs_sweep_final{,_chosen,_combined}.csv`,
`figures/uci_epochs_sweep_final.png`.

**Problem.** Przy wspólnym `batch_size=128` (D18) zadeklarowane „2000 epok" znaczyło
6 000 kroków optymalizatora na `yacht` i 136 000 na `power_plant` — 22,7×. Ta sama wada
strukturalna co przy `batch_size` przed D18 i przy `LOGVAR_CLAMP` przed D29: jedna stała
znacząca co innego na różnych zbiorach.

**Jednostką zostają EPOKI, nie kroki gradientu.** Protokół Hernándeza-Lobato i Gala
deklaruje epoki, a P13 porównuje nas z liczbami z tego protokołu; stały budżet kroków
dałby `power_plant` 89 przejść przez dane zamiast 2000 i skasował jedyny zewnętrzny test
poprawności implementacji.

**Protokół pomiaru.** Jeden split (index 0), 20% foldu treningowego wydzielone na
walidację **wyłącznie do tego pomiaru** (do E2 nie wchodzi; skaler dopasowany na
wewnętrznych 80%, bo `y` wchodzi wprost do NLL walidacyjnego). Siatka
`{5, 10, 20, 50, 100, 200, 500, 1000, 2000}`. Kryterium: najmniejsza wartość z siatki,
przy której średni NLL walidacyjny mieści się w **bezwzględnych 0,02 nata** od minimum
tej metody. Próg bezwzględny, nie „1% minimum": przy NLL ≈ −1,3 reguła procentowa daje
tolerancję 0,013 nata, poniżej rozrzutu międzyseedowego, więc wybierałaby 2000 prawie
zawsze. `map` i `mcd` na seedach 0–2, `bbb` na seedzie 0 przy `elbo_samples=8`.

**Jeden przebieg trajektoryjny zamiast dziewięciu dopasowań.** Trajektoria treningu jest
deterministyczna przy danym seedzie, więc model po epoce `e` przebiegu 2000-epokowego
jest tym samym modelem, który dałby osobny przebieg `epochs=e`. Zweryfikowane dwukrotnie:
wartości z trajektorii są identyczne z wartościami z osobnych dopasowań co do ostatniej
drukowanej cyfry, a predykcje pośrednie nie zaburzają trajektorii, bo
`uci_epochs_sweep._predict_isolated` zapisuje i przywraca stan wszystkich trzech
generatorów (`MCDropoutMethod.predict` woła `set_seed`, a maski dropoutu w treningu idą
z tego samego strumienia).

**Reguła wyboru: MAKSIMUM po metodach.** Brief (sekcja 4) wymaga jednej liczby epok
wspólnej dla wszystkich metod na danym zbiorze, więc czyjeś optimum musi stać się
budżetem wszystkich. „Nikt nie jest niedotrenowany" to jedyna reguła, której nie trzeba
bronić przez wskazanie, dlaczego akurat ta metoda ustala budżet. Pierwsze podejście
mierzyło sam `map` jako najtańszy proxy — zawiodło dokładnie tam, gdzie pytanie miało
znaczenie (`concrete`: `map` 100, `mcd` 1000, rozbieżność 10×, bo `map` się tam
przeucza, a dropout i człon KL trzymają pozostałe dwie).

**TABELA DO ROZDZIAŁU 4:**

| zbiór | n_train | kroków/epokę | epoki | kroków łącznie | wyznaczone przez |
|---|---|---|---|---|---|
| `yacht` | 277 | 3 | **2000** ᶜ | 6 000 | `mcd` |
| `energy` | 691 | 6 | **1000** | 6 000 | `map` |
| `concrete` | 927 | 8 | **1000** | 8 000 | `mcd` |
| `wine_quality_red` | 1439 | 12 | **20** | 240 | `bbb` |
| `kin8nm` | 7373 | 58 | **1000** | 58 000 | `bbb` |
| `power_plant` | 8611 | 68 | **50** | 3 400 | `map` |

ᶜ = wartość ograniczona sufitem, patrz niżej.

**TABELA ROZBIEŻNOŚCI MIĘDZY METODAMI — zmierzony fakt do rozdziału 4, nie założenie:**

| zbiór | `map` | `mcd` | `bbb` | rozpiętość |
|---|---|---|---|---|
| `yacht` | 500 | 2000 | 2000 | 4,0× |
| `energy` | 1000 | 1000 | 1000 | 1,0× |
| `concrete` | 100 | 1000 | 200 | **10,0×** |
| `wine_quality_red` | 5 | 10 | 20 | 4,0× |
| `kin8nm` | 200 | 500 | 1000 | 5,0× |
| `power_plant` | 50 | 10 | 50 | 5,0× |

**Pokrycie wszystkich sześciu metod trzema krzywymi.** `ensemble` i `laplace` przechodzą
przez ten sam `train_homoscedastic_mlp` z `dropout_p=0.0` co `map` (`ensemble.py` to `M`
niezależnych przebiegów MAP, `laplace.py` to post-hoc Hessian wokół sieci MAP), więc
krzywa `map` jest ich krzywą. `gp` nie ma pętli gradientowej. Zmierzenie `map`, `mcd`
i `bbb` pokrywa komplet — to nie jest ekstrapolacja, tylko tożsamość ścieżki treningu.

**SUFIT DLA `yacht` = 2000, i dlaczego jest potrzebny.** Na `yacht` metody idą
w przeciwne strony: `map` ma optimum przy 500 i potem się rozpada (−1,00 przy 500;
−0,08 przy 2000; **+1,35 ± 1,27 przy 5000**), a `mcd` poprawiał się na końcu każdej
próbowanej siatki (−0,306 przy 2000; −0,364 przy 5000). Przy takim układzie reguła
maksimum **nie ma punktu stałego** — każde wydłużenie siatki podnosi wspólny budżet
i kosztuje `map` coraz więcej (kara względem własnego optimum: 0,92 nata przy 2000,
2,35 nata przy 5000), czyli o odpowiedzi decydowałaby siatka, nie dane. 5000 zostało
**zmierzone i nieprzyjęte**; `yacht` jest ograniczony do 2000, co jest zarazem prawdziwym
wewnętrznym optimum `bbb` na tym zbiorze. **Optimum `mcd` na `yacht` pozostaje
niezmierzone — to jest jawne ograniczenie protokołu, nie liczba udająca pomiar.**
W kodzie jako `EPOCH_CEILING`, z uzasadnieniem.

**Poprawka reguły wczesnego przerwania (i co złapała).** Pierwsza wersja przerywała
przebieg, gdy WAHANIA NLL w oknie 300–500 nie przekraczały 0,02 nata. Przepuściła krzywą,
która wciąż opadała: `bbb` na `kin8nm` szedł 0,2693 → 0,2475 między epoką 200 a 500,
monotonicznie, a każda różnica wewnątrz okna mieściła się pod progiem — „płaskość" była
spełniona przez krzywą bez płaskiego odcinka. Reguła sprawdza teraz **TREND** na drugiej
połowie okna (400 → 500). Przeliczenie `kin8nm`/`bbb` bez przerwania dało **1000 zamiast
500** (krzywa opada do 0,2113 przy 2000), więc błąd był realny, nie hipotetyczny.
Przerwanie na `power_plant` okazało się uzasadnione (−0,0056 przy 200 wobec −0,0055
przy 500).

**Rozpiętość kroków WZROSŁA: 240–58 000, czyli 242×** (wobec 22,7× przy jednolitych 2000
epokach). To nie jest regres — **wyrównanie liczby kroków nigdy nie było celem; celem
było doprowadzenie każdego zbioru do jego własnego punktu zbieżności.** Zbiory różnią się
tym, ile treningu potrzebują, i ta liczba jest miarą tej różnicy, nie usterką protokołu.
Zapisane tutaj, bo padnie na obronie.

**`wine_quality_red` = 20 epok (240 kroków) przyjęte świadomie.** Przy `n_train=1439` sieć
widzi dane 20 razy; to mało, ale nie absurdalnie mało, a wszystkie trzy metody mają tam
optimum poniżej 100 epok. Rozważone i **odrzucone**: obniżenie progu tolerancji tylko dla
zbiorów o płaskiej krzywej — byłoby dobieraniem kryterium pod wynik (zasada z sekcji 13
briefu).

**Obserwacja poboczna, odnotowana bez badania:** `mcd` ma wyraźnie gorszy NLL walidacyjny
w wartościach bezwzględnych na małych zbiorach — `yacht` −0,306 wobec −1,311 dla `bbb`,
`energy` −0,522 wobec −1,612 dla `map`. Zmierzone, niezbadane; jeśli to samo pojawi się
w E2, będzie wymagało komentarza w rozdziale 5.

**D31. OSTRZEŻENIE DO P1 — wynik `map` na `yacht` i `concrete` jest współkształtowany
przez budżet, nie tylko przez metodę.** Reguła maksimum z D30 daje `map` budżet
wyznaczony przez metody regularyzowane. Strata każdej metody przy wspólnej liczbie epok
względem jej własnego optimum (nats, z `results/uci_epochs_sweep_final.csv`):

| zbiór | wspólne epoki | `map` | `mcd` | `bbb` |
|---|---|---|---|---|
| `yacht` | 2000 | **+0,922** | 0 | 0 |
| `energy` | 1000 | 0 | 0 | 0 |
| `concrete` | 1000 | **+0,604** | 0 | +0,157 |
| `wine_quality_red` | 20 | 0 | 0 | 0 |
| `kin8nm` | 1000 | +0,018 | 0 | 0 |
| `power_plant` | 50 | 0 | 0 | 0 |

Reguła jest praktycznie darmowa wszędzie poza dwoma zbiorami, na których `map` się
przeucza — i tam kosztuje **wyłącznie** `map`, 0,60 i 0,92 nata.

**Konsekwencja dla P1 (teza o overconfidence baseline'u, rozdz. 2.1.2).** Bez opisania
tego P1 zostałoby potwierdzone z niewłaściwego powodu: nie dlatego, że MAP jest
przeuwiarygodniony z natury, tylko dlatego, że na tych dwóch zbiorach dostał budżet
przekraczający własne optimum dziesięciokrotnie (`concrete`: 1000 wobec 100). **W
rozdziale 5, przy omawianiu P1, podać wynik `map` ORAZ jego stratę względem własnego
optimum z tabeli wyżej.** Na `energy`, `wine_quality_red` i `power_plant` strata wynosi
zero, więc tam P1 jest testowane czysto — i to są wiersze, na których należy oprzeć
wniosek.

## 4.4 Method configurations

**D13. Biblioteki zamiast implementacji własnych.** Zasada: minimalizować liczbę decyzji
bez źródła w literaturze, nie ilość kodu.

| Metoda | Źródło | Uzasadnienie |
|---|---|---|
| GP | `scikit-learn` | wnioskowanie dokładne, brak decyzji projektowych |
| BBB | `bayesian-torch` | `LinearReparameterization` = Blundell i in. 2015 |
| MC dropout | `torch.nn.Dropout` | metoda *jest* zwykłym dropoutem |
| Laplace | `laplace-torch >= 0.2.3` | biblioteka autorów pracy cytowanej w 3.4 |
| Deep ensembles | pętla po `M` treningach | metoda *jest* pętlą |
| Baseline MAP | `torch.nn` | sieć bazowa |

**Wymienić wersje bibliotek i komplet użytych argumentów, łącznie z domyślnymi.**
Domyślny parametr biblioteki jest nadal decyzją, tylko niejawną.

**D14. BBB bez MOPED i bez Flipout.** Obie opcje zmieniałyby metodę względem opisu
w rozdziale 3.2.

**D15. MC dropout: dropout TYLKO przed warstwą wyjściową (hidden→mean), NIE przed
wejściową.** Poprawka względem pierwszej wersji (i względem literalnego gal2016: „dropout
przed każdą warstwą z wagami") — odkryta empirycznie w Etapie 4 (E1), nie zaplanowana.
Własny moduł `AlwaysOnDropout`, żeby próbkowanie nie zależało od `model.train()`.

*Uzasadnienie.* Przy `d=1` (dane syntetyczne) dropout na wejściu zeruje **jedyną cechę**
w `dropout_p` przebiegów — sieć widzi `x=0` zamiast `x`, przy treningu i predykcji
jednocześnie (`AlwaysOnDropout` próbkuje niezależnie od trybu). W treningu psuje to
część batchy do stałej wartości; sieć kompensuje, zawyżając `log_sigma2`, żeby pokryć
skoki straty — nie dlatego, że dane są tak zaszumione. Zmierzone na `sin_homo`
(prawdziwe `sigma²=0,01`): `mean_var_aleatoric=0,122` (13× za dużo) przed poprawką,
`0,038` po (wciąż ~3,8× za dużo — reszta to spodziewany, udokumentowany w literaturze
efekt uboczny stochastycznego treningu MCD, nie patologia `d=1`).

**Przeliczone 2026-08-31** (`tanh`, 2000 epok, seed 0, jeden wątek, flaga
`input_dropout` z E.1b): **0,106 z dropoutem wejściowym wobec 0,038 bez**. Wartość
po poprawce potwierdza się co do drugiego miejsca; wartość przed poprawką jest dziś
niższa (0,106 wobec 0,122), bo pierwotny pomiar powstał przed przejściem na `tanh`
(D7b). **Wartość `0,0096`, która krążyła w F.1 poz. 6 i w docstringu `MLP`
w `src/methods/backbone.py`, jest błędna** — poprawiona w obu miejscach. RMSE w `[0,6]`:
`0,233→0,083`. MPIW@95 w `[0,6]`: `1,644→0,921`. Usunięte dla wszystkich `d`, nie
tylko `d=1` — mechanizm (porzucone cechy wchłaniane przez estymatę szumu) jest ten sam
przy dowolnym wymiarze, tylko proporcjonalnie słabszy, gdy więcej niż jedna cecha
przeżywa maskę.

**D16. Deep ensembles bez adversarial training.** Autorzy raportują to jako opcjonalne
wzmocnienie; włączenie dodałoby metodzie składnik nieobecny w pozostałych czterech.

**D17. Laplace: `subset_of_weights="all"`, predykcja linearised, `prior_precision`
STAŁE = `1/gamma²` (nie strojone przez marglik).** Tryb subnetwork odrzucony, bo
`KronSubnetLaplace` nie istnieje, co czyniłoby Rysunek 3.9 niewykonalnym. Marglik-strojenie
było pierwotnym podejściem, odrzucone po E1: zbiega do wartości odpowiadającej innej
`gamma` niż reszta metod, łamiąc D11. `"marglik"` i `"unregularised"` zostają jako
warianty tylko do ablacji E6c.

**D18. `T = 100`, `batch_size = 128`, `p = 0.1`, `M = 5`, `posterior_rho_init = -3,0`**
— wartości domyślne. `T` uzasadniane przez E6a, `M` przez E6b, `posterior_rho_init`
sprawdzone empirycznie (D14b). **`batch_size = 128` wspólne dla WSZYSTKICH metod
sieciowych** (map, mcd, ensemble, bbb, laplace — GP nie dotyczy, brak pętli gradientowej)
— nie jest to szczegół kosztowy, tylko czynnik kontrolowany z tabeli w 4.1: rozmiar
batcha ustala liczbę kroków optymalizatora na epokę (`ceil(N/batch_size)`), a różny
`batch_size` per metoda dodałby drugą oś różnicy obok przybliżenia posteriora — przy
`N=8611` (największy zbiór E2) różnica `batch_size=32` vs pełny wsad to 270× więcej
kroków, inny reżim optymalizacji, nie tylko inny koszt. Nie pełny wsad (`batch_size=N`)
dla wszystkich: deep ensembles (lakshminarayanan2017) czerpie część różnorodności
z kolejności batchy, nie tylko z inicjalizacji — pełny wsad byłby odstępstwem od
opisanej metody, nie tylko wolniejszą wersją tej samej.

## 4.5 Evaluation metrics

**D19. Podział kolumn.** Porównywalne z literaturą: **RMSE i LL**. Tylko te dwie występują
w opublikowanych tabelach. PICP, MPIW, interval score, CRPS, ECE — wkład własny bez wierszy
referencyjnych. Nazwać ten podział jawnie.

**D20. LL w konwencji literatury** (wyżej = lepiej, wartości ujemne), nie NLL.

**D21. Metryki po odwróceniu standaryzacji**, w jednostkach oryginalnych.
Odwrócenie dotyczy całego rozkładu predykcyjnego: `σ_original = σ_standardised · scaler.scale_`.

**D22. Dekompozycja raportowana wybiórczo.** Liczona zawsze, do tabeli głównej nie trafia —
na UCI nie ma ground truth dla żadnego z członów, więc kolumna `mean_var_epistemic` jest
liczbą, o której nie da się powiedzieć, czy niska wartość oznacza „lepiej", czy „metoda
zapada się do punktu". Raportowana w E1 (gdzie `σ(x)` znane), E3 (gap split)
i E4 (odniesienie do HMC).

**D23. Koszt jako metryka.** Czas treningu, czas inferencji, liczba parametrów.
Konieczne ze względu na D-B (budżet ensemble'u).

**D23a. Która kolumna kosztu jest pomiarem, a której nie porównywać między zbiorami.**
Dopisane 2026-08-30 po przebiegu `e2_uci.py --timing-pass`.

*Czas treningu* do rozdziału 5 bierzemy z `results/e2_cost.csv` (split 0, sekwencyjnie,
`torch_threads=1`, nic innego nie działa). Kolumna `train_time_s` w `results/e2_uci.csv`
jest produktem ubocznym przebiegu na 8 procesach i zawyża czas o **medianę 1,60×**
(zakres 1,03–2,89× na 34 komórkach). GP jest tam praktycznie nieskażony (1,03–1,22×),
bo nigdy nie szedł równolegle — to potwierdza D14k liczbą, nie przypuszczeniem.

*`predict_time_ms_per_1k` nie jest porównywalny między zbiorami.* To czas jednego
wywołania `predict` na całym zbiorze testowym, przeskalowany do 1000 punktów, więc przy
małym zbiorze testowym dominuje w nim stały narzut wywołania. Laplace: **1398 ms/1k na
yacht (31 punktów testowych) wobec 222 ms/1k na kin8nm (819) i power_plant (957)** — to
nie jest predykcja sześć razy wolniejsza, tylko ten sam narzut podzielony przez 32×
mniejszy mianownik. Porównania **w obrębie jednego zbioru** są ważne (na yacht `map`
6,2 wobec `laplace` 1398 ms/1k to prawdziwe ~200×), **między zbiorami nie**.
W rozdziale 5 albo zestawiać metody wewnątrz zbioru, albo raportować tę kolumnę tylko
dla dwóch największych zbiorów, gdzie narzut jest amortyzowany.

**D31b. Parametry estymatora trzymane stałe między eksperymentami mierzącymi tę samą
wielkość — na przykładzie odrzuconego cięcia w E3.** Zapisane 2026-08-31.

Przy planowaniu E3 rozważano obniżenie `elbo_samples` dla BBB z 32 do 8, bo BBB odpowiada
za **70,2% kosztu** tego eksperymentu (`e2_cost.csv`: 152,3 s na `concrete` i 113,2 s na
`energy` na jedno dopasowanie, przy 378,2 s dla wszystkich sześciu metod razem). Cięcie
**odrzucone**, i powód jest ogólniejszy niż ta jedna decyzja.

`K` nie jest parametrem kosztu, tylko parametrem **mierzonej wielkości**. Nasz własny
pomiar (D14e, pozycja 3 w liście F.1) pokazuje, że stosunek MPIW(8)/MPIW(3) — czyli
metryka kształtu pasma, rodzeństwo `epi_gap_ratio` z E3 — rośnie z **1,03 przy `K=1`
do 1,21 przy `K=32`**. Mniejsze `K` spłaszcza pasmo, które eksperyment ma zmierzyć.

**Decydujący jest kierunek tego błędu.** P10 przewiduje, że metody mean-field nie
podnoszą niepewności w luce, czyli `epi_gap_ratio[bbb] ≈ 1`. Zaniżone `K` popchnęłoby
wynik dokładnie **w stronę zgodności z przewidywaniem z literatury** — a to jest kierunek,
w którym czytający (i recenzent, i my sami) się nie zatrzymuje. Potwierdzenie
przewidywania wyprodukowane przez artefakt estymatora jest znacznie trudniejsze do
wychwycenia niż jego obalenie, bo nie wywołuje pytania „skąd ta rozbieżność".

**Do rozdziału 4, jako zasada, nie jako anegdota:** parametry estymatora (`elbo_samples`
dla BBB, `T` dla BBB i MC dropoutu, `M` dla ensemble'u) są takie same we wszystkich
eksperymentach, które raportują tę samą wielkość — inaczej różnica między E2 a E3 byłaby
różnicą estymatorów, a nie różnicą protokołu badania. Tam, gdzie koszt wymagał cięcia,
cięto **liczbę powtórzeń** (kontrola losowa w E3 na 5 podziałach zamiast 20), nigdy
dokładność estymatora.

## 4.6 Reproducibility

Do opisania: przypięty commit danych źródłowych, sumy kontrolne, seedy, wersje bibliotek,
`torch.use_deterministic_algorithms`, jedna komenda odtwarzająca każdy eksperyment.

**D23b. Zdanie o odtwarzalności — dokładne brzmienie i zmierzony zakres.** Dopisane
2026-08-31.

Do rozdziału 4 wchodzi: **„każdy wynik jest odtwarzalny z podanego seeda przy
`torch.set_num_threads(1)`"** — nie samo „z podanego seeda", bo to jest nieprawda.
Liczba wątków zmienia kolejność redukcji w float64 i przy metodach iteracyjnych daje
z tego samego seeda inną sieć (część F, z tabelą pomiarów).

Zmierzony zakres rozjazdów: **do 6,75·10⁻⁴ w RMSE i 3,10·10⁻³ w LL**. Odniesienie, żeby
czytelnik wiedział, co to znaczy: rozrzut międzypodziałowy w E2 dla tych samych komórek
to SE średniej **0,139 (RMSE) i 0,046 (LL) na `concrete`** oraz **4,2·10⁻⁴ i 4,7·10⁻³
na `kin8nm`**. Najgorszy przypadek (kin8nm/ensemble, LL) to ~66% SE **pojedynczego
podziału**, czyli ~15% SE średniej po dwudziestu; różnice między metodami na tym zbiorze
wynoszą 0,1–0,4 nata, czyli 30–130× więcej. **Żaden wniosek z tabeli głównej się na tym
nie ruszał.**

**Wynik przeliczenia (2026-08-31).** 600 wierszy sieciowych policzonych na jednym wątku,
81 wierszy GP przeniesionych bez zmian (`torch_threads = "not_applicable"`). Porównanie
z wersją mieszaną:

- **461 z 600 wierszy identycznych** co do 10⁻¹²; różnice dotyczą wyłącznie `map`,
  `laplace` i `mcd` — czyli dokładnie tych metod, które szły ścieżką sekwencyjną.
  `bbb` i `ensemble` są identyczne co do bitu wszędzie, bo już wcześniej liczyły się
  w workerach na jednym wątku.
- Największe różnice pojedynczych podziałów: **2,50·10⁻² RMSE i 7,85·10⁻² LL**
  (`energy`/`map`), 1,30·10⁻² RMSE (`concrete`), 3,57·10⁻² LL (`kin8nm`/`map`).
- Wpływ na **średnie trafiające do tabeli głównej**: maks. **7,47·10⁻⁴ (RMSE)**
  i **2,25·10⁻³ (LL)**, czyli ~6–7% błędu standardowego tych średnich.
- **Żadne uporządkowanie metod się nie zmieniło.** Jedyny sygnał „CHANGED" dotyczy
  `kin8nm`/RMSE i jest artefaktem sortowania wartości **dokładnie równych**: `map`
  i `laplace` mają tam teraz identyczne RMSE (0,073544) co do ostatniego bitu.
- **P8 stało się dokładne — i to jest uzasadnienie decyzji o przeliczeniu, nie skutek
  uboczny.** `max |rmse[laplace] − rmse[map]|` wynosi **0,0** na wszystkich 120 parach
  podziałów, wobec 2,36·10⁻³ w tabeli mieszanej. Przewidywanie z `daxberger2021` mówiące,
  że Laplace zachowuje dokładność sieci MAP „co do 10⁻¹⁰", jest teraz spełnione
  **z zerową różnicą**.

  Przed przeliczeniem rozdział 5 musiałby zawierać akapit w rodzaju: „P8 zachodzi na
  pięciu zbiorach dokładnie, a na `kin8nm` z różnicą 2,4·10⁻³, która nie wynika z metody,
  tylko z kolejności redukcji w float64 przy różnej liczbie wątków". Taki akapit jest
  prawdziwy i jednocześnie **osłabia twierdzenie, którego broni** — czytelnik zapamiętuje
  wyjątek, nie wyjaśnienie. Po przeliczeniu zdanie brzmi: „Laplace zwraca średnią
  predykcyjną sieci MAP co do bitu, na wszystkich 120 parach". Półtorej godziny obliczeń
  kupiło usunięcie wyjątku, którego nie dałoby się wytłumaczyć krócej niż w akapicie.

Mimo to konfigurację ujednolicono i tabelę przeliczono w całości (jeden wątek na
wszystkich ścieżkach, kolumna `torch_threads` w schemacie z sekcji 8, poprzednia wersja
zachowana jako `results/e2_uci_mixed_threads.csv`). Powód do zapisania wprost: przypis
„część wierszy powstała przy innej liczbie wątków, różnice zmierzono poniżej 7·10⁻⁴"
jest gorszy niż półtorej godziny obliczeń — odtwarzalność jest własnością zero-jedynkową
i albo komenda z pracy odtwarza liczbę z tabeli, albo nie.

**D24. `model_factory` wywoływane po `set_seed()`.** Konstruowanie modelu przed ustawieniem
seeda uzależnia wagi od globalnego stanu RNG — błąd wykryty w Etapie 2, wart jednego zdania
jako element procedury, nie anegdota.

**D25. `float64` we wszystkich metodach.** Macierz precyzji posteriora Laplace'a ma
współczynnik uwarunkowania rzędu `10^6` niezależnie od precyzji — to własność problemu
(`N` porównywalne z liczbą parametrów), nie artefakt. Przy `float32` zostaje ~1 cyfra
znacząca i wartości wariancji nie są wiarygodne. Koszt przy 151–550 parametrach pomijalny.

*Szczegół implementacyjny do rozdziału 4:* `bayesian-torch` nie przyjmuje `dtype` —
`LinearReparameterization` tworzy tensory według globalnego domyślnego dtype, a `dnn_to_bnn`
go nie przekazuje. Kontrola wyłącznie przez `torch.set_default_dtype` przed konstrukcją
modelu. **Nie przez `.double()` po fakcie** — to pobiera inny ciąg z generatora, więc
trening z tym samym seedem zbiega do innej sieci.

**D26. `N = 250` w eksperymentach syntetycznych, nie 50.** Uzasadnienie w części C, D1b.

**D27. `curvlinops-for-pytorch == 2.0.1`, `laplace-torch` z commita GitHuba.**
`laplace-torch 0.2.3` nie istnieje na PyPI. `laplace-torch` woła prywatną metodę
`_compute_kfac()`, przemianowaną w `curvlinops >= 3.0` — bez pinu `kron` w ogóle nie działa.
Do sekcji o reprodukowalności razem z obejściem builda `numpy` na Windows.

*Uwaga:* pin na commit GitHuba żyje, dopóki repo żyje i historia nie jest przepisana.
Rozważyć zapisanie w `docs/`, co dokładnie z tego commita jest potrzebne.

---

# CZĘŚĆ C — Decyzja kluczowa (rozwinięcie)

## D1. Jednolity model homoskedastyczny

**To jest decyzja o największym zasięgu w całej pracy.** Wymaga najstaranniejszego akapitu.

### Fakty
`laplace-torch` w wersji 0.2.3 odrzuca wektorowe `sigma_noise` twardym warunkiem w kodzie
(`baselaplace.py`: *„Only homoscedastic output noise supported"*). Tryb subnetwork Laplace,
pozwalający ograniczyć Hessian do głowicy średniej, działa — ale klasa `KronSubnetLaplace`
nie istnieje, więc struktura Kroneckera jest w tym trybie niedostępna.

### Rozważane warianty
1. Heteroskedastyczny + własna implementacja krzywizny — odrzucone: wymaga decyzji
   „które parametry wchodzą do Hessianu" bez źródła w literaturze
2. Heteroskedastyczny + przeskalowanie reszt przez `σ(x)` z `sigma_noise = 1` —
   matematycznie poprawne, ale wymaga akapitu wyjaśniającego, błąd byłby cichy,
   i **nie ratuje Rysunku 3.9**
3. **Jednolity homoskedastyczny** — przyjęte

### Uzasadnienie do rozdziału 4
Główny argument nie jest techniczny: **jednolity model szumu jest warunkiem tego, żeby
obserwowane różnice dało się przypisać przybliżeniu posteriora, a nie modelowi szumu.**
Argument identyczny z tym, który rozdział 3.3 formułuje na rzecz heteroskedastyczności —
zmienia się kierunek unifikacji, nie zasada.

Argumenty wspierające:
- znika wyjątek GP z 3.1
- Rysunek 3.9 staje się wykonalny w trybie pełnosieciowym
- protokół zbliża się do Gala i Hernándeza-Lobato, którzy są homoskedastyczni
- znika ryzyko przecieku dekompozycji (A.3)

### Koszt do nazwania wprost
Wariant heteroskedastyczny opisany w 2.1 i 3.3 pozostaje **nieuruchomiony**.
`sin_hetero` nie mierzy różnic między metodami, tylko ograniczenie wspólnego modelu.
To jest ograniczenie pracy, nie jej właściwość — zapisane w CZĘŚCI F (Ograniczenia).

## D1b. `N = 250` w eksperymentach syntetycznych

Druga decyzja wymuszona przez zachowanie Laplace'a, warta osobnego akapitu.

### Fakty
Sieć 1×50 ma 151 parametrów. Przy `N = 50` macierz GGN (`Σᵢ J(xᵢ)J(xᵢ)ᵀ`) ma rząd ≤ 50,
więc ponad sto kierunków w przestrzeni parametrów nie jest ograniczonych przez dane
i pozostaje przy priorze. Skutek: wąskie piki wariancji epistemicznej (szerokość ~0.02–0.03
w `x`) **wewnątrz zakresu treningowego**, o wysokości 10–50× przekraczającej wartość
na granicy ekstrapolacji. Dotyczy `full` i `kron`; `diag` ich nie ma, bo pomija korelacje.

Zweryfikowane na sześciu seedach w `float32` i `float64`. Współczynnik uwarunkowania
macierzy precyzji posteriora rzędu `10^6` w obu precyzjach — własność problemu, nie
artefakt numeryczny. Wkład rzutowania w `curvlinops` (biblioteka liczy GGN w `float32`
niezależnie od dtype modelu): ≤ 0.04% błędu względnego, czyli nieistotny.

### Rozwiązanie
`N = 250 > 151`. Po zmianie: maksimum wariancji epistemicznej wypada na granicy
ekstrapolacji w 5 z 6 seedów, a w szóstym pozostała struktura ma wysokość porównywalną
z wartością brzegową (1.43 wobec 1.22), nieodróżnialną wzrokowo.

### Uzasadnienie do rozdziału 4 (ZREWIDOWANE 2026-08-26 — patrz niżej; decyzja `N=250`
zostaje, oryginalny argument o `N>p` był niewystarczający)

~~`N = 250` odpowiada reżimowi `N > p` obowiązującemu we **wszystkich** zbiorach UCI
(277–8611). Eksperyment syntetyczny nie różni się więc od benchmarków w wymiarze,
który nie jest przedmiotem badania — a przy `N = 50` różniłby się, i to w sposób
wpływający na jedną z porównywanych metod.~~

**Ten argument obalony przez ablację szerokości sieci (D-width-E5, seedy 0-2,
`sin_homo`): przy `h=200` (`p=601`) `N/p=0,42` — praktycznie ten sam reżim `N<p` co
`N=50`/`h=50` (`50/151=0,33`) — a igła NIE wraca (`max|Δvar_epi|=0,0070`, MNIEJ niż
przy `h=50`'s `0,0112`, nie więcej).** Sam stosunek `N/p<1` nie jest więc wystarczającym
warunkiem powstania igieł — gdyby był, `h=200` przy `N=250` powinno je odtworzyć, a nie
odtwarza.

**Poprawiony mechanizm: przyczyną przy `N=50` była mała BEZWZGLĘDNA liczba punktów, nie
sam stosunek `N/p`.** Przy 50 punktach na `[0,6]` (gęstość ~8,3 pkt/jednostkę) każde
załamanie aktywacji ReLU (D7c) ma dużą szansę wypaść w rzadko próbkowanym otoczeniu —
lokalnie pustym w promieniu porównywalnym z odległością między sąsiednimi punktami
treningowymi — gdzie GGN nie ma żadnej informacji korygującej ten konkretny kierunek
parametrów, niezależnie od globalnego rzędu macierzy. Przy `N=250` (gęstość ~41,7
pkt/jednostkę) gęstość pokrycia jest wystarczająca, żeby żadne pojedyncze załamanie nie
miało takiej pustki wokół siebie — nawet jeśli globalny rząd GGN nadal nie starcza na
wszystkie kierunki parametrów (jak przy `h=200`), brakująca informacja rozkłada się na
kierunki, które nie lokalizują się w jednym punkcie `x`, więc nie tworzy igły.

**Decyzja `N=250` pozostaje słuszna — zmienia się TYLKO uzasadnienie.** Poprawne
sformułowanie do rozdziału 4: `N=250` zapewnia gęstość pokrycia `[0,6]` wystarczającą,
by żadne pojedyncze załamanie ReLU nie miało lokalnie pustego sąsiedztwa w danych —
zweryfikowane bezpośrednio (usunięcie igieł przy `N=50→250`, D1b) i potwierdzone
pośrednio przez to, że sam powrót do reżimu `N<p` (przez zwiększenie `p`, nie
zmniejszenie `N`) NIE odtwarza problemu. `N>p` jako takie zostaje jako fakt o tym
eksperymencie (i faktycznie zgadza się z UCI, `N/p` od 1,8 do 57) i jako dodatkowa,
niezależna korzyść (pełny rząd GGN, brak numerycznej niejednoznaczności rozwiązania) —
ale nie jako mechanizm tłumaczący zniknięcie igieł, bo ten mechanizm jest inny.

## D-width-E5. Ablacja szerokości sieci `h ∈ {50, 100, 200}` — `sin_homo`, seedy 0-2,
konfiguracja E1. `h` NIE zmienione na stałe (zostaje `50`).

**Kryterium wyjściowe (stosunek `std_epi(x=8)/std_epi(x=3)` wyraźnie rosnący dla
bbb/mcd/ensemble jednocześnie) NIE spełnione:**

| method | h=50 | h=100 | h=200 |
|---|---|---|---|
| bbb | 1,421 ± 0,022 | 1,545 ± 0,201 | 1,862 ± 0,122 |
| mcd | 3,153 ± 0,461 | 3,442 ± 0,302 | 3,192 ± 0,495 |
| ensemble | 2,648 ± 1,035 | 3,164 ± 3,733 | 1,415 ± 0,436 |
| laplace | 53,31 ± 4,02 | 38,45 ± 2,99 | 33,71 ± 1,39 |

Tylko bbb rośnie czysto monotonicznie; mcd jest płaskie; ensemble jest niemonotoniczne z
odchyleniem międzyseedowym (3,73 przy `h=100`) większym niż sam efekt — nierozróżnialne
od szumu tym pomiarem. **Decyzja: `h=50` zostaje.**

**Wniosek ważniejszy niż samo kryterium — materiał do rozdziału 5: BBB i Laplace
reagują na pojemność modelu w PRZECIWNE strony.** Stosunek `std_epi(8)/std_epi(3)`
rośnie z `h` dla BBB (1,42→1,55→1,86) i maleje dla Laplace'a (53,3→38,5→33,7,
konsekwentnie, małe odchylenia międzyseedowe w obu przypadkach — to nie szum).
Interpretacja: to dwie rodziny metod z różną strukturą przybliżenia posteriora
(mean-field wariacyjne vs linearyzowane Laplace'a) i pojemność modelu (liczba
kierunków w przestrzeni parametrów, po których każda metoda w ogóle liczy niepewność)
wpływa na kształt niepewności każdej z nich inaczej, nie tylko na jej wielkość. Prace
referencyjne (Blundell i in. 2015, Gal & Ghahramani 2016, Daxberger i in. 2021,
`foong2019`) testują pojedynczą szerokość sieci, więc to zjawisko nie jest tam widoczne
— nie jest to więc powtórzenie znanego wyniku.

**Naprawiony błąd pomiaru czasu treningu (D23): Laplace w pierwszym przebiegu tego
sweepu pokazał `train_time_s` rzędu `0,2-0,7s` zamiast realnych `~5-8s`, bo backbone
Laplace'a (`train_homoscedastic_mlp`, ten sam kod co MAP) trafiał w cache zapisany
przez MAP chwilę wcześniej w tym samym przebiegu (`src/methods/cache.py`, ten sam
config → ta sama pozycja w cache, po `map` a przed `laplace` w kolejności pętli).
Koszt jest raportowaną metryką eksperymentu, nie wolno, żeby cache go zaniżał w cichy
sposób. **Reguła (na stałe, dla każdego przyszłego runnera eksperymentu, nie tylko
E1): jeśli skrypt zapisuje `train_time_s`/`fit_time_s` jako metrykę wyniku, domyślnie
`use_cache=False` — cache jest opt-in (`--use-cache`), nie opt-out.** Naprawione w
`experiments/e1_synthetic.py` (`--use-cache`, domyślnie wyłączony, zamiast `--no-cache`
domyślnie włączonego); `experiments/e0_gp_scaling.py` już miał `use_cache=False` na
sztywno (timing-owy eksperyment z założenia). Do zastosowania przy pisaniu runnera E2.

**E5 (brief sekcja 9) to ablacja GŁĘBOKOŚCI (`głębokość ∈ {1,2,4}`), NIE szerokości —
ten sweep jej NIE realizuje.** Sprawdzone wprost w briefie: „E5 | Depth ablation |
głębokość ∈ {1, 2, 4} × wszystkie metody × 2 zbiory". Ablacja szerokości tutaj to
dodatkowy, sąsiedni wymiar topologii (przydatny dla D1b i D-width-E5's własnego
wniosku o BBB/Laplace). ~~Głębokość wciąż wymaga osobnej ablacji w Etapie 5.~~
**Zrealizowana 2026-08-26/27 — patrz `D-topologia-E5` niżej. E5 zamknięte.**

---

## D-topologia-E5. Głębokość × szerokość, wszystkie sześć metod — **`1×50` ZOSTAJE.** E5 zamknięte.

**Status: E5 (brief sekcja 9, „Depth ablation | głębokość ∈ {1,2,4} × wszystkie metody
× 2 zbiory") jest tym pomiarem ZREALIZOWANE, i to z nadmiarem — brief planował samą
głębokość, zmierzono głębokość ORAZ szerokość w jednej siatce. Nie powtarzać w Etapie 5.**
Razem z `D-width-E5` (szerokość `h ∈ {50,100,200}`, 3 seedy) daje to przebadany zakres
głębokości 1–3 i szerokości 20–200.

Dwa przebiegi: `experiments/e5_depth.py` (głębokość ∈ {1,2}, 6 metod × 2 zbiory ×
3 seedy, kryteria ustalone z góry) i `experiments/depth_exploration.py` (6 topologii ×
6 metod × 2 zbiory, seed 0, porównanie wzrokowe + tabela). Wyniki:
`results/e5_depth.csv`, `results/depth_exploration_summary.csv`, rysunki w
`figures/depth_exploration/` (144 sztuki, wspólna skala, poza katalogami pracy).

### Werdykt i co go rozstrzygnęło

**To jest wynik pomiarowy, nie „nie zadziałało".** Głębokość ROBI to, czego od niej
oczekiwano na podstawie `foong2020` (D14j) — naprawia kształt niepewności między
skupiskami, u obu metod objętych twierdzeniem, i wszystkie trzy kryteria ustalone
z góry dla wariantu `depth=2` zostały spełnione (a: `gap_ratio` bbb 1,161±0,111 →
1,713±0,090 i mcd 0,698±0,022 → 1,130±0,023, przedziały rozłączne, 3 seedy;
b: igła Laplace'a 0,0112 → 0,0055 i `cond` 1,47e6 → 2,52e5, czyli MALEJE;
c: dopasowanie `in_range` się poprawia). Odrzucenie wynika z czynnika, którego
kryteria nie obejmowały, a który okazał się większy.

**Rozstrzyga RMSE ekstrapolacji MAP-a — sama średnia, bez jakiegokolwiek członu
epistemicznego, więc czysty pomiar szkody dla funkcji, a nie dla oszacowania
niepewności:**

| config | głębokość | `p` | RMSE ekstrap. `sin_homo` | RMSE ekstrap. `sin_gap` |
|---|---|---|---|---|
| **1×50** | 1 | 151 | **0,2144** | **0,2254** |
| 1×20 | 1 | 61 | 0,3075 | 0,2763 |
| 2×20 | 2 | 481 | 0,4289 | 0,5464 |
| 2×50 | 2 | 2701 | 0,5687 | 0,7372 |
| 3×20 | 3 | 901 | 0,6632 | 0,6906 |
| 3×50 | 3 | 5251 | 0,7939 | 0,7909 |

**Pasma per głębokość nie zachodzą na siebie: d1 ∈ [0,214; 0,308], d2 ∈ [0,429; 0,569],
d3 ∈ [0,663; 0,794]. Po liczbie parametrów kolejność się łamie — `3×20` (`p=901`) jest
GORSZE niż `2×50` (`p=2701`), przy sieci trzykrotnie mniejszej.** Szkodzi więc
głębokość, nie rozmiar. Zwężanie nie kupuje nic: `1×20` jest gorsze od `1×50` na
wszystkim (RMSE ekstrap. 0,308 vs 0,214, MCD LL ekstrap. −18,5 vs −12,3, `gap_ratio`
ensemble 1,05 vs 1,41), a `2×20` — najtańsza opcja dwuwarstwowa — nadal psuje
ekstrapolację dwukrotnie względem `1×50`.

**Drugi, rozstrzygający powód: Laplace traci pokrycie w ekstrapolacji przy głębszych
sieciach.** PICP@95 ekstrap. (`sin_homo` / `sin_gap`): `1×50` 0,973 / 0,968 →
`2×50` 0,945 / **0,600** → `3×20` 0,578 / 0,510 → `3×50` 0,312 / 0,172. Laplace jest
obecnie jedyną metodą sieciową o wzorowej kalibracji poza danymi; ceną za `gap_ratio`
MCD 0,67 → 1,42 na jednym zbiorze byłaby jej utrata. `2×20` jako jedyna konfiguracja
dwuwarstwowa to pokrycie zachowuje (0,980 / 0,922) — czyli Laplace'owi szkodzi tu
POJEMNOŚĆ, w odróżnieniu od średniej, której szkodzi sama głębokość. Rozróżnienie warte
odnotowania w rozdziale 5, ale nie zmienia werdyktu.

**`DEFAULT_DEPTH = 1` i `hidden = 50` zostają jako stałe domyślne.** `depth` istnieje
w `src/methods/backbone.py` jako jawny parametr każdej z pięciu metod sieciowych i
`depth=1` jest bit-identyczne z backbone'em sprzed jego wprowadzenia
(`tests/test_methods.py::test_depth_one_is_bit_identical_to_pre_depth_backbone`) — więc
wszystkie wcześniejsze liczby E1 odtwarzają się bez zmian.

### NAJWAŻNIEJSZA obserwacja z tego sweepu — do rozdziału 5

**Niepewność MC dropoutu między skupiskami jest czystą funkcją GŁĘBOKOŚCI i jest
NIEZALEŻNA od szerokości.** `gap_ratio` na `sin_gap`, obie szerokości przy każdej
głębokości:

| głębokość | `h=20` | `h=50` |
|---|---|---|
| 1 | **0,67** | **0,67** |
| 2 | **1,14** | **1,14** |
| 3 | 1,42 | 1,61 |

Przy `d=1` i `d=2` wartości są identyczne co do drugiego miejsca po przecinku, mimo
że sieci różnią się liczbą parametrów 2,5-krotnie (`d=1`: 61 vs 151) i 5,6-krotnie
(`d=2`: 481 vs 2701). Dopiero przy `d=3` pojawia się rozbieżność (1,42 vs 1,61).

**Dlaczego to jest mocniejsze niż samo „głębokość pomaga":** twierdzenie `foong2020`
(D14j) mówi o LICZBIE WARSTW UKRYTYCH — nieekspresywność dotyczy sieci
jednowarstwowej, uniwersalność zaczyna się od dwóch — i o niczym innym. Nie mówi
o szerokości. Zmierzony tu efekt izoluje dokładnie tę zmienną, o której mówi
twierdzenie, i pokazuje, że pozostałe wymiary topologii jej nie zastępują.
**To jest najczystsze empiryczne potwierdzenie mechanizmu w całej pracy** — nie
„zmieniliśmy architekturę i coś się poprawiło", tylko „poprawa jest funkcją tego
jednego parametru, którego dotyczy teoria, i tylko jego". BBB zachowuje się mniej
czysto (1,22/1,18 → 1,33/1,68 → 2,31/1,48 — rośnie z głębokością, ale niemonotonicznie
i z wpływem szerokości), co samo w sobie jest do odnotowania: twierdzenie obejmuje obie
metody, empiria potwierdza je czysto tylko dla jednej.

Zastrzeżenia z D14j obowiązują nadal i nie wolno ich tu gubić: twierdzenie jest dla
ReLU (my mamy TanH), a wynik o sieciach głębokich to dowód ISTNIENIA. Poniższe jest
zgodnością empiryczną z przewidywaniem, nie weryfikacją twierdzenia.

### Znane ograniczenie tego pomiaru

**Siatka topologii stoi na JEDNYM seedzie (seed=0)** — zgodnie z metodyką z D14d pkt 1
(faza eksploracji = jeden seed). To wystarcza do tego werdyktu, bo rozstrzygnięcie
opiera się na **rozdzielności pasm rzędu 2×** (d1 vs d2 vs d3 w RMSE ekstrapolacji
MAP-a; PICP Laplace'a 0,97 → 0,31), a nie na różnicach porównywalnych z rozrzutem
międzyseedowym. Arm `{1,2}` ma niezależnie 3 seedy (`e5_depth.py`) i potwierdza
kierunek. **Gdyby ktoś chciał podważyć wnioski o `d=3` albo o konkretnych wartościach
`gap_ratio` przy `h=20`, potrzebne byłyby trzy seedy — tego nie policzono i nie należy
tych liczb podawać z niepewnością, której nie zmierzono.**

**Zaktualizowane 2026-08-31: `results/e5_depth.csv` jest kompletne** (66 wierszy,
3 seedy dla każdej komórki sieciowej) — brakujące trzy komórki zostały dopisane, a plik
przeliczony w całości (patrz część F). Komplet trzech seedów zmienia jeden wniosek:

**Ensemble przy `d=2` jest międzyseedowo niestabilny i jego liczby są NIEOSTATECZNE.**
`gap_ratio` na `sin_gap` wynosi **1,173 ± 0,614** (sd rzędu połowy wartości), wobec
mcd 1,130 ± 0,023 i bbb 1,713 ± 0,090. Przy dwóch seedach wyglądało to na stabilne.
Każde zdanie o wpływie głębokości **na ensemble** trzeba oznaczyć jako nierozstrzygnięte
— trzy seedy nie wystarczają przy tym rozrzucie.

**Werdykt o głębokości to jednak nie podważa**, bo nie stał na ensemble'u: opiera się na
RMSE ekstrapolacji MAP-a (0,214 → 0,429 → 0,663) i PICP Laplace'a (0,97 → 0,31), a te
mają komplet seedów i rozdzielność rzędu 2×. Ensemble był w tej tabeli ilustracją, nie
przesłanką.


---

# CZĘŚĆ D — Sprawy otwarte

| # | Sprawa | Kiedy rozstrzygnąć |
|---|---|---|
| O1 | Klucze `lakshminarayanan2017`, `han2022` — niezweryfikowane. **`foong2020`** (Foong, Burt, Li, Turner, *On the Expressiveness of Approximate Inference in BNNs*, NeurIPS 2020, arXiv 1909.00719) — **zweryfikowany przez autora 2026-08-26, patrz D14j**; wpisać do `.bib` samodzielnie, nie przez agenta. **`verdoja2020`** (Verdoja & Kyrki, *Notes on the Behavior of MC Dropout*, arXiv 2008.02627) — NIEZWERYFIKOWANY, cytowany w D14l, do sprawdzenia przed użyciem. ~~`foong2019`~~ **zweryfikowany** (użytkownik, 2026-08-25): Foong, Li, Hernandez-Lobato, Turner, "In-Between Uncertainty in Bayesian Neural Networks", ICML 2019 Workshop on Uncertainty and Robustness in Deep Learning, arXiv 1906.11537 — patrz D7c i D14c | przed `.bib` |
| O9 | Metoda gap-splitów E3: Foong i in. 2019 usuwają środkową 1/3 osobno DLA KAŻDEGO wymiaru (`D` podziałów na zbiór); nasz plan — jeden podział po najsilniej skorelowanej cesze. Do wyrównania z literaturą przy projektowaniu E3, nie teraz | Etap 4/6, przy E3 |
| O10 | `D14b` (posterior_rho_init sweep, asymetria BBB) jest cytowane w `bbb.py` i w D18 tego pliku, ale treść nigdy nie została tu spisana jako osobna sekcja — do zrekonstruowania z historii sesji albo do ponownego zweryfikowania, nie zgadywać liczb | przed rozdziałem 4 |
| O2 | Definicja QICE — nie implementować przed weryfikacją źródła | przed E2 |
| O3 | Czy marglik stroi też `sigma_noise` → różny człon aleatoryczny Laplace vs MAP przy identycznej średniej | Etap 2 |
| O4 | ~~Liczba epok dla tabeli głównej~~ — **ZAMKNIĘTE 2026-08-28, patrz D30** (per zbiór, maksimum po metodach; sufit na `yacht`; ostrzeżenie do P1 w D31) | zamknięte |
| O5 | ~~Czy wchodzi HMC (E4)~~ — **ZAMKNIĘTE 2026-08-31: E4 ODRZUCONE świadomie.** Powód: GP pełni już rolę punktu odniesienia (L1 dla `sin_homo`, gdzie posterior jest dokładny co do hiperparametrów jądra), a E4 kosztuje dwa–trzy dni pracy i wymusza korektę rozdziału 2.6. **Cena decyzji, do zapisania w ograniczeniach:** dekompozycja aleatoryczna/epistemiczna na UCI zostaje **bez zewnętrznego punktu odniesienia** — mamy L0 i L1 na danych syntetycznych oraz L3 (uporządkowanie w luce, E3) na rzeczywistych, ale nie mamy L2, więc o wartościach `mean_var_epistemic` na UCI nadal nie da się powiedzieć, czy niska wartość znaczy „lepiej", czy „metoda zapada się do punktu" (D22). Zdanie z rozdziału 2.6 o pominięciu MCMC **zostaje bez zmian** — to jest teraz zgodne z tym, co praca robi | zamknięte 2026-08-31 |
| O6 | Wiersze referencyjne BBB i deep ensembles do P13 — odczytać z prac ręcznie | Etap 4 |
| O7 | ~~`Y_RANGE=[-5,5]` niewystarczające przy `γ=1,0`/ReLU~~ — **zamknięte.** Po D7b (TanH domyślnie) i pełnym przeliczeniu E1 (seed=0): najgorszy przypadek to laplace/`sin_hetero`, pasmo [-3,58; 2,23] — mieści się z zapasem. `Y_RANGE` nie wymaga zmiany | zamknięte 2026-08-25 |
| O8 | Liczba epok (D12/O4) a rozrzut ensemble — **zmierzone, patrz D14k** (`experiments/ensemble_epochs_sweep.py`, `results/ensemble_epochs_sweep.csv`). Sama decyzja o liczbie epok dla tabeli głównej pozostaje otwarta (O4) | pomiar zamknięty 2026-08-26; decyzja: Etap 4 |

---

# CZĘŚĆ E — Wyniki nieoczekiwane jako materiał

Rzeczy znalezione po drodze, które nadają się do rozdziału 4 lub 5 jako obserwacje,
a nie tylko szczegóły techniczne:

- **Niespójność w README repozytorium referencyjnego:** komórka `energy`/RMSE podaje
  odchylenie standardowe (0.0605), podczas gdy wszystkie pozostałe są w konwencji
  standard error. Uzasadnia liczenie wszystkiego z plików per podział.
- **`test_rmse` vs `test_MC_rmse`:** opublikowana tabela odpowiada wersji z próbkowaniem
  MC. Pojedynczy przebieg deterministyczny daje 5.45 zamiast 4.82 dla `concrete` —
  ilustracja tego, ile wnosi samo uśrednianie po maskach dropoutu.
- **Rozbieżność „10x" w README a `100_xepochs` w nazwach plików** — rozstrzygnięta przez
  lekturę `experiment.py`. Opublikowane liczby pochodzą z 4000 epok.
- **Błąd konwencji `weight_decay` o czynnik 2** wykryty testem empirycznym, nie przeglądem
  wzoru. Argument za testowaniem konwencji tam, gdzie działa, a nie tam, gdzie jest zapisana.
- **BŁĄD: siatka ewaluacyjna E1 nie miała szumu — PICP/MPIW/interval_score/ECE/QICE mierzyły
  pokrycie gładkiej funkcji, nie obserwacji.** `y_eval` było czystym `sin(x)`, bez dodanego
  szumu, mimo że `y_train` szum miało. Skutek: `PICP@95=1,00` dla wszystkich sześciu metod
  jednocześnie — niemożliwe przy sześciu różnych szerokościach pasm, a dowód nie wymagał
  nawet nowego obliczenia: `LL` dla MAP (0,8363 po naprawie punktu 2 poniżej) zgadzało się
  co do czwartego miejsca z wartością liczoną wprost z `RMSE` i `mean_var_aleatoric`
  potraktowanych jako **dokładny** błąd i **dokładna** wariancja — możliwe tylko wtedy, gdy
  target w ogóle nie ma szumu. Naprawa: `SyntheticDataset` dostał osobne pole
  `y_eval_noisy` (świeże losowanie z tej samej `sigma_fn`, ten sam `rng` co dane
  treningowe — ciągłość strumienia z jednego seeda); `y_eval` (czyste) zostaje wyłącznie
  do rysowania linii „true f(x)". Metryki w `e1_synthetic.py` przełączone na
  `y_eval_noisy`. Sprawdzone i wykluczone dla E2: `load_uci`'s `y_test` to bezpośrednio
  wczytane, rzeczywiste dane UCI — nie ma tam rozróżnienia czysty/zaszumiony do popsucia,
  bo nie ma pojęcia „prawdziwej funkcji" bez szumu.
- **Zakres ewaluacji E1 był asymetryczny: `[-2, 10]`, czyli 2 jednostki w lewo od
  `[0,6]` ale 4 w prawo.** Błąd w oryginalnym briefie, nie w implementacji — poprawiony
  na symetryczne `[-2, 8]` (po 2 jednostki z każdej strony). Konsekwencje starego zakresu:
  prawa strona ekstrapolacji sięgała 64% okresu `sin` (2π≈6,28) dalej niż lewa, więc GP
  (którego niepewność zależy wyłącznie od odległości od danych) wypadał wizualnie lepiej
  po prawej tylko dlatego, że była bliżej; średnia sieci ReLU (ekstrapolacja liniowa)
  rozjeżdżała się z prawdą silniej po stronie z większym zasięgiem. Nie jest to symetryczne
  porównanie metod, jeśli same granice oceny nie są symetryczne.
- **`prior_sigma` → `prior_variance` w `bayesian-torch`.** Klucz `"prior_sigma"` w słowniku
  `dnn_to_bnn` trafia do `LinearReparameterization` jako parametr o nazwie `prior_variance`,
  ale wzór w `kl_div` używa go jako **odchylenia standardowego**. Nazwa sugeruje co innego,
  niż robi. Wykryte przez lekturę wzoru, nie przez test — bo testy zgodności priorów
  sprawdzają `config.py`, nie bibliotekę. Konkretny przykład tezy, że użycie biblioteki
  nie zwalnia z dokumentowania.
- **`laplace-torch 0.2.3` nie istnieje na PyPI.** Deklarowany `requirements.txt` był
  nierozwiązywalny; wykryte wyłącznie dzięki instalacji na czystym środowisku. Reguła
  do zapisania w sekcji o reprodukowalności: instalacja we własnym środowisku nie dowodzi
  niczego o odtwarzalności.
- **Duplikaty w `results/e1_synthetic.csv` po dwóch uruchomieniach tego samego configu.**
  `append_result_row`/`append_generic_csv` realizują dosłownie zasadę „dopisywanie, nie
  nadpisywanie" (sekcja 8) — dwa uruchomienia identycznego eksperymentu (bez zmiany
  seeda/kodu) dają więc dwa identyczne wiersze, nie jeden. Wykryte przy pierwszym
  zajrzeniu do samego pliku wynikowego, nie przy uruchamianiu skryptu. Naprawione
  wyczyszczeniem `results/` przed jednym czystym przeliczeniem — reguła procedury, nie
  zmiana kodu: „append-only" wymaga świadomego czyszczenia przy powtórce niezmienionego
  configu, samo z siebie tego nie pilnuje.
- **Jeden wiersz metryk na cały zakres [-2,10] maskował jakość dopasowania ekstrapolacją.**
  MAP miał RMSE=0,76 na całym zakresie mimo dokładnego dopasowania w `[0,6]`
  (RMSE tam: 0,034) — jedna liczba myliła dwie różne własności. Naprawione: `e1_synthetic.py`
  zapisuje osobny wiersz na `split_type ∈ {in_range, extrapolation, in_gap}` (ten
  ostatni tylko dla `sin_gap`), każdy z pełnym kompletem metryk z sekcji 7. Do
  rozdziału 5: tabela główna powinna też rozbijać RMSE/LL w ten sposób, nie tylko E1.
- **Czas treningu ensemble (M=5) nie jest 5× kosztu MAP — jest ~38×.** Zmierzone
  (sin_homo, jeden model, w izolacji): `batch_size=None` (MAP) → 1,78 s / 2000 epok;
  `batch_size=32` (ensemble) → 12,8 s / 2000 epok — 8× więcej kroków optymalizatora
  (`ceil(250/32)=8` batchy na epokę zamiast 1). `5 × 8 = 40×`, zmierzone `37,8×` —
  zgodne. To nie jest błąd w kodzie ensemble'u, tylko konsekwencja `batch_size=32`
  (domyślna, „do weryfikacji empirycznej", D18) pomnożona przez `M`. Przy E2 (N do
  8611) ta sama arytmetyka daje `ceil(8611/32)=270` batchy/epokę — koszt eksploduje
  z rozmiarem zbioru w sposób, którego MAP/Laplace/BBB (pełny batch lub batch_size
  niezależny od N) nie odczuwają. Decyzja o `batch_size` dla E2 musi to uwzględnić,
  nie tylko liczbę epok.
- **Ten sam prior nie znaczy ten sam wpływ priora.** Sweep `γ ∈ {1,0; 0,5; 0,3; 0,1}`
  × 6 metod × 3 seedy (`sin_homo`, pełny opis w D11b) pokazał, że identyczna wartość
  `γ` degraduje różne metody w bardzo różnym tempie: BBB przechodzi przez `γ=0,1`
  niemal bez szkody (RMSE `in_range` 0,12, LL 0,69), podczas gdy map/mcd/ensemble/
  laplace rozpadają się już w tym punkcie (RMSE ~0,38, LL ~-0,46) — mechanizm: BBB
  wnosi prior przez człon KL w ELBO, pozostałe cztery przez jawną karę `‖θ‖²/(2γ²N)`,
  która skaluje się jak `1/γ²` i przy małym `γ` zaczyna dominować stratę. Ensemble
  degraduje się już przy umiarkowanym `γ=0,3` (widoczne pogorszenie dopasowania), a
  Laplace i MCD przy tym samym `γ=0,3` tracą jakość dopasowania bez korzyści
  proporcjonalnej do tego kosztu — stąd ostateczna decyzja (D11b) zatrzymania się przy
  `γ=1,0` mimo że `γ=0,3` łagodzi jeden konkretny artefakt (skok Laplace'a). Wniosek
  do rozdziału 4: „ta sama `γ`" jako zasada porównywalności (D11) nie implikuje
  „ta sama wrażliwość na `γ`" — struktura, którą prior wchodzi do straty, różni się
  między metodami przy identycznej wartości hiperparametru.
- **Skok `var_epistemic` Laplace'a pokrywa się z załamaniami ReLU co do 6. miejsca po
  przecinku — nie jest artefaktem numerycznym.** Zmierzone lokalizacje skoku na gęstej
  siatce (`sin_homo`): x≈6,011056 i x≈-0,025838. Niezależnie policzone lokalizacje
  załamań aktywacji ReLU sieci (`x_j = -bias_j / weight_j` dla każdej ukrytej jednostki
  `j`) dały dokładnie te same wartości: 6,011056 i -0,025838. Mechanizm: predykcyjna
  wariancja linearyzowanego Laplace'a to `f_var(x) = J(x)ᵀ Σ J(x)`, gdzie `J(x)` to
  jakobian sieci względem wag — dla ReLU `J(x)` jest funkcją schodkową (zmienia się
  skokowo za każdym razem, gdy `x` przekracza próg aktywacji jednej z ukrytych
  jednostek, bo ta jednostka przełącza się między „aktywna"/„nieaktywna", zmieniając,
  które kolumny jakobianu są niezerowe). Skok w `var_epistemic` jest więc realną
  własnością linearyzowanego przybliżenia Laplace'a wokół sieci ReLU, nie błędem
  implementacji ani artefaktem siatki (potwierdzone niezależnością od gęstości siatki
  i od precyzji float32 vs float64). Motywacja dla ablacji E6d (TanH zamiast ReLU,
  rozdział 3.3): TanH ma ciągły jakobian, więc ten konkretny mechanizm nieciągłości
  znika u źródła — sprawdzone empirycznie, wynik w osobnym raporcie/tabeli.


---

## E.1 GP na `wine_quality_red` zapamiętuje powtórzone wiersze — GOTOWY SZKIELET AKAPITU do rozdziału 5

Decyzja autora 2026-08-30: komórka LL dla `gp`/`wine` raportowana jako **dwie liczby**
(+0,424 na pełnym zbiorze testowym, −1,134 bez duplikatów) z przypisem w tabeli, a pełne
wyjaśnienie jako **akapit** w rozdziale 5 — to nie jest usterka do odnotowania drobnym
drukiem, tylko różnica strukturalna między metodą nieparametryczną a parametrycznymi,
z liczbami, których nie ma w pracach referencyjnych (one podają na `wine` jedną liczbę).

Źródła liczb: `experiments/duplicate_ll_diagnostic.py` →
`results/duplicate_ll_diagnostic.csv`, `results/dataset_duplicates.csv`;
`experiments/gp_convergence_diagnostic.py --collapse-check` →
`results/gp_duplicate_collapse.csv`. Odtwarzalność wobec tabeli głównej sprawdzona:
`max |ΔLL| = 4,4·10⁻¹⁶` na 120 komórkach.

### Tabela do wstawienia (średnie po 20 podziałach)

| metoda | LL całość | LL duplikaty | LL bez duplikatów | luka | RMSE duplikaty | RMSE bez dup. |
|---|---|---|---|---|---|---|
| mcd | −0,964 | −0,882 | −0,995 | +0,11 | 0,578 | 0,652 |
| map | −0,972 | −0,874 | −1,009 | +0,14 | 0,575 | 0,657 |
| laplace | −0,955 | −0,876 | −0,985 | +0,11 | 0,575 | 0,657 |
| ensemble | −0,955 | −0,860 | −0,991 | +0,13 | 0,566 | 0,648 |
| bbb | −0,962 | −0,887 | −0,990 | +0,10 | 0,579 | 0,651 |
| **gp** | **+0,424** | **+4,721** | **−1,134** | **+5,86** | **0,0000** | **0,762** |

Na podział przypada średnio 42,6 zduplikowanych i 117,4 niezduplikowanych wierszy
testowych; **we wszystkich 20 podziałach każdy zduplikowany wiersz testowy ma tę samą
wartość celu co jego treningowy bliźniak** (zero przypadków sprzecznych), więc GP nigdy
nie płaci za zapamiętywanie.

### Szkielet akapitu (proza do rozdziału 5, do redakcji)

> **Zdanie 1 — obserwacja, która wygląda na wynik.** On `wine_quality_red` the exact GP
> attains a test log-likelihood of +0.42, more than a nat above every other method
> (−0.95 to −0.97), at an indistinguishable RMSE (0.65 against 0.63–0.64). Read from the
> main table alone, this is the one dataset on which the non-parametric reference is
> decisively better calibrated than the five approximate-inference methods.
>
> **Zdanie 2 — co jest w danych.** `wine_quality_red` contains 240 exactly repeated
> feature vectors among its 1599 rows (15.0%); under the fixed literature splits this puts
> a mean of 26.6% of test rows (18.8–35.0% across splits) in the training set as an exact
> duplicate, always carrying the identical target. Of the six datasets only `concrete`
> (3.7%) and `power_plant` (0.4%) contain repeats at all; `yacht`, `energy` and `kin8nm`
> contain none.
>
> **Zdanie 3 — mechanizm, ze wskazaniem przyczyny w wiarygodności brzegowej.** Repeated
> inputs carrying identical targets are direct evidence, to the marginal likelihood, that
> the observation noise is zero: the fitted `noise_level` sits on sklearn's lower bound of
> 1e-5 on all 20 splits. Collapsing the repeated training rows (195.2 of 1439 per split on
> average) lifts it to 0.574, off the bound on all 20 splits, takes the length-scale from
> 0.62 to 3.61, and brings the reported log-likelihood from +0.424 down to −0.942 — where
> every other method already is. On `energy`, which contains no repeats, the same
> operation changes nothing at all (identical hyperparameters to five figures), which is
> the control. The noise floor is a property of the data, not a failure of the optimiser.
>
> **Zdanie 4 — konsekwencja, z rozdzieleniem metod.** With zero assumed noise the GP
> interpolates: on duplicated test rows its RMSE is exactly 0.0000 and its
> log-likelihood +4.72, against −1.13 on the rest — a gap of 5.86 nats. The five
> parametric methods gain 0.10–0.14 nats on the same rows, a factor of roughly 45 smaller,
> and they cannot do otherwise: a finite-capacity network fitted by (penalised) maximum
> likelihood does not reproduce individual training targets.
>
> **Zdanie 5 — odwrócenie wniosku, czyli po co ten akapit.** Restricted to test rows that
> were not seen in training, the GP is the **worst** of the six methods on this dataset,
> both in log-likelihood (−1.134 against −0.985 to −1.009) and in RMSE (0.762 against
> 0.648–0.657); the duplicates alone contribute +1.56 nats to its reported mean. The
> apparent advantage measures memorisation of seen observations, not the quality of its
> uncertainty.
>
> **Zdanie 6 — sens ogólny, do przeniesienia poza `wine`.** The GP draws the correct
> conclusion about *this dataset* — given these observations, the noise really is zero —
> and a false one about the *phenomenon*. This is the failure mode of a non-parametric
> interpolator on data with repeated measurements, and it is invisible in the reference
> papers, which report a single number per dataset for `wine`.

**Czego w tym akapicie NIE twierdzić:** że duplikaty są błędem w danych (są własnością
zbioru UCI i literatura używa go tak samo); że deduplikacja naprawiłaby porównanie
(zmieniłaby zbiór, unieważniając zgodność z podziałami Gala i porównywalność między
metodami); że podniesienie dolnego ograniczenia `noise_level` jest poprawką (1e-5 to
domyślna granica sklearn, a nie nasza decyzja — podniesienie jej dlatego, że wynik wyszedł
podejrzanie dobrze, byłoby dobieraniem hiperparametru pod wynik).

## E.1b Konstrukcja luki w E3 — dwa odstępstwa od sekcji 5.4 briefu

Obie zmiany wymuszone przez dane, obie zapisane 2026-08-31, obie z pomiarem.
Kod: `experiments/e3_gap_split.py` (`middle_third_membership`, `gap_leak_fraction`),
podział wyników: `experiments/e3_summary.py`.

### Odstępstwo 1: środkowa jedna trzecia po RANDZE, nie po wartości

Brief 5.4: „usunąć z treningu obserwacje z przedziału `[q33, q66]` tej cechy". To działa
dla cechy ciągłej. Na naszych zbiorach **tak zachowują się 4 z 16 kolumn** — reszta ma
tyle powtórzonych wartości, że przedział `[q33, q66]` nie zawiera jednej trzeciej wierszy:

| zbiór | wymiar | wartości unikalnych | udział wierszy w paśmie `[q33, q66]` |
|---|---|---|---|
| energy | 4 (wysokość) | **2** | **1,000** — trening pusty, `ValueError: n must be positive, got 0` |
| energy | 3 | 4 | 0,750 |
| energy | 6 | 4 | 0,625 |
| energy | 7 | 6 | 0,562 |
| energy | 2, 5 | 7, 4 | 0,500 |
| concrete | 1, 2, 4 (żużel, popiół, superplastyfikator) | 185, 156, 111 | ~0,670 — bo `q33 = 0`, cechy mają masę w zerze |
| energy 0–1, concrete 0, 5, 6 | | 12, 278–302 | ~0,335 (poprawnie) |

To nie jest przypadek brzegowy do obsłużenia wyjątkiem — to jest większość kolumn.
Przyjęta konstrukcja (i konstrukcja `foong2019`): sortować po cesze i usunąć **środkową
jedną trzecią posortowanych wierszy**. Zawsze usuwa dokładnie 1/3, niezależnie od
powtórzeń. Rangi liczone raz na pełnym zbiorze, zgodnie z decyzją o wspólnej definicji
luki dla wszystkich 20 podziałów.

### Odstępstwo 2: remisy łamane LOSOWO, nie kolejnością wierszy

Pierwsza implementacja sortowała stabilnie, czyli łamała remisy kolejnością wierszy
w pliku. `energy` jest eksperymentem planowanym i jego wiersze są ułożone według planu
doświadczenia — więc „środkowa jedna trzecia" kolumny o dwóch wartościach wycinała
spójny blok pliku, czyli **dziurę w innej zmiennej niż deklarowana**. Skutek zmierzony
na `gp`/`energy`, podział 0:

| wymiar | `gap_leak_fraction` | remisy po kolejności wierszy | remisy losowo |
|---|---|---|---|
| 4 (2 wartości — **luki nie ma**) | 1,000 | **35,94** | **0,991** |
| 3 | 0,627 | 16,74 | 1,076 |
| 7 | 0,350 | 1,10 | 0,945 |
| 0, 1 (**prawdziwa luka**) | 0,000 | 37,97 | 37,97 |

Wersja z kolejnością wierszy wstawiłaby do tabeli `epi_gap_ratio = 36` dla kolumny,
w której **każdy** zostawiony wiersz treningowy ma wartość z zakresu wierszy usuniętych.
Liczba wyglądałaby wiarygodnie, bo GP rzeczywiście wykrywa luki — ale ta konkretna
pochodziłaby z kolejności wierszy w pliku źródłowym. Po losowym łamaniu remisów
(ziarno 12345) kolumny zdegenerowane raportują ~1, a czyste zostają bez zmian.

### Konsekwencja dla raportowania: dwie tabele, nie jedna średnia

`gap_leak_fraction` = udział zostawionych wierszy treningowych, których wartość cechy
mieści się w zakresie wartości usuniętych. Rozkład:

| zbiór | wymiary z leak < 0,05 (**prawdziwa luka**) | wymiary z leak ≥ 0,05 (**kontrola negatywna**) |
|---|---|---|
| energy | 0 (0,000), 1 (0,000) — ale **1 jest duplikatem 0** | 2 (0,25), 3 (0,63), 4 (1,00), 5 (0,25), 6 (0,44), 7 (0,34) |
| concrete | 0 (0,006), 5 (0,010), 6 (0,000) | 1 (0,51), 2 (0,51), 3 (0,13), 4 (0,51), 7 (0,12) |

**Wynik główny stoi na czterech niezależnych osiach**: jednej na `energy` i trzech na
`concrete`. Pozostałych jedenaście wymiarów to **kontrola negatywna** — tam luki nie ma,
więc `epi_gap_ratio ≈ 1` jest odpowiedzią POPRAWNĄ dla każdej metody, a wzrost byłby
sygnałem błędu, nie zdolności. Średniej po wszystkich ośmiu wymiarach nie raportować:
miesza dwie różne wielkości.

**`energy` 0 i 1 to ta sama luka.** Cechy „relative compactness" i „surface area" są na
tym zbiorze wzajemnie jednoznaczne (12 różnych par dla 12 różnych wartości, korelacja
−0,992), więc ich środkowe jedne trzecie to **identyczny zbiór wierszy** i identyczne
`epi_gap_ratio`. Do wyniku głównego wchodzi jedna z nich; uśrednienie obu ważyłoby tę
samą dziurę podwójnie. Trzy czyste wymiary `concrete` pokrywają się w 0,28–0,37, czyli
na poziomie losowym dla jednych trzecich — to są różne dziury.

Próg 0,05 jest **podziałem raportowania, nie filtrem**: każdy wymiar jest policzony,
zapisany i wypisany, a `gap_leak_fraction` siedzi w `results/e3_gap_ratio.csv` przy
każdym wierszu.

## E.1c Wyniki E3 — cztery tabele, nie jedna (przebieg 2026-08-31)

Źródła: `experiments/e3_gap_split.py` → `results/e3_gap_split.csv` (schemat z sekcji 8,
`split_type ∈ {gap_in, gap_out}`) i `results/e3_gap_ratio.csv` (jeden wiersz na
dopasowanie, z `gap_leak_fraction`); podział na tabele:
`experiments/e3_summary.py` → `results/e3_gap_summary.csv`. 1998 dopasowań, 8 procesów,
jeden wątek na proces.

### Tabela 1 — luki czyste (`gap_leak_fraction < 0,05`), WYNIK GŁÓWNY

Cztery niezależne osie: `energy` wymiar 0 (wymiar 1 to ta sama luka, patrz E.1b),
`concrete` wymiary 0, 5, 6. `epi_gap_ratio`, średnia ± SE:

| metoda | energy | concrete | sin_gap |
|---|---|---|---|
| **mcd** | **1,048 ± 0,008** | **0,988 ± 0,006** | **0,663 ± 0,006** |
| **bbb** | **1,121 ± 0,025** | **0,965 ± 0,005** | **1,274 ± 0,061** |
| laplace | 31,585 ± 1,207 | 2,094 ± 0,039 | 5,123 ± 0,308 |
| gp | 32,749 ± 1,203 | 1,556 ± 0,030 | 2,455 ± 0,195 |
| ensemble | 11,180 ± 0,583 | 2,518 ± 0,056 | 1,755 ± 0,344 |
| map | — | — | — (człon epistemiczny zerowy z konstrukcji) |

**P10 potwierdzone na danych rzeczywistych.** Metody mean-field stoją na 1,0 (MC dropout
na `sin_gap` nawet 0,66 — pasmo się w luce ZWĘŻA), podczas gdy GP i Laplace idą w
trzydziestki. To jest ten sam wniosek co z E1, ale na czterech osiach dwóch zbiorów
rzeczywistych i z kontrolą, a nie na jednym zbiorze syntetycznym.

Uwaga do rozdziału 5: **skala odpowiedzi zależy od zbioru bardziej niż od metody**.
Na `energy` (8 cech, silnie sprzężonych, eksperyment planowany) GP daje 32,7; na
`concrete` (8 cech ciągłych, słabo skorelowanych) — 1,56. Ta sama metoda, ta sama
konstrukcja luki. Wycięcie jednej cechy z ośmiu niezależnych zostawia punkt testowy
podparty siedmioma pozostałymi.

### Tabela 2 — luki częściowe (`0,05 ≤ leak < 0,9`), PER WYMIAR, BEZ ŚREDNIEJ

Uśrednianie tej grupy nie znaczy nic — wartości różnią się w niej ośmiokrotnie przy
praktycznie identycznym `leak`:

| zbiór | wymiar | leak | mcd | bbb | laplace | ensemble | gp |
|---|---|---|---|---|---|---|---|
| energy | 5 | 0,238 | 1,00 | 1,00 | **0,87** | 0,97 | 0,95 |
| energy | 2 | 0,245 | 0,91 | 1,17 | **7,77** | 2,57 | 4,24 |
| energy | 7 | 0,344 | 0,99 | 0,95 | 0,87 | 1,00 | 0,86 |
| energy | 6 | 0,438 | 1,01 | 0,96 | 2,44 | 1,44 | 1,65 |
| energy | 3 | 0,625 | 1,03 | 0,96 | 1,16 | 1,05 | 1,10 |
| concrete | 7 | 0,119 | 0,97 | 0,97 | 1,75 | 1,86 | 1,57 |
| concrete | 3 | 0,132 | 0,93 | 0,96 | 1,99 | 2,25 | 1,45 |
| concrete | 1 | 0,505 | 1,01 | 1,01 | 1,92 | 2,02 | 1,82 |
| concrete | 2 | 0,509 | 1,02 | 1,00 | 1,42 | 1,60 | 1,19 |
| concrete | 4 | 0,512 | 0,94 | 0,96 | 2,60 | 2,68 | 2,00 |

### Tabela 3 — kontrola negatywna właściwa (`leak ≈ 1`)

`energy` wymiar 4 (wysokość budynku, **dwie wartości**): każdy zostawiony wiersz
treningowy ma wartość z zakresu wierszy usuniętych, więc luki fizycznie nie ma.

| mcd | bbb | laplace | ensemble | gp |
|---|---|---|---|---|
| 0,995 | 0,996 | 1,003 | 0,968 | 1,010 |

**Wszystkie pięć ≈ 1,0.** Żadna metoda nie produkuje wzrostu niepewności tam, gdzie nie
ma czego wykrywać. To jest test poprawności całego pomiaru: gdyby któraś metoda dała tu
wzrost, oznaczałoby to błąd konstrukcji, nie zdolność metody.

### Tabela 4 — kontrola losowa (ten sam ubytek danych, bez struktury)

| zbiór | mcd | bbb | laplace | ensemble | gp |
|---|---|---|---|---|---|
| energy | 0,989 | 0,965 | 0,948 | 0,974 | 0,924 |
| concrete | 0,956 | 0,960 | 0,890 | 0,899 | 0,842 |

Wszystkie **0,84–0,99**. Żadna metoda nie podnosi niepewności od samego ubytku jednej
trzeciej danych. Bez tej tabeli `epi_gap_ratio = 2,5` dla ensemble na `concrete` dałoby
się przypisać temu, że model ma mniej danych; z nią nie da się.

### Obserwacja metodologiczna, której nie ma u foong2019

**„Wycięcie środkowej jednej trzeciej wymiaru" nie gwarantuje powstania luki, a pokrycie
w jednej cesze nie wystarcza, żeby stwierdzić, czy luka powstała.** Dowód jest w tabeli 2:
`energy` wymiary 5 i 2 mają `leak` 0,238 i 0,245 — różnica w trzeciej cyfrze — a Laplace
daje na nich **0,89 i 7,77**, czyli blisko dziewięć razy więcej. `gap_leak_fraction` mierzy pokrycie
w JEDNEJ cesze; o tym, czy w rozkładzie łącznym powstała dziura, decyduje sprzężenie
z pozostałymi cechami. Na `energy`, który jest eksperymentem planowanym, cechy są
sprzężone konstrukcyjnie, więc wycięcie środka jednej z nich potrafi zrobić realną dziurę
mimo 24% pokrycia — albo nie zrobić żadnej, zależnie od tego, którą cechę się wytnie.

U `foong2019` problem nie występuje, bo tam zbiory mają cechy ciągłe i mało powtórzeń —
środkowa jedna trzecia wymiaru jest tam zawsze dziurą. **To jest wkład metodologiczny
tej pracy, nie usterka protokołu**: gap split na danych z cechami dyskretnymi lub
sprzężonymi wymaga sprawdzenia, czy luka faktycznie powstała, a nie tylko deklaracji,
że się ją wycięło. Minimalna diagnostyka to `gap_leak_fraction` plus kontrola negatywna
na wymiarze o `leak ≈ 1`.

### Jak wykryto błąd w `sin_gap` — argument za liczeniem tego samego dwa razy

Pierwsza wersja E3 partycjonowała siatkę ewaluacyjną na „luka" i „wszystko poza luką".
Siatka biegnie po `[-2, 8]`, a dane treningowe `sin_gap` leżą w `[0,2] ∪ [4,6]`, więc
„poza luką" zawierało ogony ekstrapolacyjne — czyli obszar, gdzie człon epistemiczny jest
z definicji największy. Mierzone było więc **luka wobec ekstrapolacji**, a nie luka wobec
gęstych danych.

**Sygnałem była sprzeczność między dwoma niezależnymi obliczeniami tej samej wielkości:**
E3 dawało dla GP **0,286** (czyli „w luce GP jest PEWNIEJSZE niż poza nią"), podczas gdy
`results/epistemic_growth.csv` z E1 dawało **3,266**. Żadna z tych liczb nie wygląda
w izolacji na błędną — dopiero ich zestawienie pokazuje, że jedna z nich mierzy co
innego, niż deklaruje. Po poprawce (`gap_out` = nośnik treningowy, ekstrapolacja
wykluczona z obu partycji) E3 daje 2,455, zgodnie co do kierunku z E1.

**Do sekcji o metodologii:** liczenie tej samej wielkości w dwóch miejscach niezależnymi
ścieżkami jest tanie i wykrywa błędy, których nie wykryje żaden test jednostkowy — bo tu
nie było wyjątku ani ostrzeżenia, tylko liczba o poprawnym typie, poprawnym znaku
i wiarygodnym rzędzie wielkości. Ta sama zasada wyłapała wcześniej komórkę `gp`/`wine`
(E.1) i mieszankę wątków w E2 (D23b).

## E.2 P13 — luka wobec gal2016 znika po zrównaniu protokołu (przebieg 2026-08-31)

**Wynik jednym zdaniem:** przy protokole Gala nasza implementacja MC dropoutu odtwarza
jego opublikowane liczby na wszystkich trzech sprawdzonych zbiorach — różnice sparowane
są nieistotne statystycznie (najniższe `p = 0,19`), podczas gdy przy naszym protokole
(E2) były istotne z `p` rzędu `10⁻⁸`–`10⁻²⁴` i miały ten sam znak na 95–100% podziałów.
Rozbieżność w tabeli głównej **nie jest usterką implementacji**, tylko konsekwencją
strojenia: Gal przeszukuje `(p, τ)` wewnątrz każdego z 20 foldów, my trzymamy jedno
`p = 0,1` na całą tabelę (D18).

Źródła: `experiments/p13_gal_protocol.py` → `results/p13_gal_protocol.csv`
(per fold) i `results/p13_gal_protocol_grid.csv` (wszystkie 12 komórek walidacyjnych
każdego foldu). Konwersja współczynnika kary zweryfikowana w
`tests/test_p13_gal_protocol.py`. Protokół odtworzony z `experiment.py` i `net/net.py`
z tego samego przypiętego commita, z którego pochodzą dane (`6eb4497`).

### Tabela do rozdziału 5 (różnice sparowane, 20 podziałów, ten sam podział = te same wiersze testowe)

| zbiór | N | RMSE, protokół E2 | RMSE, protokół Gala | LL, protokół E2 | LL, protokół Gala |
|---|---|---|---|---|---|
| yacht | 277 | +1,031 ± 0,110 (**+155%**, p=1e-8) | +0,056 ± 0,048 (**+8,4%**, p=0,26) | −1,112 ± 0,016 (p=4e-24) | −0,011 ± 0,023 (p=0,63) |
| energy | 691 | +0,652 ± 0,031 (**+121%**, p=1e-14) | +0,020 ± 0,015 (**+3,7%**, p=0,19) | −0,570 ± 0,011 (p=6e-22) | −0,007 ± 0,006 (p=0,24) |
| concrete | 927 | +0,834 ± 0,118 (**+17%**, p=1e-6) | −0,033 ± 0,082 (**−0,7%**, p=0,70) | −0,231 ± 0,018 (p=5e-11) | +0,009 ± 0,017 (p=0,61) |

Znak różnicy przestaje być systematyczny: udział podziałów o zgodnym znaku spada
z 0,95–1,00 (E2) do 0,50–0,65 (protokół Gala), czyli do poziomu rzutu monetą. Na
`concrete` nasza replikacja wypada nieznacznie LEPIEJ od opublikowanej w obu metrykach.

### Wybrane hiperparametry — to jest odpowiedź na pytanie „dlaczego luka rosła na małych zbiorach"

| zbiór | wybrane `p` | wybrane `τ` |
|---|---|---|
| yacht | 0,005 × 13, 0,01 × 7 | 0,75 × 20 |
| energy | 0,005 × 20 | 0,75 × 20 |
| concrete | 0,005 × 12, 0,01 × 8 | 0,075 × 14, 0,05 × 6 |

**Ani razu, w żadnym z 60 foldów, walidacja nie wybrała `p = 0,05` ani `p = 0,1`.**
Nasze domyślne `p = 0,1` (D18) leży 10–20× powyżej wartości, którą protokół referencyjny
wybiera na tych zbiorach — i to jest mechanizm luki, zgodny z tym, co pokazał wcześniej
`p13_dropout_diagnostic.py` (część F.1, pozycja 5: `dropout_p` steruje amplitudą pasma).

### Dwie obserwacje o samym protokole Gala, warte zdania w rozdziale 4

- **Jego własna siatka jest ograniczona krawędzią.** `τ` wypada na górnym krańcu siatki
  w 20/20 foldów `yacht` i `energy` (0,75) oraz w 14/20 `concrete` (0,075); `p` wypada na
  dolnym krańcu (0,005) w 45/60 foldów. „Strojone per fold" znaczy więc „strojone w
  granicach siatki, która sama nie obejmuje optimum" — nie jest to zarzut wobec naszej
  replikacji, bo replikujemy siatkę z jego plików, ale opisując protokół nie należy
  twierdzić, że hiperparametry są dobrane swobodnie.
- **Jego regularyzacja L2 jest praktycznie nieaktywna.** `reg = l²(1−p)/(2Nτ)` przy
  `l = 1e-2` daje `≈ 7·10⁻⁷` dla `yacht` (N=277, τ=0,25, p=0,005) i `3·10⁻⁶` w warunkach
  testu jednostkowego. Wykryte przy pisaniu testu konwersji: kontrola negatywna
  (pominięty czynnik `1/(2σ²)`, czyli błąd ośmiokrotny) zmieniała trajektorię Adama
  po trzech krokach o 1,2·10⁻⁶, czyli o tyle co nic — dlatego test sprawdza konwersję
  także przy sztucznie podbitym `reg`, gdzie ma moc wykrywania. **Nie testowaliśmy
  wariantu bez kary**, więc nie twierdzimy, że kara nie robi nic; twierdzimy tylko, że
  przy tych wartościach jest o rzędy wielkości mniejsza od członu danych.

### Czego ten wynik NIE licencjonuje

Nie jest to powód do zmiany `dropout_p` w tabeli głównej. Tabela główna porównuje sześć
metod przy **wspólnym, jednakowym dla wszystkich** protokole (D18: jedna wartość `p`,
jeden zestaw epok per zbiór, wspólny model szumu); strojenie `p` per fold tylko dla MC
dropoutu dałoby mu przewagę proceduralną, której pozostałe pięć metod nie ma. P13 jest
osobnym przebiegiem walidacyjnym i tak ma być opisany: **różnica między naszą tabelą a
literaturą jest udokumentowaną różnicą protokołu, nie usterką**, i teraz ma to zmierzone
na trzech punktach krzywej (N = 277, 691, 927).

## E.3 Wyniki E6 — trzy ablacje, trzy przewidywania zamknięte (2026-08-31)

`experiments/e6a_mc_samples.py`, `e6c_laplace_structure.py`, `e6d_activation.py` →
`results/e6a_mc_samples.csv`, `e6c_laplace_structure.csv`, `e6d_activation.csv`.
Po tym przebiegu `results/expectations_check.csv` nie ma już pozycji `pending`:
**9 confirmed, 4 refuted, 1 inconclusive**.

### E6a — uzasadnienie `T = 100` (Rysunek 3.7)

Sieć trenowana RAZ na (metoda, zbiór); zmienia się wyłącznie strumień próbkowania,
10 powtórzeń na każde `T`. Mierzona wielkość to **rozrzut między powtórzeniami**, bo `T`
jest własnością estymatora, nie posteriora. Względne odchylenie standardowe `mpiw95`:

| `T` | mcd / sin_homo | bbb / sin_homo | mcd / sin_gap | bbb / sin_gap |
|---|---|---|---|---|
| 2 | 0,0052 | 0,0923 | 0,0051 | 0,1515 |
| 10 | 0,0041 | 0,0899 | 0,0028 | 0,0522 |
| 20 | 0,0019 | 0,0473 | 0,0017 | 0,0218 |
| 50 | 0,0011 | 0,0190 | 0,0016 | 0,0167 |
| **100** | **0,0010** | **0,0176** | **0,0010** | **0,0115** |
| 500 | 0,0002 | 0,0096 | 0,0004 | 0,0073 |

**Dwa wnioski do rozdziału 4.** Po pierwsze, **BBB potrzebuje mniej więcej dziesięć razy
więcej próbek niż MC dropout dla tej samej precyzji estymatora** — przy `T = 100` szum
BBB (0,018) jest większy niż szum MC dropoutu przy `T = 2` (0,005). To nie jest własność
posteriora, tylko wariancji reparametryzacji: MC dropout losuje maski binarne z jednego
rozkładu Bernoulliego na warstwę, BBB losuje pełny wektor wag z rozkładu ciągłego.

Po drugie, **obciążenie samego `T = 100` jest znikome**: wobec `T = 500` różnica `mpiw95`
wynosi −0,03% (mcd) i +1,15% (bbb, `sin_homo`). Czyli `T = 100` jest dla MC dropoutu
hojne (`T = 20` wystarczyłoby z zapasem), a dla BBB rozsądnym kompromisem — nie jest to
wartość, przy której któraś z metod jest systematycznie faworyzowana.

### E6c — struktura kowariancji Laplace'a (Rysunek 3.9, P5, P6)

`epi_extrap_ratio` przy wspólnym priorze (`fixed`), średnia po 3 seedach:

| struktura | sin_homo | sin_gap | `mpiw95` sin_homo (fixed / marglik / unregularised) |
|---|---|---|---|
| diag | **1,148** | **1,031** | 0,530 / 0,529 / 0,534 |
| kron | 7,099 | 4,001 | 0,621 / 0,678 / **2,894** |
| full | 18,352 | 7,816 | 0,819 / 1,149 / **nie liczy się** |

**P6 potwierdzone w obu połowach.** KFAC daje wyraźnie wyższą niepewność poza rozkładem
niż wariant diagonalny (7,10 wobec 1,15; 4,00 wobec 1,03), a diagonalny jest **płaski
dokładnie tak jak MC dropout** przy tej samej aktywacji (mcd/tanh: 1,33 na obu zbiorach).
To jest ta „podobność do dropoutu", o której mówi `ritter2018`, i wychodzi ona liczbowo,
nie tylko jakościowo.

**P5 potwierdzone mocniej, niż brzmi przewidywanie.** Wariant `full`/`unregularised`
**w ogóle nie daje posteriora**: rozkład Choleskiego macierzy precyzji posteriora zawodzi
na 6 z 6 komórek („leading minor of order 14 is not positive-definite"). To jest silniejsze
stwierdzenie niż „przeszacowuje" — nieregularyzowany Laplace pełny jest przy tym `N`
i tej liczbie parametrów **numerycznie nieokreślony**. Tam, gdzie da się go policzyć
(KFAC), kierunek zachodzi jednoznacznie: `mpiw95` 2,894 wobec 0,621 na `sin_homo` (4,7×)
i 3,010 wobec 0,632 na `sin_gap` (4,8×).

**Ten wariant został POLICZONY i udokumentowany jako numerycznie nieokreślony, a nie
pominięty.** Każda z sześciu komórek ma w `results/e6c_laplace_structure.csv` swój wiersz
ze `status="failed"` i pełnym komunikatem w kolumnie `failure`; nie ma tam pustego miejsca
ani wiersza, którego brak trzeba by tłumaczyć. Świadomie nie podnosiliśmy
`prior_precision` do wartości, przy której macierz się faktoryzuje — dobrana w ten sposób
liczba odpowiadałaby na pytanie „jak wąski musi być prior, żeby Laplace pełny dał się
policzyć", a nie na P5.

**To jest wynik MOCNIEJSZY niż przewidywanie, nie jego potwierdzenie.** `ritter2018` mówi,
że nieregularyzowany Laplace pełny *przeszacowuje* niepewność; u nas przy `N = 250`
i 151 parametrach on jej w ogóle nie produkuje. W rozdziale 5 zapisać to jako korektę
w stronę ostrzejszą: przy tej relacji `N` do liczby parametrów wariant nie jest
„źle skalibrowany", tylko niezdefiniowany.

**Obserwacja uboczna do rozdziału 5:** `marglik` („strojony") daje pasma SZERSZE niż
`fixed`, nie węższe (1,149 wobec 0,819 dla `full`), i szybszy wzrost poza zakresem
(28,9 wobec 18,4). To dokłada się do obalenia P7 — strojenie prior precision nie czyni
Laplace'a ostrożniejszym w sensie, w jakim zakłada `daxberger2021`.

### E6d — aktywacja (P4)

`epi_extrap_ratio`, średnia po 3 seedach:

| metoda | sin_homo relu | sin_homo tanh | sin_gap relu | sin_gap tanh |
|---|---|---|---|---|
| **mcd** | **3,665** | **1,334** | **4,360** | **1,329** |
| bbb | 12,522 | 1,096 | 16,103 | 1,011 |
| ensemble | 3,880 | 1,848 | 4,347 | 1,250 |
| laplace | 11,189 | **18,352** | 6,143 | **7,816** |

**P4 potwierdzone dla MC dropoutu**: 2,75× i 3,28× (relu wobec tanh, oba zbiory), przy
błędach standardowych 0,014 dla tanh i 0,596 / 0,105 dla relu.

**Ale to nie jest własność samego ReLU — i to jest wynik, którego nie ma w P4.** Efekt
jest jeszcze silniejszy dla BBB (16,1 wobec 1,01, czyli 16×) i wyraźny dla ensemble'u,
a **dla Laplace'a idzie w drugą stronę**: pod tanh rośnie SZYBCIEJ niż pod relu (18,4
wobec 11,2 na `sin_homo`). Wyjaśnienie jest w mechanizmie: predykcja Laplace'a jest
linearyzowana wokół MAP-u i jej wariancja to `J^T Σ J`, więc zależy od jakobianu, a nie
od tego, jak sieć ekstrapoluje wartość. Gładki jakobian tanh nie tłumi tam wzrostu,
tylko go wygładza. Dla metod próbkujących (mcd, bbb, ensemble) decyduje rozbieżność
próbek, którą liniowa ekstrapolacja ReLU rozdmuchuje.

**To jest POPRAWKA do P4, nie jego potwierdzenie** — i w tej formie idzie do rozdziału 5:

> „Niepewność rosnąca nieograniczenie poza danymi" jest własnością **pary (aktywacja,
> sposób propagacji niepewności)**, nie samej aktywacji. Dla metod próbkujących (MC
> dropout, BBB, deep ensembles) rośnie ona, bo próbki rozbiegają się tym szybciej, im
> szybciej rośnie sama funkcja — a ReLU ekstrapoluje liniowo bez ograniczenia. Laplace
> nie propaguje niepewności przez próbki: jego predykcja jest zlinearyzowana wokół MAP-u,
> a wariancja to `J^T Σ J`, więc zależy od **jakobianu**, a nie od tego, jak sieć
> ekstrapoluje wartość. Gładki jakobian TanH nie tłumi tam wzrostu — on go wygładza, i na
> obu zbiorach syntetycznych daje wzrost SZYBSZY niż ReLU (18,4 wobec 11,2 na `sin_homo`;
> 7,8 wobec 6,1 na `sin_gap`).

`gal2016` formułuje przewidywanie dla MC dropoutu i tam zachodzi (2,75× i 3,28×); błędem
byłoby uogólnienie go na „metody bayesowskie pod ReLU".

## E.4 Cztery obalone przewidywania — gotowy materiał do rozdziału 5

Po jednym zdaniu o najbardziej prawdopodobnej przyczynie, z liczbami. Wszystkie werdykty
liczone w `experiments/expectations_check.py`; to nie jest nowa analiza, tylko zestawienie.

**P2 — „BBB ma węższe pasma niż GP i pokrycie poniżej nominalnego" (`blundell2015`,
`arbel2023`).** Zachodzi na danych syntetycznych (PICP 0,55–0,65, pasma węższe od GP na
2 z 3 zbiorów), a na UCI odwrotnie: BBB jest węższy od GP tylko na 1 z 4 zbiorów,
a jego PICP schodzi poniżej 0,95 tylko na 2 z 6 (zakres 0,917–0,990). **Najbardziej
prawdopodobna przyczyna: niedoszacowanie wariancji przez mean-field ujawnia się przy
`N` małym względem liczby parametrów; przy `N` rzędu tysięcy człon aleatoryczny (uczony,
wspólny dla wszystkich metod) dominuje pasmo i różnice w aproksymacji posteriora się w nim
chowają** — spójne z `epi_ratio` BBB na UCI rzędu kilku procent.

**P3 — „MC dropout przeuwiarygodniony na interpolacji, GP przeszacowuje"
(`gal2016appendix`).** Odwrócone w obie strony: na UCI MC dropout jest POWYŻEJ 0,95 na
5 z 6 zbiorów (0,942–0,994), a GP PONIŻEJ na 4 z 4 (0,919–0,949). **Najbardziej
prawdopodobna przyczyna: `dropout_p = 0,1` bez strojenia per fold (D18) daje pasma
szersze niż protokół Gala** — P13 pokazuje to wprost, bo przy strojeniu `p` walidacja
wybiera 0,005–0,01 w 60 na 60 foldów, a nasze RMSE spada wtedy o 44% na `yacht`. GP
z kolei ma jeden `noise_level` na cały zbiór i nie ma jak podnieść pokrycia lokalnie.

**P7 — „strojony Laplace: pokrycie powyżej nominalnego, wolniejszy wzrost niż GP"
(`ritter2018`, `daxberger2021`).** Obie połowy zawodzą, i to trzykrotnie niezależnie:
PICP ≥ 0,95 na 1 z 6 zbiorów UCI; wzrost przy `x = 8` na `sin_gap` 60,9 wobec 39,6 dla GP
(szybszy, nie wolniejszy); a strojenie (`marglik`) **poszerza** pasma zamiast je
kalibrować (`mpiw95` 1,149 wobec 0,819, wzrost 28,9× wobec 18,4×). **Najbardziej
prawdopodobna przyczyna: `marglik` optymalizuje wiarygodność brzegową linearyzowanego
modelu, a nie pokrycie** — to są dwa różne kryteria i na tych zbiorach rozjeżdżają się
w przewidywalną stronę.

**P11 — „koszt LA ≈ MAP < MCD < BBB < DE(×M)" (`daxberger2021` + pomiar własny).**
Pierwsza połowa zachodzi (46,5 ≈ 49,1 < 51,3 s na `kin8nm`), druga nie: **BBB jest 4,1×
droższy od ensemble'u** (1111,4 wobec 274,4 s), nie tańszy. **Przyczyna jest wprost
w konfiguracji: `elbo_samples = 32` oznacza 32 przebiegi w przód na krok, podczas gdy
ensemble to 5 niezależnych sieci trenowanych po jednym przebiegu** — czyli 32 wobec 5
przebiegów na jednostkę pracy. Przewidywanie zakłada `elbo_samples = 1`, co jest
domyślne w wielu implementacjach BBB, ale u nas zostało podniesione do 32 decyzją
o wariancji estymatora ELBO (D14e). Do rozdziału 5: uporządkowanie kosztów zależy od
jednego hiperparametru estymatora, więc podawać je zawsze z jego wartością.

# CZĘŚĆ L — Literatura wykorzystana w eksperymentach

Zebrane 2026-08-31 z decyzji rozproszonych po D7c, D14c, D14j, D14k, D14l, E.1b, E.3,
O1, O2 i sekcji 11 briefu. **To jest lista robocza do notatki, nie bibliografia** —
w repo nie ma `.tex` ani `.bib` i nic tu nie było z nimi porównywane.

**Jak czytać kolumnę „status":**
- *zweryfikowane w sesji* — dane bibliograficzne albo artefakt (repozytorium, plik)
  sprawdzone bezpośrednio, z zapisaną datą i źródłem;
- *zweryfikowane przez autora* — potwierdzone przez autora pracy w rozmowie, data w O1;
- *NIESPRAWDZONE* — klucz używany w kodzie lub w uzasadnieniu decyzji, ale danych
  bibliograficznych **nikt nie sprawdził**; nie cytować bez weryfikacji.

**Zasada, która obowiązywała przez cały czas i obowiązuje dalej:** danych
bibliograficznych nie uzupełniamy zgadywaniem. Tam, gdzie poniżej jest „—", oznacza to,
że autorów, tytułu albo miejsca publikacji **nie ustalano w tej sesji**, a nie że ich nie
ma. Uzupełnia autor.

## L.1 Pozycje, na których stoją decyzje eksperymentalne

| klucz | autorzy, tytuł, gdzie | link | status | co uzasadnia |
|---|---|---|---|---|
| `gal2016` | — (praca o MC dropout; danych bibliograficznych nie ustalano) | — | **artefakt zweryfikowany w sesji**, praca NIESPRAWDZONA | Metoda `mcd` (`src/methods/mcd.py`). Protokół P13 odtworzony **z kodu**, nie z tekstu: `experiment.py` i `net/net.py` z przypiętego commita, siatki `dropout_rates.txt` / `tau_values.txt`, liczba epok 4000 = 40 × 100. Przewidywania P3, P4, P13 |
| — repozytorium `gal2016` | `yaringal/DropoutUncertaintyExps`, commit `6eb4497628d12b0f300f4b4f6bdc386bebad565c` (2018-08-09) | <https://github.com/yaringal/DropoutUncertaintyExps> | **zweryfikowane w sesji 2026-08-31**: pobrane, przeczytane, sumy kontrolne w `data/uci_splits.checksums.json` (306 plików) | Dane i 20 podziałów literaturowych dla całego E2; siatka `(p, τ)` dla P13; opublikowane wartości per podział (`test_MC_rmse_*`, `test_ll_*`) jako strona referencyjna porównania sparowanego |
| `hernandez-lobato2015` | — (protokół 20 podziałów; danych nie ustalano) | — | **NIESPRAWDZONE** (brief sekcja 11 nakazuje sprawdzić) | Pochodzenie protokołu 20 podziałów UCI, przejętego przez `gal2016` i używanego w E2 i P13 (`docs/datasets.md`) |
| `blundell2015` | — (Bayes by Backprop; danych nie ustalano) | — | **NIESPRAWDZONE** | Metoda `bbb` (`src/methods/bbb.py`); waga członu KL `π_i = 1/M` („Blundell eq. 8", F.1); przewidywanie P2 |
| `lakshminarayanan2017` | — (deep ensembles; danych nie ustalano) | — | **NIESPRAWDZONE** (O1) | Metoda `ensemble`; uzasadnienie `batch_size=128` zamiast pełnego batcha (część różnorodności członków pochodzi z kolejności minibatchy — `src/methods/backbone.py`); przewidywanie P9 |
| `ritter2018` | — (Laplace: struktury kowariancji; danych nie ustalano) | — | **NIESPRAWDZONE** | Wybór trzech struktur `full`/`kron`/`diag` w E6c i Rysunek 3.9; przewidywania P5, P6, P7 |
| `daxberger2021` | — (`laplace-torch`; danych nie ustalano) | biblioteka: `laplace-torch` (pin w `requirements.txt`, commit GitHuba — D27) | **NIESPRAWDZONE** (biblioteka używana i przypięta, praca nie) | Implementacja `laplace`; przewidywania P7, P8, P11. P8 („Laplace zachowuje dokładność MAP") potwierdzone z zerową różnicą po przeliczeniu E2 (D23b) |
| `foong2019` | Foong, Li, Hernández-Lobato, Turner, *In-Between Uncertainty in Bayesian Neural Networks*, ICML 2019 Workshop on Uncertainty and Robustness in Deep Learning | arXiv **1906.11537** | **zweryfikowane przez autora 2026-08-25** (O1) | Pojęcie *in-between uncertainty* i zbiór `sin_gap`; wariant „setup Foonga" w D14c (stałe `σ_o`, prior warstwowy); **konstrukcja gap splitu w E3** — środkowa jedna trzecia osobno dla każdego wymiaru (E.1b); przewidywanie P10 |
| `foong2020` | Foong, Burt, Li, Turner, *On the Expressiveness of Approximate Inference in Bayesian Neural Networks*, NeurIPS 2020 | arXiv **1909.00719** | **zweryfikowane przez autora 2026-08-26** (O1, D14j) | Uzasadnienie ablacji głębokości (E5): wynik o wyrażalności dla ≥ 2 warstw ukrytych jest dowodem istnienia, więc głębokości nie wolno obiecywać jako naprawy — ale wolno ją zmierzyć |
| `verdoja2020` | Verdoja & Kyrki, *Notes on the Behavior of MC Dropout* | arXiv **2008.02627** | **NIESPRAWDZONE** (O1) | Cytowane w D14l przy sweepie `dropout_p` (charakter estymaty niepewności MC dropoutu). **Nie jest podstawą żadnej decyzji o kodzie** — usunięcie go z tekstu niczego nie zmienia w wynikach |
| `gal2016appendix` | — (dodatek do pracy o MC dropout; nie ustalano, czy to osobna pozycja) | — | **NIESPRAWDZONE** | Wyłącznie przewidywanie P3 (obalone: `picp[mcd]` wychodzi POWYŻEJ 0,95 na 5 z 6 zbiorów) |
| `arbel2023` | — (danych nie ustalano) | — | **NIESPRAWDZONE**, nieużywane w kodzie | Współźródło przewidywania P2 (obalonego na UCI) |
| `han2022` | — (definicja QICE; danych nie ustalano) | — | **NIESPRAWDZONE** (O1, O2) | Miała dostarczyć definicji `QICE`. **Kolumna `qice` jest w schemacie CSV pusta i taka zostaje**, dopóki definicja nie zostanie sprawdzona u źródła — świadomie nie wymyślaliśmy własnej metryki pod tą nazwą |
| `kendall2017` | — (heteroskedastyczna niepewność aleatoryczna; danych nie ustalano) | — | **NIESPRAWDZONE** | Wariant heteroskedastyczny **rozważony i odrzucony** (D1): `laplace-torch` odrzuca wektorowe `sigma_noise`, więc jednolity model homoskedastyczny. Praca zostaje opisana w rozdziale 3.3 jako wariant z literatury, nie jako to, co uruchamiamy. Rysunek `img2_1.png` pokazuje sieć dwugłowicową **poza porównaniem głównym** |

## L.2 Rozważone i odrzucone — z powodem

| klucz | co to jest | status | dlaczego odrzucone |
|---|---|---|---|
| `osband2018` | randomised prior functions (dodatkowa, nieuczona sieć priorowa dodawana do każdego członka ensemble'u) | **NIESPRAWDZONE**, nieliczone | Byłoby **zmianą metody, nie wariantem hiperparametru** (D14k): deep ensembles przestałyby być tą metodą, którą opisuje rozdział 3.5, a porównanie sześciu metod przy wspólnym backbone przestałoby mieć wspólny backbone |
| `he2020` | NTKGP (predykcja ensemble'u w granicy neural tangent kernel) | **NIESPRAWDZONE**, nieliczone | Ten sam powód co wyżej (D14k) — inna metoda, nie inna konfiguracja |
| `kendall2017` (model heteroskedastyczny) | druga głowica na `log σ²(x)` | patrz L.1 | Niewykonalne z `laplace-torch` przy zachowaniu symetrii między metodami (D1); pełne uzasadnienie w części C |
| HMC / NUTS jako referencja (E4) | `numpyro`, 4 łańcuchy × 2000 próbek | — | **Odrzucone 2026-08-31 (O5)**: dwa–trzy dni pracy plus korekta rozdziału 2.6, przy GP już pełniącym rolę odniesienia (L1). Cena zapisana: dekompozycja na UCI zostaje bez zewnętrznego punktu odniesienia (D22) |

## L.3 Pojawiło się w rozmowie, ale nieużywane i niesprawdzone

**Czego nie cytować bez weryfikacji** — te pozycje nie stoją za żadną linijką kodu ani
żadnym wynikiem w `results/`:

- `arbel2023` — jedyne wystąpienie to współźródło P2 w tabeli przewidywań;
- `verdoja2020` — jedno zdanie w D14l, bez wpływu na kod;
- `han2022` — definicja QICE, której nie zaimplementowano (O2);
- `gal2016appendix` — nie ustalono nawet, czy to osobna pozycja bibliograficzna, czy
  dodatek do `gal2016`.

**Klucze, które wymagają weryfikacji przed wpisaniem do `.bib`, bo stoją za decyzjami:**
`gal2016`, `hernandez-lobato2015`, `blundell2015`, `lakshminarayanan2017`, `ritter2018`,
`daxberger2021`. Zweryfikowane są tylko `foong2019` i `foong2020` (przez autora) oraz
**repozytorium** `gal2016` (w sesji, z sumami kontrolnymi).

---

# CZĘŚĆ F — Ograniczenia

Sekcja założona 2026-08-27. Zbiera rzeczy, które są ograniczeniem PRACY, nie jej
wynikiem, oraz — poniżej — listę dróg sprawdzonych i odrzuconych.

- **`sin_hetero` nie mierzy różnic między metodami, tylko ograniczenie wspólnego
  modelu** (D1, „Koszt do nazwania wprost"): backbone jest homoskedastyczny, więc
  na zbiorze heteroskedastycznym wszystkie metody sieciowe zawodzą z tego samego
  powodu i zbiór nie różnicuje aproksymacji posteriora.
- **Siatka topologii (D-topologia-E5) stoi na jednym seedzie.** Uzasadnione
  rozdzielnością pasm rzędu 2×, ale liczby dla `d=3` i dla `h=20` nie mają zmierzonej
  niepewności międzyseedowej.
- **`results/e5_depth.csv` przeliczone w całości 2026-08-31** (66 wierszy: 60 komórek
  sieciowych + 6 GP, brak luk). Wcześniej brakowało trzech komórek (`sin_gap`, seed 2:
  `laplace` d2, `ensemble` d1/d2). Przy próbie ich dopisania **wąski przebieg nadpisał
  cały plik** — `main()` zapisywał przez `df.to_csv()`, mimo że skrypt ma funkcję
  `write_merged()` z upsertem po kluczu `(method, depth, dataset, seed)`, napisaną
  wcześniej po dokładnie takim samym wypadku i **niepodpiętą w tym miejscu**. Poprawione
  (`main()` woła `write_merged`), plik odtworzony pełnym przebiegiem.
  **Odtworzenie jest wierne**: przy seedzie 0 wartości zgadzają się co do trzeciego
  miejsca po przecinku z niezależnym plikiem `results/depth_exploration_summary.csv`
  (mcd `gap_ratio` 0,674 → 1,140; laplace 7,318 → 8,053; bbb 1,222 → 1,682), więc
  żadna liczba cytowana wcześniej w tych notatkach nie uległa zmianie.
- **Odtwarzalność wymaga przypięcia LICZBY WĄTKÓW, nie tylko seeda** (zmierzone
  2026-08-31, `experiments/thread_determinism_check.py` →
  `results/thread_determinism_check.csv`). Ten sam seed, ta sama konfiguracja, jedyna
  różnica to `torch.set_num_threads(1)` wobec domyślnych 8: liczba wątków zmienia
  kolejność redukcji w float64, a optymalizator to zaburzenie (rzędu 10⁻¹⁵ na operację)
  wzmacnia. Zmierzone na 15 komórkach (3 zbiory × 5 metod sieciowych, podział 0):

  | zbiór | kroków optymalizatora | metody wrażliwe | maks. \|ΔRMSE\| |
  |---|---|---|---|
  | yacht | 6 000 | żadna z 5 | **0,00** |
  | concrete | 8 000 | ensemble, map, laplace, mcd (4 z 5) | **6,75·10⁻⁴** |
  | kin8nm | 58 000 | ensemble, map, laplace (3 z 5) | 5,71·10⁻⁵ |

  Maksima w całym pomiarze: **ΔRMSE 6,75·10⁻⁴** (concrete/ensemble) i **ΔLL 3,10·10⁻³**
  (kin8nm/ensemble).

- **Hipoteza „im dłuższy przebieg, tym większy rozjazd" jest FAŁSZYWA — i to jest
  właściwe sformułowanie do rozdziału 4.** Naturalne wyjaśnienie brzmiałoby: błąd narasta
  z liczbą kroków. Dane mu przeczą: `concrete` z 8 000 kroków rozjeżdża się **mocniej**
  (6,75·10⁻⁴) niż `kin8nm` z 58 000 (5,71·10⁻⁵), a `yacht` z 6 000 nie rozjeżdża się
  wcale — zero co do bitu na wszystkich pięciu metodach. Decyduje nie długość trajektorii,
  tylko czy przechodzi ona blisko rozgałęzienia (inna kolejność wyboru minibatcha, inna
  strona nieciągłości gradientu, inny basen atrakcji). Narastanie jest **chaotyczne, nie
  monotoniczne w krokach**. W rozdziale 4 nie pisać „przy długich przebiegach błędy
  numeryczne narastają", bo to zdanie jest wygodne i nieprawdziwe; pisać, że wrażliwość
  jest niemonotoniczna i dlatego konfigurację trzeba przypiąć zamiast szacować, gdzie
  zaszkodzi.

- **GP jest odporny, i to z powodu STRUKTURALNEGO — kontrast wart odnotowania.**
  `GPMethod` to sklearn, więc `torch.set_num_threads` w ogóle go nie dotyczy; sprawdzone
  przez właściwą zmienną, `OMP_NUM_THREADS` (plus `OPENBLAS_/MKL_/VECLIB_/NUMEXPR_`)
  ustawianą przed importem numpy, 1 wobec 8 wątków:

  | zbiór | RMSE | constant | length_scale | noise | LML |
  |---|---|---|---|---|---|
  | concrete | 5,260917253269 = | 10,67943065 = | 2,862249331 = | 0,07015233408 = | −416,2996123163 = |
  | wine | 0,641587931676 = | 1,077247408 = | 0,5895671138 = | 1e-05 = | różnica na 10. cyfrze |

  Identyczne do 12 cyfr znaczących. Powód nie jest przypadkiem: **GP nie ma pętli, która
  by zaburzenie narastała.** L-BFGS zbiega do tego samego optimum niezależnie od kolejności
  redukcji, a predykcja to jedno rozwiązanie układu równań, nie 58 000 kolejnych kroków.
  Metody iteracyjne są wrażliwe, metoda z rozwiązaniem w postaci zamkniętej nie —
  to jest zdanie do rozdziału 4 o tym, że kwestia odtwarzalności ma różny ciężar dla
  różnych klas metod. Odnotować też, że **sprawdziliśmy właściwą zmienną**, a nie
  założyliśmy odporność z tego, że „GP nie używa torcha".

- **Ostrzeżenia zbieżności GP na `energy` i `wine` są przypisem, nie usterką**
  (sprawdzone 2026-08-30: `experiments/gp_convergence_diagnostic.py`,
  `results/gp_convergence_diagnostic.csv`, `results/gp_restarts_check.csv`).
  Ostrzega 20/20 podziałów obu zbiorów, ale z dwóch różnych powodów: na `energy`
  `lbfgs` kończy jako ABNORMAL na 19/20 podziałów, a `constant_value` staje na GÓRNYM
  ograniczeniu 1e5 we wszystkich 20; na `wine` nie ma ani jednego ABNORMAL, jest tylko
  `noise_level` na DOLNYM ograniczeniu 1e-5 we wszystkich 20 (mechanizm: powtórzone
  wiersze, patrz część E). Porównanie „podziały z ostrzeżeniem vs bez" jest niewykonalne —
  grupy bez ostrzeżenia nie ma. Rozstrzyga test sparowany: `n_restarts_optimizer` 5 → 20
  na `energy`, podział 0, nie zmienia NICZEGO — log-wiarygodność brzegowa 757,82 w obu,
  RMSE 0,3871, LL −0,4806, hiperparametry co do czwartego miejsca identyczne; rośnie tylko
  liczba ostrzeżeń (3 → 5), bo więcej restartów to więcej nieudanych restartów w logu,
  a zatrzymywane optimum jest to samo. Ostrzeżenia raportują odrzucone restarty i optimum
  leżące na granicy, nie błędną odpowiedź. Do nazwania w rozdziale 4 jednym zdaniem;
  `constant_value` przyklejone do 1e5 na `energy` warto wymienić, bo to jawna degeneracja
  jądra (duża amplituda przy `length_scale` ≈ 11,7), mimo że komórka wypada dobrze
  (RMSE 0,466, LL −0,666 — drugie miejsce na tym zbiorze).
- **`predict_time_ms_per_1k` nie jest porównywalny między zbiorami** (D23a): przy
  31-punktowym zbiorze testowym yacht kolumna mierzy głównie stały narzut wywołania
  `predict`, nie koszt predykcji. Porównywać wewnątrz zbioru.
- **`train_time_s` w `results/ensemble_epochs_sweep.csv` i w `sin_gap` seedy 1–2
  z `results/e5_depth.csv` jest skażony** — te przebiegi dzieliły CPU z innymi
  zadaniami (D14k). Metryki niepewności i dopasowania są deterministyczne, więc
  nietknięte; czasy nie są pomiarem.

## F.1. Drogi sprawdzone i odrzucone dla BBB i MC dropoutu — lista do rozdziału 5 w całości

**Po co ta lista w rozdziale 5:** płaskie/odwrócone pasmo niepewności BBB i MCD nie
jest kwestią doboru hiperparametrów. Poniżej jest jedenaście osi, po których to
sprawdzono, każda z liczbami, każda negatywna dla tych dwóch metod. Wartość
argumentacyjna jest w kompletności listy, nie w pojedynczych pozycjach — pojedynczy
sweep zawsze można zbyć „a próbowaliście X?".

| # | Co sprawdzono | Zakres | Wynik dla bbb / mcd | Źródło |
|---|---|---|---|---|
| 1 | **Aktywacja** | ReLU vs TanH, 6 metod × 3 seedy × 3 zbiory | Negatywny w obie strony: TanH usuwa asymetrię BBB (MPIW L/P 11,7× → 0,99×), ale ZAŁAMUJE kalibrację ekstrapolacji obu — PICP@95 bbb 0,64–0,74 → 0,35–0,49, mcd 0,51–0,83 → 0,35–0,38 | D7b |
| 2 | **Szerokość priora `γ`** | `γ ∈ {1,0; 0,5; 0,3; 0,1}` × 6 metod × 3 seedy | Nie ma wartości korzystnej dla wszystkich naraz; `γ=0,1` rozwala dopasowanie 4 z 5 metod (RMSE 0,10 → 0,38). BBB jako jedyny przechodzi `γ=0,1` bez szkody — jego prior wchodzi przez KL, nie przez jawną karę | D11b |
| 3 | **Wariancja estymatora ELBO / overpruning** | `elbo_samples ∈ {1, 8, 16, 32}`, `sin_homo`, **seed 0** (faza eksploracji, D14e) | Overpruning zbity 43% → 0,66%, a pasmo NADAL płaskie (MPIW(8)/MPIW(3) = 1,21 przy 32 próbkach wobec 1,03 przy 1). Usunięcie overpruningu NIE przywraca wzrostu | D14d, D14e |
| 4 | **Pełny setup `foong2019`** | 4 warianty: bazowy / pełny Foong / samo stałe `σ_o` / same 32 próbki ELBO; `sin_homo`, `tanh`, **seed 0** (D14d) | Żaden nie daje rosnącego pasma. Stosunek MPIW mieści się w 1,01–1,21 we wszystkich czterech; GP w tej samej metryce daje 5,49 | D14d |
| 5 | **`dropout_p`** | `{0,05; 0,10; 0,20; 0,30}` × 3 seedy | Steruje WYŁĄCZNIE amplitudą. Poziom rośnie 1,51×, ale stosunki wzrostu MALEJĄ (1,39 → 1,17); korelacja profili `log std_epi` 0,89–0,96. PICP ekstrap. SPADA 0,305 → 0,066 | D14l |
| 6 | **Dropout na warstwie wejściowej** | jest/nie ma, `d=1`, `sin_homo`, **seed 0** | Usunięty, bo przy `d=1` zeruje jedyną cechę i zawyża człon aleatoryczny: `mean_var_aleatoric` **0,106 z dropoutem wejściowym wobec 0,038 bez** (przeliczone 2026-08-31, `tanh`, 2000 epok, jeden wątek; prawdziwe `σ² = 0,01`). Poprawa dopasowania, nie kształtu niepewności | D15 |
| 7 | **Głębokość** | `1, 2, 3` × 2 szerokości × 2 zbiory (+ `{1,2}` na 3 seedach) | Kształt POPRAWIA — `gap_ratio` na `sin_gap`, średnia ± sd po 3 seedach: mcd **0,698 ± 0,022 → 1,130 ± 0,023**, bbb **1,161 ± 0,111 → 1,713 ± 0,090** (przy `d=3`, tylko seed 0: mcd 1,609, bbb 1,476). Ale kosztem ekstrapolacji średniej: RMSE ekstrap. MAP-a 0,214 → 0,429 → 0,663, pasma rozłączne. **Ensemble jest przy `d=2` niestabilny między seedami** (1,173 ± 0,614, sd rzędu połowy wartości), więc żadnego wniosku o głębokości dla tej metody nie da się z tego wyciągnąć. Odrzucone | D-topologia-E5 |
| 8 | **Szerokość** | `h ∈ {20, 50, 100, 200}`, `sin_homo`, **3 seedy (0–2)** (D-width-E5) | Nie zastępuje głębokości i sama nic nie kupuje: `1×20` gorsze od `1×50` na wszystkim. Przy `h=200` (`N/p=0,42`) tylko bbb rośnie monotonicznie (1,42 → 1,86), mcd płaskie (3,15 → 3,19) | D-width-E5, D-topologia-E5 |
| 9 | **Liczba członków ensemble `M`** | `{5, 10, 20}` × 3 seedy | Steruje PRECYZJĄ oszacowania, nie jego wartością: stosunek 2,65 / 3,09 / 3,62 przy rozrzucie międzyseedowym 1,04. Dotyczy ensemble, nie bbb/mcd — w liście dla kompletności obrazu metod próbkujących | D14h |
| 10 | **Liczba epok ensemble** | `{300, 600, 1000, 2000}` × 2 zbiory × 3 seedy | Mniej epok daje szersze pasmo (stosunek 11,8× vs 4,4×), ale PICP ekstrap. SPADA 0,624 → 0,347 — rozrzut niedotrenowanych członków to szum optymalizacji, nie rozrzut posteriora | D14k |
| 11 | **Punkt odniesienia metryki** | jeden punkt (`x=3`) vs mediana po nośniku, **seed 0** (D14i) | Nie hiperparametr, ale poprawka metodyczna, która ZMIENIŁA wnioski: raportowane 3,9× dla MCD było artefaktem zapadnięcia profilu przy `x≈3,2`; rzeczywista wartość 1,31× | D14i |

**Czego NIE sprawdzono, a bywa wymieniane w tym kontekście — nie wpisywać do listy
jako sprawdzone:**
- **Przeważanie członu KL (β-VI, „KL annealing"/`β ≠ 1`) NIE było osobnym sweepem.**
  Waga KL jest ustalona jako `π_i = 1/M` (jednorodna, Blundell eq. 8) i nigdy nie
  przemiatana. Najbliższym istniejącym pomiarem jest sweep `γ` (poz. 2), bo dla BBB
  prior wchodzi właśnie przez człon KL — ale to NIE to samo co przeskalowanie samego
  członu KL przy ustalonym priorze. Jeśli ta pozycja ma trafić do rozdziału 5, trzeba
  ją najpierw policzyć.
- Randomised prior functions (`osband2018`) i NTKGP (`he2020`) — rozważone, nieliczone,
  bo byłyby zmianą metody, nie wariantem (D14k). Oba klucze NIEZWERYFIKOWANE.


---

# CZĘŚĆ G — Stan prac na 2026-08-31

Zamknięty Etap 4. Zestawienie po to, żeby następna sesja nie musiała rekonstruować,
co jest policzone, z listy plików w `results/`.

## G.1 Policzone i zamknięte

| Eksperyment | Zakres | Artefakty |
|---|---|---|
| **E0** — skalowanie GP | koszt i pamięć wobec `N` | brak CSV w `results/` (przebieg sprzed obecnej struktury); liczby w briefie sekcja 11 — **do przeliczenia, jeśli mają wejść do pracy jako wykres** |
| **E1** — dane syntetyczne | 6 metod × 3 zbiory (`sin_homo`, `sin_hetero`, `sin_gap`), seed 0 | `e1_synthetic.csv`, `e1_sigma_calibration.csv`, `predictions_1d/`, `epistemic_growth.csv`, rysunki `rodzial3_rys/` |
| **E2** — tabela główna UCI | 6 metod × 6 zbiorów × 20 podziałów = 681 wierszy (GP pominięty na `power_plant` (D5) i porzucony na `kin8nm` po 31,1 min) | `e2_uci.csv`, `calibration_curves.csv`, `e2_cost.csv`, `e2_gp_skipped.csv` |
| **E5** — głębokość | 6 metod × 3 głębokości × 2 zbiory × 3 seedy | `e5_depth.csv`, `depth_exploration_summary.csv` — **3 brakujące wiersze** (`sin_gap`, seed 2) |
| **P13** — weryfikacja wobec gal2016 | 3 zbiory × 20 foldów w pełnym protokole Gala | `p13_gal_protocol.csv`, `p13_gal_protocol_grid.csv`, `p13_dropout_diagnostic.csv`, `literature_comparison.csv` |
| Diagnostyki | duplikaty na `wine`, zbieżność GP, zapadnięcie szumu, koszt równoległy vs sekwencyjny | `duplicate_ll_diagnostic.csv`, `dataset_duplicates.csv`, `gp_convergence_diagnostic.csv`, `gp_restarts_check.csv`, `gp_duplicate_collapse.csv` |

Sprawdzenie przewidywań P1–P14: `results/expectations_check.csv`
(`experiments/expectations_check.py`) — **6 potwierdzonych, 4 obalone, 1 nierozstrzygnięte,
3 oczekujące**. Obalone to P2, P3, P7 (wszystkie o kalibracji: przewidywane uporządkowanie
PICP nie zachodzi na UCI, a na `sin_*` tylko częściowo) i P11 (BBB jest 4,1× droższy od
ensemble'u, nie tańszy). To są wyniki do opisania, nie usterki.

## G.2 Zostaje do policzenia

- **E3 — gap splits na UCI.** Nie ruszone. Punkt otwarty O9: Foong i in. usuwają środkową
  1/3 osobno dla każdego wymiaru (`D` podziałów na zbiór), nasz plan zakłada jeden podział
  po najsilniej skorelowanej cesze. Do rozstrzygnięcia przed uruchomieniem.
- **E6a — ablacja `T`** (liczba przebiegów MC dla mcd/bbb). Domyślne `T=100` jest wybrane
  arbitralnie i to jedyna rzecz, która to uzasadni.
- **E6c — warianty Laplace'a** (`full`/`kron`/`diag`, `fixed`/`marglik`/`unregularised`).
  Blokuje P5 i P6, oba obecnie `pending`. Kod wariantów już istnieje w `src/methods/laplace.py`.
- **E6d — ablacja aktywacji** (relu vs tanh). Blokuje P4. D7b zmierzył to raz, ale nie
  zostawił CSV, więc do rozdziału 5 trzeba przeliczyć.

## G.3 Otwarte decyzje

- **E4 / HMC** — czy w ogóle wchodzi. To decyzja o zakresie pracy, nie o harmonogramie:
  wejście HMC oznacza `numpyro` (jedyne dopuszczone odstępstwo od tabeli zależności),
  punkt odniesienia „prawdziwego" posteriora dla dekompozycji niepewności, oraz korektę
  zdania o pominięciu MCMC w rozdziale 2 (O5, errata A.1). Bez HMC dekompozycja
  aleatoryczna/epistemiczna na UCI zostaje bez punktu odniesienia (D22).
- **Komórka `gp`/`wine`** — sposób raportowania rozstrzygnięty (dwie liczby + akapit, E.1),
  ale sam akapit trzeba napisać w rozdziale 5.
- **Klucze bibliograficzne** — `lakshminarayanan2017`, `han2022`, `verdoja2020`
  niezweryfikowane (część D, O1). Weryfikuje autor.
