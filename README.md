# System Rozpoznawania Emocji Twarzy z wykorzystaniem CNN

Projekt zaliczeniowy zrealizowany w ramach przedmiotu **Sztuczna Inteligencja**. Celem projektu było opracowanie oraz implementacja systemu umożliwiającego automatyczne rozpoznawanie emocji na podstawie obrazu twarzy z wykorzystaniem **Konwolucyjnej Sieci Neuronowej (Convolutional Neural Network – CNN)**.

Model został wytrenowany na zbiorze danych **FER-2013** i służy do klasyfikacji emocji na podstawie pojedynczych zdjęć. Działanie systemu obejmuje dwa etapy: detekcję twarzy w obrazie oraz analizę wyodrębnionego obszaru twarzy przez wytrenowaną sieć neuronową.

System rozpoznaje następujące klasy emocji:
- Złość
- Odraza
- Strach
- Radość
- Smutek
- Zaskoczenie
- Neutralny

Wynik predykcji prezentowany jest w postaci nazwy emocji oraz wartości procentowej, określającej poziom pewności klasyfikacji. Wartość ta (Confidence Score) pochodzi z warstwy wyjściowej sieci neuronowej z funkcją aktywacji **Softmax**, która zwraca rozkład prawdopodobieństwa dla wszystkich 7 klas. Wysokie wartości (np. >90%) wskazują na dużą pewność predykcji, natomiast niższe mogą wynikać z niejednoznacznego wyrazu twarzy lub ograniczonej jakości obrazu wejściowego.

## Wykorzystane technologie

Projekt został zaimplementowany w środowisku **Python 3.13** z użyciem następujących bibliotek:
- **TensorFlow / Keras** – budowa oraz trening modelu CNN
- **OpenCV (cv2)** – przetwarzanie obrazu i detekcja twarzy (Haar Cascade)
- **NumPy** – operacje numeryczne i macierzowe
- **Kagglehub** – automatyczne pobieranie zbioru danych

## Instrukcja uruchomienia

### Instalacja zależności

pip install -r requirements.txt

### Trening modelu (opcjonalnie)

W celu wytrenowania modelu od podstaw należy uruchomić:

python train.py

Skrypt automatycznie pobierze zbiór danych FER-2013 oraz przeprowadzi proces uczenia. Po zakończeniu treningu w katalogu projektu zostanie utworzony plik z zapisanym modelem:

moj_model_emocji.h5

### Predykcja (testowanie modelu)

Aby przeprowadzić predykcję na wybranym obrazie:
1. Umieść plik `.jpg` lub `.png` w katalogu projektu.
2. Otwórz plik `predict.py` i ustaw nazwę pliku w zmiennej:

SCIEZKA_DO_ZDJECIA = "nazwa_pliku.jpg"

3. Uruchom program:

python predict.py

## Struktura projektu

train.py – pobieranie danych, preprocessing, definicja architektury CNN, trening
predict.py – wczytanie modelu i predykcja emocji na obrazach
moj_model_emocji.h5 – zapisany model sieci neuronowej
dataset/ – zbiór danych treningowych i testowych (tworzony automatycznie)

## Autorzy

Szymon Nawracała
Mikołaj Ziółek