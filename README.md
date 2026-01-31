# System Rozpoznawania Emocji Twarzy

Projekt zaliczeniowy zrealizowany w ramach przedmiotu **WDSI**. Celem projektu było opracowanie oraz implementacja systemu umożliwiającego automatyczne rozpoznawanie emocji na podstawie obrazu twarzy.

Model został wytrenowany na zbiorze danych **FER-2013** i służy do klasyfikacji emocji na podstawie pojedynczych zdjęć. Działanie systemu obejmuje dwa etapy: 
1. 📸 **Detekcję twarzy** w obrazie.
2. 🤖 **Analizę obszaru twarzy** przez wytrenowaną sieć neuronową.

System rozpoznaje następujące klasy emocji:
System klasyfikuje twarz do jednej z 7 kategorii:
* 😠 **Złość**
* 🤢 **Odraza**
* 😨 **Strach**
* 😄 **Radość**
* 😢 **Smutek**
* 😲 **Zaskoczenie**
* 😐 **Neutralny**

### 🎯 Interpretacja Wyników
Wynik predykcji prezentowany jest w postaci nazwy emocji oraz **wartości procentowej** (Confidence Score).
* Wartość ta pochodzi z funkcji aktywacji **Softmax** 📉.
* **Wysokie wyniki (>90%)** ✅ oznaczają dużą pewność modelu.
* **Niższe wyniki** ⚠️ mogą wynikać z niejednoznacznej mimiki lub słabej jakości zdjęcia.

## 🛠️ Wykorzystane Technologie
Projekt został zaimplementowany w środowisku **Python 3.13** z użyciem bibliotek:

* **TensorFlow / Keras** – budowa oraz trening modelu CNN.
* **OpenCV (cv2)** – przetwarzanie obrazu i detekcja twarzy (Haar Cascade).
* **NumPy** – operacje numeryczne i macierzowe.
* **Kagglehub** – automatyczne pobieranie zbioru danych.

## 🚀 Instrukcja uruchomienia

### 📦 Instalacja bibliotek

`pip install -r requirements.txt`

### 🏋️Trening modelu (opcjonalnie)

W celu wytrenowania modelu od podstaw należy uruchomić:

`python trening.py`

Skrypt automatycznie pobierze zbiór danych FER-2013 oraz przeprowadzi proces uczenia. Po zakończeniu treningu w katalogu projektu zostanie utworzony plik z zapisanym modelem:

`moj_model_emocji.h5`

### 🔮 Predykcja (testowanie modelu)

Aby przeprowadzić predykcję na wybranym obrazie:
1. Umieść plik `.jpg` lub `.png` w katalogu projektu.
2. Otwórz plik `predykcja.py` i ustaw nazwę pliku w zmiennej:

`SCIEZKA_DO_ZDJECIA = "nazwa_pliku.jpg"`

3. Uruchom program:

`python predykcja.py`

## 📂 Struktura projektu

`trening.py` – pobieranie danych, preprocessing, definicja architektury CNN, trening

`predykcja.py` – wczytanie modelu i predykcja emocji na obrazach

`moj_model_emocji.h5` – zapisany model sieci neuronowej

`dataset/` – zbiór danych treningowych i testowych (tworzony automatycznie)


## 👥 Autorzy

* Szymon Nawracała

* Mikołaj Ziółek






