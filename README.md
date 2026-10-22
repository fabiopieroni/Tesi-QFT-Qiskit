# Quantum Fourier Transform (QFT) - Implementazione e Simulazioni NISQ

## Descrizione del Progetto
Il progetto analizza il formalismo della Quantum Fourier Transform (QFT), la sua scomposizione in porte logiche quantistiche elementari, l'applicazione all'elaborazione di immagini 2D e la modellazione del rumore su hardware NISQ.

## Struttura del Codice

1. `1_validazione_1d.py`: Validazione spettrale 1D (NumPy FFT vs Qiskit QFT) con Amplitude Encoding.
2. `2_circuito_custom.py`: Scomposizione del circuito QFT a 2 qubit in porte elementari.
3. `3_qft_2d_immagine.py`: Applicazione della 2D-QFT per il processing di un'immagine 4x4 via decomposizione riga-colonna e validazione con `scipy.fft.fft2`.
4. `4_simulazione_rumore_nisq.py`: Modellazione degli errori hardware reali (depolarizzazione, readout error) con `qiskit-aer` ed estrazione della fedeltà spettrale.

## Installazione delle dipendenze
```bash
pip install -r requirements.txt
```

## Esecuzione degli script
```bash
python 1_validazione_1d.py
python 2_circuito_custom.py
python 3_qft_2d_immagine.py
python 4_simulazione_rumore_nisq.py
```