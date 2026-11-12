#PARTE 3: Elaborazione immagine 2D(Amplitude Encoding e 2D-QFT): prendo un immagine 4x4 pixel in scala di grigi, la converto
#in uno stato quantistico a 4 qubit tramite Amplitude Encoding ed eseguo la 2D-QFT, confrontandola con scipy.fft2.
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
from scipy.fft import fft2

#DEFINIZIONE IMMAGINE E NORMALIZZAZIONE
#Definisco una matrice 4x4 di 16 pixel che rappresenta una scacchiera bianca e nera.
immagine_4x4 = np.array(
    [
        [255.0, 0.0, 255.0, 0.0],
        [0.0, 255.0, 0.0, 255.0],
        [255.0, 0.0, 255.0, 0.0],
        [0.0, 255.0, 0.0, 255.0],
    ],
    dtype=complex,
)

#Converto la matrice bidimensionale in un vettore 1D di 16 elementi ordinati per righe.
vettore_immagine = immagine_4x4.flatten()

#Calcolo la norma euclidea.
norma = np.linalg.norm(vettore_immagine)
vettore_norm = vettore_immagine / norma

#CALCOLO CLASSICO
#Eseguo la Trasformata di Fourier 2D classica sull'immagine normalizzata usando norm="ortho".
#Calcolo poi il modulo |X[Kr,Kc]| per ottenere lo spettro di frequenza bidimensionale.
spettro_classico_2d = fft2(immagine_4x4 / norma, norm="ortho")
moduli_classici_2d = np.abs(spettro_classico_2d)

#COSTRUZIONE CIRCUITO QUANTISTICO 2D
#Inizilizzo un circuito con 4 qubit. qubit 0 e 1 per colonna(c), 2 e 3 per riga(r).
num_qubits = 4
qc = QuantumCircuit(num_qubits)

#Mappo i 16 valori di vettore_norm direttamente nelle ampiezze del registro quantistico a 4 qubit.
qc.initialize(vettore_norm, range(num_qubits))

#Creo e applico il gate QFT a 2 qubit sui primi due qubit, così applico
#la Trasformata di Fourier 1D lungo la direzione delle colonne.
qft_colonne = QFT(num_qubits=2, do_swaps=True).to_gate(label="QFT_Col")
qc.append(qft_colonne, [0, 1])

#Creo e applico il gate QFT a 2 qubit sugli ultimi due qubit, questo applica la trasformata di Fourier 1D
#lungo la direzione delle righe, completando la scomposizione Row-Column 2D-QFT.
qft_righe = QFT(num_qubits=2, do_swaps=True).to_gate(label="QFT_Righe")
qc.append(qft_righe, [2, 3])

#SIMULAZIONE E RICOSTRUZIONE DELLA MATRICE 2D
#Calcolo analiticamente lo stato quantistico finale del sistema, estraendo il vettore colonna di 16 ampiezze complesse.
stato_finale = Statevector.from_instruction(qc)
vettore_spettro_quantistico = stato_finale.data

#Prendo il vettore 1D di 16 ampiezze trasformate e lo rimodello nella matrice originale 4x4, estraendone i moduli delle frequenze.
spettro_quantistico_2d = vettore_spettro_quantistico.reshape((4, 4))
moduli_quantistici_2d = np.abs(spettro_quantistico_2d)

print("=" * 60)
print("PARTE 3: ELABORAZIONE IMMAGINI 2D - 2D-QFT vs SciPy FFT2")
print("=" * 60)
print("Matrice Moduli Classici 2D (SciPy):\n", np.round(moduli_classici_2d, 3))
print("-" * 60)
print(
    "Matrice Moduli Quantistici 2D (Qiskit):\n",
    np.round(moduli_quantistici_2d, 3),
)
print("-" * 60)

#Verifico che la matrice ottenuta dal circuito quantistico sia identica a quella generata da SciPy.
esito_2d = np.allclose(moduli_classici_2d, moduli_quantistici_2d)
if esito_2d:
    print("ESITO: 2D-QFT CORRETTA! ✅ (Coincide perfettamente con SciPy FFT2)")
else:
    print("ESITO: NON CORRETTO! ❌")
print("=" * 60)


#VERIFICA COSTO COMPUTAZIONALE
N_lato = immagine_4x4.shape[0]
M_pixel = immagine_4x4.size
n_asse = int(np.log2(N_lato))

#Costo FFT 2D classica (algoritmo Row-Column di scipy.fft.fft2): 2*N*(N log2 N) = O(N^2 log2 N).
costo_fft_2d = int(2 * (N_lato**2) * np.log2(N_lato))

#Costo 2D-QFT quantistica: estraggo le porte reali dai due gate (qft_colonne e qft_righe).
#Scompongo il blocco qft_colonne e conto quante porte ci sono.
porte_col = qft_colonne.definition.decompose().size()
#Faccio uiguale ma per le righe.
porte_rig = qft_righe.definition.decompose().size()
#Sommo ottenendo il costo reale totale.
porte_qft_2d_empirico = porte_col + porte_rig

#Calcolo il numero previsto dalla formula teorica.
porte_qft_2d_teorico = 2 * (n_asse + (n_asse * (n_asse - 1)) // 2 + (n_asse // 2))

#Verifico che il conteggio reale coincida con querllo teorico.
assert porte_qft_2d_empirico == porte_qft_2d_teorico

print("\nCOSTO COMPUTAZIONALE 2D (Immagine 4x4 -> M=16 pixel, 4 qubit totali)")
print("-" * 60)
print(f"FFT 2D (scipy.fft2) -> Teorico O(N^2 log2 N): {costo_fft_2d} operazioni")
print(f"2D-QFT (Qiskit)     -> Empirico: {porte_qft_2d_empirico} porte | Teorico O(n^2): {porte_qft_2d_teorico} porte")
print(f"Dettaglio circuito 2D: {porte_col} porte (QFT_Col) + {porte_rig} porte (QFT_Righe)")
print("=" * 60)

#Verifico l'andamento teorico vs empirico al crescere della risoluzione NxN (da 2x2 a 64x64 pixel)
n_assi = np.arange(1, 7)
N_lati = 2**n_assi
M_pixels = N_lati**2

curva_fft_2d = 2 * (N_lati**2) * n_assi
curva_qft_2d_emp = [2 * QFT(num_qubits=int(na), do_swaps=True).decompose().size() for na in n_assi]
curva_qft_2d_teo = 2 * (n_assi + (n_assi * (n_assi - 1)) // 2 + (n_assi // 2))
assert list(curva_qft_2d_emp) == list(curva_qft_2d_teo)

#GENERAZIONE GRAFICO COSTO COMPUTAZIONALE
fig_c2, ax_c2 = plt.subplots(figsize=(8, 4.8))
ax_c2.plot(M_pixels, curva_fft_2d, "o--", label=r"FFT 2D (scipy.fft2) — $\mathcal{O}(N^2 \log_2 N)$", color="#1f77b4", linewidth=2)
ax_c2.plot(M_pixels, curva_qft_2d_emp, "s-", label=r"2D-QFT (Empirico = Teorico) — $\mathcal{O}(n^2)$", color="#ff7f0e", linewidth=2)
ax_c2.set_xscale("log", base=2)
ax_c2.set_yscale("log", base=10)
ax_c2.set_xlabel("Numero totale di pixel M = N × N")
ax_c2.set_ylabel("Operazioni / Porte quantistiche (scala log)")
ax_c2.set_title("Verifica Costo Computazionale 2D: FFT2 Classica vs 2D-QFT")
ax_c2.grid(True, which="both", linestyle="--", alpha=0.6)
ax_c2.legend()
plt.tight_layout()
plt.savefig("costo_computazionale_2d.png", dpi=300)
plt.close(fig_c2)

#GENERAZIONE FIGURA
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

im0 = axes[0].imshow(
    np.real(immagine_4x4), cmap="gray", interpolation="nearest"
)
axes[0].set_title("1. Immagine Originale (4x4)")
axes[0].set_xticks(range(4))
axes[0].set_yticks(range(4))
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(moduli_classici_2d, cmap="viridis", interpolation="nearest")
axes[1].set_title("2. Spettro 2D Classico (SciPy fft2)")
axes[1].set_xticks(range(4))
axes[1].set_yticks(range(4))
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(
    moduli_quantistici_2d, cmap="viridis", interpolation="nearest"
)
axes[2].set_title("3. Spettro 2D Quantistico (Qiskit 2D-QFT)")
axes[2].set_xticks(range(4))
axes[2].set_yticks(range(4))
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle(
    "Elaborazione Immagini Quantistica: Validazione 2D-QFT",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()

nome_grafico_2d = "spettro_2d_immagine_confronto.png"
plt.savefig(nome_grafico_2d, dpi=300, bbox_inches="tight")
print(
    f"\nGrafico 2D salvato con successo come: '{nome_grafico_2d}'"
)
plt.show()