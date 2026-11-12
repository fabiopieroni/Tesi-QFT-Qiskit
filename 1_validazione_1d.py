#PARTE 1: Validazione 1D Base, eseguo la FFT classica con numpy.fft e la confronto con la QFT di Qiskit. 
#OBIETTIVO: Verificare matematicamente che la simulazione quantistica dia gli stessi identici risultati della teoria classica.

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector

#PREPARAZIONE DEL SEGNALE
#Definisco un segnale di test a N=4 bit (un segnale alternato [1,0,1,0])
x_classico = np.array([1.0, 0.0, 1.0, 0.0], dtype=complex)

#Normalizzo(Amplitude Encoding), perchè i principi della meccanica quantistica impongono che la somma dei quadrati dei moduli delle ampiezze
#di uno stato sia sempre uguale a 1. Dividendo il vettore per la sua norma euclidea, converto il segnale classico in uno stato quantistico valido.
norma = np.linalg.norm(x_classico)
x_norm = x_classico / norma

#Calcolo la Trasformata di Fourier Discreta classica(DFT). Con norm="ortho" applico la normalizzazione ortogonale(divido per radq di N), 
#garantendo la conservazione dell'energia(perchè le trasformazioni quantistiche sono unitarie e mantengono la norma). 
fft_classica = np.fft.fft(x_norm, norm="ortho")

#COSTRUZUINE E SIMULAZIONE DEL CIRCUITO QUANTISTICO
#Per rappresentare N=4 campioni servono 2 qubit.
num_qubits = 2

#Creo il registro quantistico composto da 2 qubit(inizializzati di default a |00>).
qc = QuantumCircuit(num_qubits)

#Mappo il vettore classico Xnorm all'interno delle ampiezze del registro quantistico.
qc.initialize(x_norm, range(num_qubits))

#Genero l'operatore unitario della QFT a 2 qubit. do_swaps=True inserisce le porte SWAP finali per correggere l'inversione dell'ordine tipica dell'algoritmo QFT.
qft_gate = QFT(num_qubits=num_qubits, do_swaps=True).to_gate()
qc.append(qft_gate, range(num_qubits))

#Simulo il circuito calcolando analiticamente il vettore di stato finale algebrico ed estraendone le ampiezze complesse trasformate.
stato_finale = Statevector.from_instruction(qc)
spettro_quantistico = stato_finale.data

#CONFRONTO E GENERAZIONE GRAFICO
#Calcolo il modulo dei numeri complessi |X[k]| sia per il risultato classico che per quello quantistico, ottenendo lo spettro di ampiezza.
moduli_classici = np.abs(fft_classica)
moduli_quantistici = np.abs(spettro_quantistico)

print("=" * 60)
print("TEST VALIDAZIONE 1D: FFT CLASSICA vs QFT QUANTISTICA")
print("=" * 60)
print(f"Segnale Inserito (Normalizzato): {np.round(x_norm, 3)}")
print("-" * 60)
print(f"Moduli Spettro Classico (NumPy):     {np.round(moduli_classici, 4)}")
print(f"Moduli Spettro Quantistico (Qiskit): {np.round(moduli_quantistici, 4)}")
print("-" * 60)

#Confronto se i due vettori sono matematicamente identici elemento per elemento, tenendo conto delle minime approssimazioni di macchina.
esito_confronto = np.allclose(moduli_classici, moduli_quantistici)
if esito_confronto:
    print("ESITO: CONFRONTO POSITIVO! ✅ (Spettro Classico == Spettro Quantistico)")
else:
    print("ESITO: CONFRONTO NEGATIVO! ❌")
print("=" * 60)


#VERIFICA COSTO COMPUTAZIONALE
N_1d = len(x_classico)

#Costo FFT 1D (algoritmo Cooley-Tukey usato da np.fft.fft): O(N * log2(N)).
costo_fft_1d = int(N_1d * np.log2(N_1d))

#Costo QFT 1D: estraggo le porte reali dal gate (qft_gate).
qft_1d_decomp = qft_gate.definition.decompose()
porte_qft_1d_empirico = qft_1d_decomp.size()
porte_qft_1d_teorico = num_qubits + (num_qubits * (num_qubits - 1)) // 2 + (num_qubits // 2)
assert porte_qft_1d_empirico == porte_qft_1d_teorico

print("\nCOSTO COMPUTAZIONALE 1D (N=4 campioni, n=2 qubit)")
print("-" * 60)
print(f"FFT Classica (np.fft.fft) -> Teorico O(N log2 N): {costo_fft_1d} operazioni")
print(f"QFT Quantistica (Qiskit)  -> Empirico: {porte_qft_1d_empirico} porte | Teorico O(n^2): {porte_qft_1d_teorico} porte")
print(f"Dettaglio porte del circuito QFT: {dict(qft_1d_decomp.count_ops())}")
print("=" * 60)

#Verifico l'andamento teorico vs empirico al crescere di N = 2^n.
#Definisco una serie di prove da n=2 a n=10.
n_range = np.arange(2, 11)
N_range = 2**n_range

#Calcolo complessità teorica della FFT classica.
curva_fft_1d = N_range * n_range

#Costruisco davvero il circuito QFT per ciascun valore di n e conto quante porte contiene realmente.
curva_qft_1d_emp = [QFT(num_qubits=int(n), do_swaps=True).decompose().size() for n in n_range]

#Applico la formula matematica del numero di porte della QFT (Costo QFT Teorico).
curva_qft_1d_teo = n_range + (n_range * (n_range - 1)) // 2 + (n_range // 2)

#Verifico che per ogni n, il numero di porte generate dal mio codice sia identico a quello previsto dalla teoria.
assert list(curva_qft_1d_emp) == list(curva_qft_1d_teo)

#GENERAZIONE GRAFICO COSTO COMPUTAZIONALE
fig_c1, ax_c1 = plt.subplots(figsize=(8, 4.8))
ax_c1.plot(N_range, curva_fft_1d, "o--", label=r"FFT 1D (np.fft.fft) — $\mathcal{O}(N \log_2 N)$", color="#1f77b4", linewidth=2)
ax_c1.plot(N_range, curva_qft_1d_emp, "s-", label=r"QFT 1D (Empirico = Teorico) — $\mathcal{O}(n^2)$", color="#ff7f0e", linewidth=2)
ax_c1.set_xscale("log", base=2)
ax_c1.set_yscale("log", base=10)
ax_c1.set_xlabel("Numero di campioni N = 2^n")
ax_c1.set_ylabel("Operazioni / Porte quantistiche (scala log)")
ax_c1.set_title("Verifica Costo Computazionale 1D: FFT Classica vs QFT")
ax_c1.grid(True, which="both", linestyle="--", alpha=0.6)
ax_c1.legend()
plt.tight_layout()
plt.savefig("costo_computazionale_1d.png", dpi=300)
plt.close(fig_c1)


frequenze = [f"k={i}" for i in range(len(moduli_classici))]
x_axis = np.arange(len(frequenze))
larghezza_barre = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(
    x_axis - larghezza_barre / 2,
    moduli_classici,
    larghezza_barre,
    label="NumPy FFT (Classica)",
    color="#1f77b4",
)
rects2 = ax.bar(
    x_axis + larghezza_barre / 2,
    moduli_quantistici,
    larghezza_barre,
    label="Qiskit QFT (Quantistica)",
    color="#ff7f0e",
)

ax.set_ylabel("Ampiezza di Frequenza |X[k]|")
ax.set_title("Confronto Spettrale 1D: FFT Classica vs QFT Quantistica (Ideale)")
ax.set_xticks(x_axis)
ax.set_xticklabels(frequenze)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
nome_grafico = "spettro_1d_confronto.png"
plt.savefig(nome_grafico, dpi=300)
print(f"\nGrafico salvato con successo come: '{nome_grafico}'")
plt.show()