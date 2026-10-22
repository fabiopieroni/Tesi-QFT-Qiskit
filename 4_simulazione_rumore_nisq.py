#PARTE 4:Simulazione di Rumore NISQ e calcolo dell'SNR: introduco il realismo fisico
#usando qiskit-aer, applico il modelli di rumore(decomposizione di fase,decoerenza,rumore di depolarizzazione)
#e misuro il degrado dello spettro calcolando il rapporto segnale-rumore.
#OBIETTIVO: Valutare l'impatto della decoerenza e del rumore delle porte quantistiche
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

#PREPARAZIONE SEGNALE
#Creo il circuito a 2 qubit e carico il segnale [1,0,1,0], normalizzato mediante Amplitude Encoding.
x_classico = np.array([1.0, 0.0, 1.0, 0.0], dtype=complex)
norma = np.linalg.norm(x_classico)
x_norm = x_classico / norma

num_qubits = 2
qc_ideale = QuantumCircuit(num_qubits)
qc_ideale.initialize(x_norm, range(num_qubits))

#Applico la QFT e aggiungo l'istruzione .measure_all(). 
#Nei dispositivi NISQ reali non è possibile estrarre direttamente il vettore, 
#ma occorre eseguire misurazioni ripetute per campionare la distribuzione di probabilità degli stati di base.
qft_gate = QFTGate(num_qubits=num_qubits)
qc_ideale.append(qft_gate, range(num_qubits))
qc_ideale.measure_all()

#CONFIGURAZIONE RUMORE
#Inizializzo un contenitore vuoto per il modello di rumore.
noise_model = NoiseModel()

#Definisco un canale di depolarizzazione a 1 qubit con probabilità di errore=1%, rappresentativo del rumore sulle porte di Hadamard.
p_error_1q = 0.01
error_1q = depolarizing_error(p_error_1q, 1)

#Definisco un errore di depolarizzazione a 2 qubit con probabilità di errore=5%, che modella le imperfezioni 
#e gli sfasamenti indesiderati durante l'esecuzione delle porte controllate e SWAP.
p_error_2q = 0.05
error_2q = depolarizing_error(p_error_2q, 2)

#Modella l'errore dei detector di misura con una matrice di sottomisura stochasticamente
#confusa: 3% di probabilità che uno stato |0> misurato venga letto come |1> e viceversa.
p_readout = 0.03
readout_err = ReadoutError(
    [[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]
)

#Associo ciascuna tipologia di errore quantistico alle rispettive porte logiche presenti nel circuito.
noise_model.add_all_qubit_quantum_error(error_1q, ["h", "u1", "u2", "u3", "rz", "sx"])
noise_model.add_all_qubit_quantum_error(error_2q, ["cp", "swap", "cx", "cz"])
noise_model.add_all_qubit_readout_error(readout_err)

#CAMPIONAMENTI E CALCOLO AMPIEZZE
#Eseguo 10000 simulazioni(shots) del circuito sul simulatore ideale senza rumore per ottenere l'istogramma delle frequenze teoriche.
shots = 10000
sim_ideale = AerSimulator()
qc_transpiled_ideale = transpile(qc_ideale, sim_ideale)
result_ideale = sim_ideale.run(qc_transpiled_ideale, shots=shots).result()
counts_ideale = result_ideale.get_counts()

#Eseguo 10000 campionamenti iniettando il modello di rumore NISQ ad ogni porta e misura del circuito.
sim_rumoroso = AerSimulator(noise_model=noise_model)
qc_transpiled_rumoroso = transpile(qc_ideale, sim_rumoroso)
result_rumoroso = sim_rumoroso.run(qc_transpiled_rumoroso, shots=shots).result()
counts_rumoroso = result_rumoroso.get_counts()

stati_base = ["00", "01", "10", "11"]

#Converte i conteggi assoluti delle letture nei rispettivi moduli di ampiezza.
prob_ideali = [counts_ideale.get(st, 0) / shots for st in stati_base]
prob_rumorose = [counts_rumoroso.get(st, 0) / shots for st in stati_base]
moduli_ideali = np.sqrt(prob_ideali)
moduli_rumorosi = np.sqrt(prob_rumorose)

print("=" * 60)
print("PARTE 4: SIMULAZIONE RUMORE NISQ SU QFT A 2 QUBIT")
print("=" * 60)
print(f"Moduli Spettro Ideale (Senza Rumore):  {np.round(moduli_ideali, 4)}")
print(f"Moduli Spettro NISQ (Con Rumore):      {np.round(moduli_rumorosi, 4)}")
print("-" * 60)

#CALCOLO DELLA FEDELTA
#Calcolo il coefficiente che misura la sovrapposizione tra la distribuzione ideale
#e quella affetta da rumore NISQ(coefficiente di Bhattacharyya).
fedelta = np.sum(np.sqrt(np.array(prob_ideali) * np.array(prob_rumorose)))
print(f"Fedelta della Distribuzione Spettrale (Fidelity): {fedelta * 100:.2f}%")
print("=" * 60)

frequenze_labels = ["k=0 (|00⟩)", "k=1 (|01⟩)", "k=2 (|10⟩)", "k=3 (|11⟩)"]
x_axis = np.arange(len(frequenze_labels))
larghezza_barre = 0.35

#Genero il grafico a barre che mostra come il rumore provochi il decadimento dei picchi ideali 
#e la comparsa del "rumore di fondo" negli stati che teoricamente dovrebbero essere nulli.
fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(
    x_axis - larghezza_barre / 2,
    moduli_ideali,
    larghezza_barre,
    label="QFT Ideale (Senza Rumore)",
    color="#1f77b4",
)
rects2 = ax.bar(
    x_axis + larghezza_barre / 2,
    moduli_rumorosi,
    larghezza_barre,
    label=f"QFT NISQ con Rumore (Fedeltà = {fedelta*100:.1f}%)",
    color="#d62728",
)

ax.set_ylabel("Ampiezza di Frequenza Modulo |X[k]|")
ax.set_title("Impatto del Rumore NISQ sulla QFT (Simulazione Hardware Reale)")
ax.set_xticks(x_axis)
ax.set_xticklabels(frequenze_labels)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
nome_grafico_nisq = "spettro_nisq_rumore.png"
plt.savefig(nome_grafico_nisq, dpi=300)
print(f"\nGrafico NISQ salvato con successo come: '{nome_grafico_nisq}'")
plt.show()