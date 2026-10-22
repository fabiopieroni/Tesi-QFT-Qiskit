#PARTE 2: Costruzione "Custom" del circuito QFT: creo da zero una funzione che monta manualmente
#le porte di Hadamard, le rotazioni di fase controllate e le porte SWAP.
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


#FUNZIONE CUSTOM
#Definisco una funzione che accetta come argomento il circuito qc e vi aggiunge manualmente le porte elementari della QFT per 2 qubit.
def applica_qft_custom_2qubit(qc):
    #Applico la porta di Hadamard(H) sul qubit 1:Porta il qubit più significativo in una sovrapposizione
    #equiprobabile dei suoi stati di base, creando il primo sfasamento di pigreco.
    qc.h(1)
    #Porta di Fase Controllata(CP):Applica uno sfasamento di pigreco/2 (equivalente a i). Il qubit 0 fa da controllo e
    #il qubit 1 da target:la fase viene aggiunta allo stato del sistema solo se entrambi i qubit sono nello stato |1>.
    #Questo implementa la rotazione incrociata nella formula teorica di Fourier.
    qc.cp(np.pi / 2, 0, 1)
    #Applicazione della porta di Hadamard(H) sul qubit 0:Crea la sovrapposizione sul qubit meno significativo,
    #completando le combinazioni di sfasamento necessarie per la trasformata a 2 qubit.
    qc.h(0)
    #Porta SWAP:Scambia lo stato del qubit 0 con quello del qubit 1, necessario perchè l'algoritmo QFT genera le frequenze in ordine inverso.
    qc.swap(0, 1)

#PREPARAZIONE DEL SEGNALE E CALCOLO CLASSICO
#Definisco il segnale di ingresso vettoriale a 4 campioni espresso con numeri complessi.
x_classico = np.array([1.0, 0.0, 1.0, 0.0], dtype=complex)

#Calcolo la norma euclidea e divido il segnale per essa
norma = np.linalg.norm(x_classico)
x_norm = x_classico / norma

#Calcolo la DFT classica mediante NumPy con la normalizzazione ortogonale norm="ortho".
fft_classica = np.fft.fft(x_norm, norm="ortho")

#CREAZIONE ED ESECUZIONE DEL CIRCUITO
#Creo il registro quantistico a 2 qubit.
qc = QuantumCircuit(2)

#Mappo i valori del vettore Xnorm direttamente nelle ampiezze del registro quantistico.
qc.initialize(x_norm, range(2))

#Chiamo la funzione definita sul circuito caricato.
applica_qft_custom_2qubit(qc)

print("=" * 60)
print("PARTE 2: CIRCUITO QFT CUSTOM SCOMPOSTO IN PORTE LOGICHE")
print("=" * 60)

#Simulo algebricamente il circuito ed estraggo il vettore complesso delle ampiezze finali trasformate.
stato_finale = Statevector.from_instruction(qc)
spettro_quantistico_custom = stato_finale.data

#Calcolo il modulo dei vettori complessi per ottenere lo spettro di ampiezza visibile.
moduli_classici = np.abs(fft_classica)
moduli_custom = np.abs(spettro_quantistico_custom)

print(f"Moduli Spettro Classico (NumPy): {np.round(moduli_classici, 4)}")
print(f"Moduli QFT Custom (Scomposta):  {np.round(moduli_custom, 4)}")
print("-" * 60)

#Confronto i due vettori risultanti se coincidono.
esito = np.allclose(moduli_classici, moduli_custom)
if esito:
    print("ESITO: CIRCUITO CUSTOM CORRETTO! ✅ (Coincide con NumPy FFT)")
else:
    print("ESITO: CIRCUITO CUSTOM ERRATO! ❌")
print("=" * 60)

#GENERAZIONE E SALVATAGGIO SCHEMA GRAFICO
fig = qc.draw(output="mpl", style="bw")
plt.title("Circuito QFT Custom a 2 Qubit")
plt.savefig("circuito_qft_custom.png", dpi=300, bbox_inches="tight")
print("\nSchema del circuito salvato con successo come: 'circuito_qft_custom.png'")
plt.show()