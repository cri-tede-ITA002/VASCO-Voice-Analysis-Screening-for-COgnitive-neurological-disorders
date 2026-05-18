# 🎙️ ParkVoice — Rilevazione Precoce del Parkinson tramite Analisi del Parlato

> Classificazione automatica del Morbo di Parkinson da registrazioni vocali tramite tre approcci di deep learning: LSTM, CRNN e VGGish + Regressione Logistica.

---

## 📋 Indice

- [Panoramica](#panoramica)
- [Struttura del Progetto](#struttura-del-progetto)
- [Pipeline](#pipeline)
- [Modelli](#modelli)
- [Risultati](#risultati)
- [Requisiti](#requisiti)
- [Utilizzo](#utilizzo)
- [Contesto Clinico](#contesto-clinico)

---

## Panoramica

Questo progetto esplora l'uso di biomarcatori acustici per la classificazione precoce del Morbo di Parkinson (PD). Le registrazioni vocali di soggetti sani (HC) e pazienti con Parkinson (PD) vengono elaborate attraverso tre pipeline indipendenti, ciascuna con una strategia di modellazione diversa.

L'ipotesi centrale è che il Parkinson introduca anomalie acustiche misurabili — tremore vocale, disfonia, ritmo irregolare — catturabili da brevi registrazioni del parlato.

---

## Struttura del Progetto

```
ParkVoice/
│
├── data/
│   ├── data_original/
│   │   ├── HC/
│   │   └── PD/
│   ├── Training/
│   │   ├── HC/
│   │   └── PD/
│   ├── Training_augmented/
│   │   ├── HC/
│   │   └── PD/
│   └── Test/
│       ├── HC/
│       └── PD/
│
├── artifacts/
│   ├── checkpoints/
│   ├── plots/
│   └── stats/
│
├── augmented_def.py        # Augmentazione dati offline
├── LSTM.py                 # Classificatore LSTM
├── CRNN.py                 # Classificatore CRNN
└── m2_vggishg_lr_128.py    # VGGish + Regressione Logistica
```

---

## Pipeline

```
File WAV grezzo
  │
  ├─ 1. Data Augmentation (augmented_def.py)
  │       velocità ±5% | pitch ±2 semitoni | rumore rosa @ 20dB SNR
  │       → 5× dati di training
  │
  ├─ 2a. LSTM
  │       Mel-spettrogramma logaritmico (64 bande, 16kHz)
  │       → finestre da 2 sec con overlap 50%
  │       → LSTM 2 layer → BCEWithLogitsLoss
  │
  ├─ 2b. CRNN
  │       Mel + delta + delta2 (3 canali)
  │       → 3× Conv2D → LSTM bidirezionale
  │       → Max+Avg pooling → BCEWithLogitsLoss + label smoothing
  │
  └─ 2c. VGGish + Regressione Logistica
          VGGish pre-addestrato (frozen) → embedding 512D
          → Regressione Logistica (5-Fold Stratified CV)
```

### Data Augmentation

Ogni file WAV del training genera 5 versioni aumentate:

| Tecnica | Parametro | Motivazione |
|---------|-----------|-------------|
| Speed down | −5% | Simula la bradifrenia |
| Speed up | +5% | Simula la festinazione |
| Rumore rosa | SNR = 20 dB | Robustezza all'ambiente di registrazione |
| Pitch down | −2 semitoni | Variabilità inter-sessione |
| Pitch up | +2 semitoni | Variabilità inter-sessione |

---

## Modelli

### 1. LSTM

| Parametro | Valore |
|-----------|--------|
| Input | Mel-spettrogramma log (200 frame × 64 bande mel) |
| Architettura | LSTM 2 layer, hidden size 128 |
| Classificatore | LayerNorm → Linear(128→1) |
| Loss | BCEWithLogitsLoss |
| Ottimizzatore | Adam lr=1e-3, weight_decay=1e-5 |
| Epoche | 30 |
| Valutazione | Speaker-level (aggregazione finestre per soggetto) |

### 2. CRNN

| Parametro | Valore |
|-----------|--------|
| Input | Mel + Δ + ΔΔ (3 × 64 × 1024) |
| Architettura | 3× Conv2D + BatchNorm → LSTM bidirezionale |
| Pooling | Max + Avg temporale → cat → (B, 512) |
| Classificatore | Dropout(0.5) → LayerNorm → Linear(512→1) |
| Loss | BCEWithLogitsLoss + label smoothing (ε=0.1) |
| Ottimizzatore | Adam lr=1e-5 + ReduceLROnPlateau |
| Epoche | 20 |
| Checkpoint | Salvato ad ogni miglioramento del val AUC |

### 3. VGGish + Regressione Logistica

| Parametro | Valore |
|-----------|--------|
| Estrattore feature | VGGish (pre-addestrato, frozen) |
| Dimensione embedding | 512D (4 frame × 128D) |
| Classificatore | Regressione Logistica (class_weight=balanced) |
| Valutazione | 5-Fold Stratified CV + predizioni OOF |
| Audio in input | Max 4 secondi @ 16kHz |

---

## Risultati

> **Note sulla valutazione:**
> - Le metriche di LSTM e CRNN sono calcolate a livello **speaker** (probabilità delle finestre mediate per soggetto).
> - Le metriche VGGish sono **Out-Of-Fold (OOF)** aggregate sui 5 fold — la stima statisticamente più robusta dato il numero limitato di soggetti.
> - In diagnostica clinica, il **Recall** (sensibilità) è la metrica prioritaria: un falso negativo (paziente PD non riconosciuto) è più pericoloso di un falso positivo.

### Tabella di Confronto

| Modello | AUC | Recall | Specificità | Precisione | F1 | Accuratezza |
|---------|-----|--------|-------------|------------|----|-------------|
| **CRNN** | **0.885** | 0.813 | **0.864** | **0.897** | **0.852** | **0.833** |
| VGGish + LR | 0.796 | 0.757 | 0.659 | 0.713 | 0.734 | 0.711 |
| LSTM | 0.751 | **1.000** | 0.000 | 0.593 | 0.744 | 0.593 |

### Matrici di Confusione

| | **LSTM** (Speaker) | **CRNN** | **VGGish** (OOF) |
|---|---|---|---|
| TP | 32 | 26 | 72 |
| FP | 22 | 3 | 29 |
| TN | 0 | 19 | 56 |
| FN | 0 | 6 | 23 |

### Osservazioni Principali

**CRNN** ottiene le performance migliori in assoluto — AUC 0.885, miglior bilanciamento tra Recall e Specificità, e il tasso di falsi positivi più basso. La LSTM bidirezionale combinata con l'estrazione di feature tramite CNN offre un vantaggio netto rispetto alla LSTM semplice.

**LSTM** presenta Recall perfetto (1.0) ma Specificità zero — classifica tutti i soggetti come PD. Si tratta di una soluzione degenere: il modello ha imparato il bias della classe maggioritaria invece di feature discriminative. Il risultato è clinicamente inutile nonostante l'alta sensibilità.

**VGGish + LR** rappresenta una buona via di mezzo — l'AUC OOF di 0.796 è valutato sull'intero dataset senza un test set fisso, rendendolo la stima statisticamente più affidabile. Gli embedding pre-addestrati generalizzano bene anche senza fine-tuning.

---

## Requisiti

```bash
pip install torch torchaudio librosa soundfile numpy scikit-learn matplotlib torchvggish
```

---

## Utilizzo

### 1. Data Augmentation

```bash
python augmented_def.py \
  --train-dir data/Training \
  --out-dir data/Training_augmented
```

### 2. Addestramento LSTM

```bash
python LSTM.py --epochs 30
python LSTM.py --epochs 30 --recache   # ricalcola gli spettrogrammi
python LSTM.py --epochs 30 --plot      # salva anche i mel-spettrogrammi
```

### 3. Addestramento CRNN

```bash
python CRNN.py --epochs 20
python CRNN.py --epochs 20 --plot
```

### 4. VGGish + Regressione Logistica

```bash
python m2_vggishg_lr_128.py
python m2_vggishg_lr_128.py --n_splits 10   # k-fold personalizzato
```

---

## Contesto Clinico

Il Morbo di Parkinson colpisce il sistema motorio vocale, producendo anomalie acustiche misurabili:

- **Disfonia** — vibrazione irregolare delle corde vocali (jitter, shimmer)
- **Bradifrenia** — rallentamento del ritmo del parlato
- **Ipofonia** — riduzione dell'intensità vocale
- **Monotonia del tono** — ridotta variabilità della frequenza fondamentale (F0)

Queste caratteristiche acustiche sono presenti prima che i sintomi motori diventino clinicamente evidenti, rendendo l'analisi del parlato uno strumento promettente per lo **screening precoce e non invasivo**.