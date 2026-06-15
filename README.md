# VASCO — Voice Analysis Screening for COgnitive/neurological disorders

> Classificazione automatica del Morbo di Parkinson da registrazioni vocali tramite tre architetture di deep learning (LSTM, CRNN, AST), con integrazione in un agente di screening end-to-end.

---

## Panoramica

Questo progetto esplora l'uso di biomarcatori acustici per la classificazione precoce del Morbo di Parkinson (PD). Le registrazioni vocali di soggetti sani (HC) e pazienti con Parkinson (PD) — consistenti nella vocalizzazione sostenuta della vocale */a/* — vengono elaborate attraverso tre pipeline di deep learning di complessità crescente, confrontate con un protocollo sperimentale rigoroso e integrate in un agente di screening automatico.

L'ipotesi centrale è che il Parkinson introduca anomalie acustiche misurabili — tremore vocale, disfonia, instabilità microstrutturale del segnale (jitter, shimmer) — catturabili da brevi registrazioni vocali prima che compaiano i sintomi motori evidenti.

---

## Struttura del Progetto

```
VASCO/
│
├── data/                          # Dataset vocali
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
├── artifacts/                     # Artefatti di training (stats, metriche)
├── checkpoints/                   # Checkpoint salvati (LSTM, CRNN)
├── plots/                         # Plot generati durante il training
│
├── LSTM_FINALE.py                 # Classificatore LSTM (baseline)
├── CRNN_FINALE.py                 # Classificatore CRNN
├── AST.FINALE.py                  # Audio Spectrogram Transformer (linear probing)
├── agente.FINALE.py               # Agente VASCO (LangGraph + CRNN + MedGemma)
├── app_FINALE.py                  # Interfaccia web Gradio
├── augmented_def.py               # Data augmentation offline
├── pre_process.ipynb              # Preprocessing esplorativo e EDA
└── README.md
```

---

## Dataset

Il dataset complessivo è ottenuto dall'unione di due raccolte indipendenti:

| Sorgente | HC | PD | Totale |
|----------|----|----|--------|
| Prior et al. | 41 | 40 | 81 |
| Dimauro & Girardi | 44 | 55 | 99 |
| **Totale** | **85** | **95** | **180** |

Entrambe le sorgenti riguardano la vocalizzazione sostenuta della vocale */a/*. La durata media delle registrazioni è 7.3 s (mediana 5.8 s, std 5.6 s).

### Suddivisione train/test

Lo split è effettuato **a livello di soggetto** e **prima** dell'augmentation, per evitare data leakage:

| Sottoinsieme | Registrazioni | HC | PD |
|---|---|---|---|
| Train set | 126 (70%) | 63 | 63 |
| Test set | 54 (30%) | 22 | 32 |

---

## Pipeline

```
File audio grezzo (WAV, MP3, M4A, FLAC, OGG, ...)
  │
  ├─ 1. Data Augmentation (augmented_def.py)
  │       velocità ±5% · pitch ±2 semitoni · rumore rosa @ 20 dB SNR
  │       → 5 versioni aumentate per soggetto (training set ×6)
  │
  ├─ 2a. LSTM (baseline)
  │       Log-Mel spettrogramma (64 bande, 16 kHz)
  │       → finestre 2 s, overlap 50%
  │       → LSTM 2 layer, hidden 128 → BCEWithLogitsLoss
  │       → inferenza speaker-level (media finestre)
  │
  ├─ 2b. CRNN
  │       Log-Mel + Δ + ΔΔ (tensore 3×64×1024)
  │       → 3× Conv2D+BN+ReLU+MaxPool
  │       → LSTM bidirezionale, hidden 128
  │       → Max+Avg pooling → BCEWithLogitsLoss + label smoothing ε=0.1
  │       → inferenza speaker-level (intera registrazione per volta)
  │
  └─ 2c. AST — Audio Spectrogram Transformer (linear probing)
          ASTFeatureExtractor (HuggingFace) → patch 16×16
          → backbone MIT/ast-finetuned-audioset congelato
          → solo classification head addestrabile (<0.002% parametri)
          → inferenza speaker-level (media softmax su finestre da 5 s)
```

---

## Data Augmentation

Ogni file WAV del training set genera 5 versioni aumentate:

| Tecnica | Parametro | Motivazione |
|---------|-----------|-------------|
| Speed down | −5% (`time_stretch`) | Simula la bradifrenia |
| Speed up | +5% (`time_stretch`) | Simula la festinazione |
| Pitch down | −2 semitoni (`pitch_shift`) | Variabilità inter-sessione |
| Pitch up | +2 semitoni (`pitch_shift`) | Variabilità inter-sessione |
| Rumore rosa | SNR = 20 dB (metodo Kellet, 1/f) | Robustezza ambientale |

Le sei versioni (originale + 5 aumentate) di ciascun soggetto sono trattate come un gruppo inscindibile in tutti gli split successivi (train/validation), garantendo l'assenza di data leakage a livello di soggetto.

---

## Protocollo di Valutazione

Tutti i modelli seguono lo stesso protocollo sperimentale per garantire un confronto equo:

1. **Split train/validation augmentation-aware** — 20% dei soggetti in validation, stratificato per classe, con vincolo che tutte le versioni di uno stesso soggetto siano nello stesso sottoinsieme.
2. **Selezione del checkpoint** — basata su ROC-AUC speaker-level sul validation set (metrica indipendente dalla soglia).
3. **Calibrazione della soglia decisionale** — τ★ fissata sul validation set massimizzando l'F1-score.
4. **Valutazione finale** — sul test set, una sola volta, con bootstrap non parametrico (B=1000) per stimare gli intervalli di confidenza al 95%.

---

## Modelli

### LSTM (baseline)

| Parametro | Valore |
|-----------|--------|
| Input | Log-Mel (200 frame × 64 bande) |
| Architettura | LSTM 2 layer, hidden 128, unidirezionale |
| Classificatore | LayerNorm → Linear(128→1) |
| Loss | BCEWithLogitsLoss |
| Ottimizzatore | Adam lr=1e-3, weight_decay=1e-5 |
| Epoche | 15 |
| Inferenza | Speaker-level (media sigmoidi su finestre da 2 s) |

### CRNN

| Parametro | Valore |
|-----------|--------|
| Input | Log-Mel + Δ + ΔΔ (3×64×1024) |
| Front-end CNN | 3× Conv2D(3→16→32→64) + BN + ReLU + MaxPool(2×2) |
| LSTM | Bidirezionale, 1 layer, hidden 128/dir → uscita 256 |
| Aggregazione | Global max-pooling + avg-pooling → cat → (B, 512) |
| Classificatore | Dropout(0.5) → LayerNorm → Linear(512→1) |
| Loss | BCEWithLogitsLoss + label smoothing ε=0.1 |
| Ottimizzatore | Adam lr=1e-5 + ReduceLROnPlateau(patience=3, factor=0.5) |
| Epoche | 20 |
| Inferenza | Speaker-level (intera registrazione, no windowing) |

### AST — Audio Spectrogram Transformer

| Parametro | Valore |
|-----------|--------|
| Backbone | `MIT/ast-finetuned-audioset-10-10-0.4593` (congelato) |
| Strategia | Linear probing (solo classification head addestrabile) |
| Parametri addestrabili | < 0.002% del totale (86 M) |
| Pre-processing | ASTFeatureExtractor (HuggingFace), 128 bande Mel |
| Finestratura | Finestre 5 s, hop 2.5 s (50% overlap) |
| Loss | Cross-entropy 2 classi |
| Ottimizzatore | AdamW lr=1e-4, weight_decay=1e-4 |
| Epoche | 15 |
| Inferenza | Speaker-level (media softmax su finestre) |

---

## Risultati

Valutazione sul test set (54 soggetti: 22 HC, 32 PD). Intervalli di confidenza al 95% stimati via bootstrap (B=1000).

| Metrica | LSTM | CRNN | AST |
|---------|------|------|-----|
| Soglia applicata | 0.535 | 0.416 | 0.749 |
| Accuracy | 0.63 [0.50, 0.76] | **0.72** [0.61, 0.83] | 0.69 [0.57, 0.80] |
| Precision | 0.83 [0.64, 1.00] | 0.73 [0.58, 0.87] | **0.89** [0.75, 1.00] |
| Recall (Sens.) | 0.47 [0.29, 0.66] | **0.84** [0.71, 0.95] | 0.53 [0.35, 0.70] |
| Specificity | 0.86 [0.70, 1.00] | 0.55 [0.33, 0.76] | **0.91** [0.77, 1.00] |
| F1 | 0.60 [0.41, 0.75] | **0.78** [0.67, 0.88] | 0.67 [0.49, 0.80] |
| ROC-AUC | 0.72 [0.57, 0.85] | 0.79 [0.67, 0.90] | **0.85** [0.73, 0.94] |

### Osservazioni principali

**CRNN** ottiene il miglior profilo complessivo per uno scenario di screening: recall 0.84 (identifica 27/32 pazienti PD) con F1 0.78. La combinazione CNN + LSTM bidirezionale supera nettamente la sola LSTM e offre un bilanciamento sensibilità/specificità superiore all'AST al punto operativo ottimale.

**AST** presenta la migliore capacità discriminativa in assoluto (ROC-AUC 0.85), confermando l'efficacia del transfer learning da AudioSet anche in regime di solo linear probing. Tuttavia, la soglia ottima sul validation set risulta conservativa (τ=0.749): abbassarla per aumentare il recall richiederebbe τ=0.136 per recall ≥ 0.80, con specificità di validazione pari ad appena 0.23. Il limite non riguarda quindi la sola soglia, ma la distribuzione delle probabilità assegnate ai soggetti PD più difficili.

**LSTM** (baseline) mostra una buona specificità (0.86) e precision (0.83), ma il recall di 0.47 ne limita l'utilità come strumento di screening.

In ottica clinica, il costo di un falso negativo (paziente non individuato) è superiore a quello di un falso positivo: la **CRNN** è il modello scelto per l'agente VASCO.

---

## Agente VASCO

A valle della fase sperimentale, la CRNN è integrata in **VASCO** (*Voice Analysis Screening for COgnitive/neurological disorders*), un agente di screening automatico implementato con **LangGraph**.

### Architettura del grafo

```
START
  │
  ▼
[convert] ──(formato non valido)──┐
  │ formato ok                    │
  ▼                               │
[preprocess] ──(durata < 0.5 s)──┤
  │ audio ok                      │
  ▼                               │
[inference]                       │
  │                               ▼
  ▼                            [reject] ──► END
[report] ──────────────────────────────────► END
```

### Fasi operative

1. **convert** — Accetta WAV, MP3, M4A, FLAC, OGG e altri formati; converte in WAV mono 16 kHz via `ffmpeg` (o fallback `librosa`+`soundfile`).
2. **preprocess** — Normalize → trim (top_db=35) → log-Mel + Δ + ΔΔ → tensore (3, 64, 1024). Rifiuta audio con durata utile < 0.5 s.
3. **inference** — CRNN addestrata → sigmoid → score PD ∈ [0,1] → confronto con soglia decisionale.
4. **report** — Template deterministico (score, soglia, livello di rischio, esito) + **MedGemma** (via Ollama, solo in caso di esito positivo) per la sezione di suggerimenti clinici in linguaggio naturale.
5. **reject** — Termina il flusso con messaggio di errore se l'audio non è idoneo.

### Soglia operativa

Nell'agente la soglia è impostata a **0.5** (più conservativa della soglia F1-ottima 0.416 usata in valutazione), per ridurre i falsi positivi in produzione. L'output non costituisce diagnosi medica.

### Referto

Il referto è bilingue (italiano/inglese) e include: dati del campione, durata utile, spettrogramma Mel, score vocale, soglia, livello di rischio e esito. In caso di esito positivo, MedGemma — LLM a orientamento biomedico di Google DeepMind — genera una sezione di suggerimenti clinici rassicuranti, privi di terminologia tecnica, con invito a rivolgersi al medico di base. La decisione classificativa resta esclusivamente deterministica (CRNN + soglia).

---

## Requisiti

```bash
pip install torch torchaudio librosa soundfile numpy scikit-learn \
            matplotlib langgraph transformers requests
```

Per la generazione del referto con MedGemma è necessario [Ollama](https://ollama.com) con il modello `medgemma`:

```bash
ollama pull medgemma
```

---

## Utilizzo

### 1. Data Augmentation

```bash
python augmented_def.py \
  --train-dir data/Training \
  --out-dir data/Training_augmented
```

### 2. Training modelli

```bash
# LSTM (baseline)
python LSTM_FINALE.py --epochs 15

# CRNN
python CRNN_FINALE.py --epochs 20

# AST (linear probing)
python AST.FINALE.py --epochs 15
```

### 3. Agente VASCO

```bash
python agente.FINALE.py percorso/audio.wav
python agente.FINALE.py percorso/audio.wav --patient-name "Mario Rossi"
```

Il referto viene salvato in `output_reports/` e stampato a terminale.

### 4. Interfaccia web

```bash
python app_FINALE.py
```

---

## Contesto Clinico

Il Morbo di Parkinson colpisce il sistema motorio vocale producendo anomalie acustiche misurabili prima che i sintomi motori diventino clinicamente evidenti:

- **Disfonia** — vibrazione irregolare delle corde vocali (jitter, shimmer elevati)
- **Bradifrenia** — rallentamento del ritmo del parlato
- **Ipofonia** — riduzione dell'intensità vocale
- **Monotonia del tono** — ridotta variabilità della frequenza fondamentale (F0)
- **Riduzione HNR** — maggiore componente di rumore nel segnale vocale

VASCO è concepito come ausilio allo screening preliminare e **non sostituisce in alcun modo una valutazione neurologica clinica**. Un esito positivo indica la presenza di pattern vocali compatibili con un possibile rischio da approfondire; un esito negativo non esclude clinicamente la presenza della malattia.

---

## Autori

- **Gaia Farace** — Università della Calabria, Data Science per le Strategie Aziendali
- **Cristian Tedesco** — Università della Calabria, Data Science per le Strategie Aziendali
