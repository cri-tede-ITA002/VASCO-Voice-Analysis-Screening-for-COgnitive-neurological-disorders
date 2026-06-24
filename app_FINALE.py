#!/usr/bin/env python3
"""
app.py
======
Interfaccia Gradio per l'agente CRNN di screening vocale del Parkinson.

Avvio:
    python app.py
    python app.py --share

Dipendenze aggiuntive:
    pip install gradio

URL locale:
    http://127.0.0.1:7860/
"""
from __future__ import annotations

import gradio as gr

# Importa l'agente CRNN corretto
from agente_FINALE import run_agent


# ─────────────────────────────────────────────────────────────────────────────
# LOGICA DI INTERFACCIA
# ─────────────────────────────────────────────────────────────────────────────

def analyze(audio_path: str | None, patient_name: str) -> tuple:
    """
    Callback Gradio.

    Riceve il path temporaneo del file audio caricato dall'utente,
    lancia l'agente CRNN e restituisce i valori per tutti i componenti UI.

    Returns:
        status_md, prob_slider, report_it, report_en, spec_image, dl_file
    """
    if audio_path is None:
        return (
            "⚠️ **Nessun file caricato.** Carica un file audio prima di procedere.",
            gr.update(visible=False),
            "",
            "",
            None,
            None,
        )

    try:
        state = run_agent(audio_path, patient_name=patient_name)
    except Exception as exc:
        return (
            f"❌ **Errore interno:** `{exc}`",
            gr.update(visible=False),
            "",
            "",
            None,
            None,
        )

    if state.get("error"):
        status_md = (
            "### ⚠️ Audio non idoneo\n\n"
            f"{state['error']}\n\n"
            "_Carica un nuovo file e riprova._"
        )
        return (
            status_md,
            gr.update(visible=False),
            "",
            "",
            None,
            None,
        )

    prob = float(state["prob_score"])
    is_pd = bool(state["is_parkinson"])
    pct = f"{prob:.1%}"

    if is_pd:
        verdict_emoji = "🟠"
        verdict_it = "POSITIVO — Segnali vocali compatibili con possibile rischio Parkinson"
        verdict_en = "POSITIVE — Vocal patterns compatible with possible Parkinson's risk"
    else:
        verdict_emoji = "🟢"
        verdict_it = "NEGATIVO — Nessun segnale vocale rilevante nel campione analizzato"
        verdict_en = "NEGATIVE — No relevant vocal anomaly detected in the analyzed sample"

    status_md = (
        f"## {verdict_emoji} {verdict_it}\n"
        f"*{verdict_en}*\n\n"
        f"**Indice vocale PD:** {pct} &nbsp;|&nbsp; "
        f"**Soglia decisionale:** {state['threshold']:.1%} &nbsp;|&nbsp; "
        f"**Durata audio:** {state['duration_s']:.1f}s"
    )

    return (
        status_md,
        gr.update(value=prob, visible=True, label=f"Indice vocale PD: {pct}"),
        state["report_it"],
        state["report_en"],
        state.get("spectrogram_path"),
        state.get("report_path"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT GRADIO
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
#main-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: .2rem;
    color: var(--body-text-color);
}

#subtitle {
    text-align: center;
    color: var(--body-text-color-subdued);
    font-size: .95rem;
    margin-bottom: 1.4rem;
}

#disclaimer {
    border-left: 3px solid #E24B4A;
    padding: .6rem .9rem;
    background: var(--block-background-fill);
    border-radius: 6px;
    font-size: .85rem;
    color: var(--body-text-color-subdued);
    margin-top: .6rem;
}

#audio-upload .wrap {
    min-height: 120px;
}

#run-btn {
    font-size: 1.05rem;
    font-weight: 600;
}
"""

DISCLAIMER = (
    "⚠️ **Avvertenza / Warning** — "
    "Questo strumento è un prototipo di ricerca per screening vocale preliminare. "
    "Un risultato positivo non costituisce diagnosi e un risultato negativo non esclude clinicamente la malattia. "
    "In presenza di sintomi, familiarità o dubbi, consultare sempre il medico di base o uno specialista neurologo. / "
    "This tool is a research prototype for preliminary vocal screening. "
    "A positive result is not a diagnosis, and a negative result does not clinically rule out the disease. "
    "In case of symptoms, family history, or concerns, always consult a general practitioner or neurologist."
)

with gr.Blocks(css=CSS, title="V.A.S.C.O.") as demo:
    gr.HTML('<div id="main-title">V.A.S.C.O.</div>')
    gr.HTML('<div id="subtitle">Voice Analysis Screening for COgnitive/neurological disorders</div>')

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Carica registrazione vocale (WAV, MP3, M4A, FLAC, OGG, AAC, …)",
                type="filepath",
                sources=["upload", "microphone"],
                elem_id="audio-upload",
            )

            patient_input = gr.Textbox(
                label="Nome e Cognome del paziente (opzionale)",
                placeholder="es. Mario Rossi",
                max_lines=1,
            )

            run_btn = gr.Button(
                "▶  Avvia analisi",
                variant="primary",
                elem_id="run-btn",
            )

            prob_slider = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.001,
                label="Indice vocale PD",
                interactive=False,
                visible=False,
            )

            status_out = gr.Markdown("_Carica un file audio e premi **Avvia analisi**._")
            gr.HTML(f'<div id="disclaimer">{DISCLAIMER}</div>')

        with gr.Column(scale=2):
            spec_out = gr.Image(
                label="Spettrogramma Mel",
                type="filepath",
                show_download_button=True,
            )

            with gr.Tabs():
                with gr.Tab("📄 Referto — Italiano"):
                    report_it_out = gr.Textbox(
                        label="",
                        lines=22,
                        max_lines=30,
                        show_copy_button=True,
                    )

                with gr.Tab("📄 Report — English"):
                    report_en_out = gr.Textbox(
                        label="",
                        lines=22,
                        max_lines=30,
                        show_copy_button=True,
                    )

            dl_btn = gr.File(label="⬇  Scarica referto completo (.txt)")

    with gr.Accordion("ℹ️ Istruzioni / Instructions", open=False):
        gr.Markdown(
            """
            **Come registrare il campione vocale corretto:**
            1. Trova un ambiente silenzioso.
            2. Tieni il microfono a circa 10–15 cm dalla bocca.
            3. Pronuncia la vocale **"aaaa"** in modo sostenuto e uniforme per **3–5 secondi**.
            4. Evita rumori di fondo, colpi di tosse o interruzioni.
            5. Salva il file in uno dei formati supportati: **.wav, .mp3, .m4a, .flac, .ogg, .aac, .opus, .aiff, .wma, .webm** e caricalo qui.
               La conversione in WAV mono a 16 kHz viene gestita automaticamente quando possibile.

            ---

            **How to record the correct vocal sample:**
            1. Find a quiet environment.
            2. Hold the microphone 10–15 cm from your mouth.
            3. Sustain the vowel **"aaaa"** evenly for **3–5 seconds**.
            4. Avoid background noise, coughs, or interruptions.
            5. Save the file in one of the supported formats: **.wav, .mp3, .m4a, .flac, .ogg, .aac, .opus, .aiff, .wma, .webm** and upload it here.
               Conversion to WAV mono 16 kHz is handled automatically whenever possible.
            """
        )

    run_btn.click(
        fn=analyze,
        inputs=[audio_input, patient_input],
        outputs=[status_out, prob_slider, report_it_out, report_en_out, spec_out, dl_btn],
        show_progress="full",
    )


# ─────────────────────────────────────────────────────────────────────────────
# AVVIO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share",
        action="store_true",
        help="Genera link pubblico temporaneo (Gradio Share).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Porta locale su cui avviare Gradio.",
    )

    args = parser.parse_args()

    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
