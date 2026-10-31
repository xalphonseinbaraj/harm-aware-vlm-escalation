# When to Ask for Help: Harm-Aware, Confidence-Gated Escalation to Vision-Language Models

Code, trained gates and per-image result logs for the paper

> A. I. Xavier, *When to Ask for Help: Confidence-Gated Escalation to Vision-Language Models for Efficient Polyp Segmentation*, Biomedical Signal Processing and Control (under revision, manuscript BSPC-D-26-16498).

## Idea

A lightweight segmentation backbone segments every colonoscopy image. A vision-language model (VLM) could review and correct the mask, but VLM calls are slow, and VLMs are **unreliable editors**: they often damage masks that were already correct. A learned gate therefore decides, per image, whether a mask is sent for VLM review, and a causal faithfulness check can reject an edit before it is accepted.

```
image -> (1) backbone: mask m + confidence c
      -> (2) four mask-level features
      -> (3) logistic-regression gate: escalate if P(poor mask) >= theta
      -> (4) VLM proposes expand / shrink / shift / no change -> edited mask m'
      -> (5) causal verification: keep m' if faithfulness F >= tau, else m
```

## Main findings

* Escalating every image to Llama-3.2-11B-Vision or the medical VLM MedGemma-4B lowered mean IoU on Kvasir-SEG from 0.745 to 0.716; the gate escalated 30% of images and avoided 82-84% of that loss with 70% fewer VLM calls.
* Without retraining, on CVC-ClinicDB, CVC-ColonDB and ETIS, the gate on a SegFormer backbone avoided 61-93% of the loss caused by MedGemma.
* No policy improved on the backbone alone: no VLM reliably improved difficult masks. This negative result is reported in full.

## Repository layout

| Path | Contents |
|---|---|
| `models/` | PolypUNet backbone with confidence head, confidence gate, VLM interface, causal verification |
| `data/` | Kvasir-SEG loader and transforms |
| `train_backbone.py`, `train_gate.py`, `run_pipeline.py`, `eval.py` | Original training and pipeline scripts |
| `ablations/` | never-refine, always-refine and no-verify configurations |
| `app.py` | Streamlit demonstration of the five stages on a single image |
| `revision/` | Scripts for every experiment in the revised paper (see below) |
| `gates/` | Trained gates (`gate.joblib`), validation reports and training logs for every backbone |
| `results/kvasir/`, `results/external/` | Per-image logs of every run and the summary tables (`tables.md`) |

## Setup

```bash
uv venv && uv pip install -r requirements.txt     # or: pip install -r requirements.txt
cp .env.example .env                             # add NVIDIA_NIM_API_KEY only for the Llama runs
```

Local VLMs run through [Ollama](https://ollama.com): `ollama pull qwen2.5vl:7b` and `ollama pull medgemma:4b`.

### Data (not included)

* Kvasir-SEG: <https://datasets.simula.no/kvasir-seg/> -> `data/Kvasir-SEG/images`, `data/Kvasir-SEG/masks`
* CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB: the PraNet `TestDataset` release -> `data/TestDataset/<name>/images`, `.../masks`

## Reproducing the paper

Run from the repository root (Windows batch files; each step resumes if interrupted):

| Step | Script | Paper |
|---|---|---|
| 1 | `revision\phase1_train_backbones.bat` | Backbones (3 seeds x 2 augmentations, lambda ablation), gates; Table 8 |
| 2 | `revision\phase2_eval_vlm.bat` | Escalation policies with Llama-3.2-11B and Qwen2.5-VL-7B; Tables 3 and 7 |
| 3 | `revision\phase3_baselines.bat` | Matched baselines and efficiency; Tables 5 and 9 |
| 4 | `revision\phase4_analyze.bat` | All summary tables and figures |
| 5 | `revision\phase5_external_medgemma.bat` | MedGemma-4B; external datasets; Table 10 |
| 6 | `revision\phase6_segformer_external.bat` | SegFormer + MedGemma-4B in-domain and external; Tables 7 and 10 |

`python revision/analyze.py --dir results/kvasir` regenerates the tables from the included logs without any GPU or VLM.

## Citation

Citation details will be added on publication.
