# When to Ask for Help: Confidence-Gated Escalation to Vision-Language and Segmentation Foundation Models for Polyp Segmentation

Code, trained gates and per-image result logs for the paper

> A. I. Xavier, *When to Ask for Help: Confidence-Gated Escalation to Vision-Language and Segmentation Foundation Models for
> Polyp Segmentation*, Biomedical Signal Processing and Control (under revision, manuscript BSPC-D-26-16498).

## Idea

A lightweight backbone segments every colonoscopy image. A more expensive model can be asked to review uncertain cases, but it
can also damage masks that were already correct. A learned gate therefore decides, per image, whether to escalate, and an
acceptance step decides whether the escalated output replaces the backbone mask.

```
image -> (1) backbone: mask m + confidence c
      -> (2) four mask-level features
      -> (3) logistic-regression gate: escalate if P(poor mask) >= theta
      -> (4) escalation target: (a) a VLM proposes expand / shrink / shift / no change, or
                                (b) MedSAM re-segments from a box around m       -> candidate m'
      -> (5) acceptance: (a) causal verification (faithfulness F >= tau), or
                         (b) a learned acceptance rule fitted on the validation split
```

## Main findings (see the paper for all qualifications)

* **VLM editing (negative result).** Escalating every image to Llama-3.2-11B-Vision or the medical VLM MedGemma-4B lowered mean
  IoU on Kvasir-SEG from 0.745 to 0.716. The gate escalated 30% of images and avoided 82-84% of that loss with 70% fewer VLM
  calls than escalating every image. With three SegFormer seeds it avoided 92% of the loss in-domain and 56-91% on three
  external datasets. No VLM policy improved on the backbone alone; even perfect selection of the recorded edits would add at
  most 0.009 IoU.
* **MedSAM with learned acceptance (bounded positive result).** Gated escalation to MedSAM with a learned acceptance rule
  improved PolypUNet by +0.012 to +0.017 IoU on CVC-ClinicDB, CVC-ColonDB and ETIS (exploratory, because these results informed
  the rule's design) and by +0.021 IoU (95% CI 0.018-0.024) in a retrospective external evaluation of the frozen pipeline on
  1,411 PolypGen images from six centres. This is image-level evidence for the evaluated pipeline, not evidence of independent
  clinical generalization: MedSAM's pretraining may include some public polyp images, and PolypGen has no patient identifiers.
  A one-parameter change-size rule reached +0.013 IoU, and a stronger backbone alone (SegFormer, 0.752 IoU) is the better route
  to accuracy where it can be deployed.

## Repository layout

| Path | Contents |
|---|---|
| `models/` | PolypUNet backbone with confidence head, confidence gate, VLM interface, causal verification |
| `data/` | Kvasir-SEG loader and transforms (no images are included) |
| `train_backbone.py`, `train_gate.py`, `run_pipeline.py`, `eval.py` | Original training and pipeline scripts |
| `ablations/` | never-refine, always-refine and no-verify configurations |
| `app.py` | Streamlit demonstration of the stages on a single image |
| `revision/` | Scripts for every experiment and analysis of the revised paper (tables below) |
| `gates/` | Trained gates (`gate.joblib`), validation reports and training logs for every backbone |
| `results/kvasir/`, `results/validation/`, `results/external/` | Per-image logs of every run and summary tables |
| `results/polypgen_checks/` | PolypGen preparation and de-duplication reports |

Backbone weights (`*.pt`) are not included because of their size; `revision/phase1_train_backbones.bat` and
`revision/train_baselines.py` retrain them. All analyses of the paper can be recomputed from the included logs without any
weights, GPU or VLM.

## Setup

```bash
uv venv && uv pip install -r requirements.txt     # exact versions: requirements_lock.txt
cp .env.example .env                             # add NVIDIA_NIM_API_KEY only for the Llama runs
```

Local VLMs run through [Ollama](https://ollama.com): `ollama pull qwen2.5vl:7b` and `ollama pull medgemma:4b`. MedSAM is
loaded from Hugging Face (`flaviagiammarino/medsam-vit-base`, a port of MedSAM, Ma et al., Nature Communications 2024); set
`MEDSAM_DEVICE=cpu` to run it on the CPU, in which case its outputs agree with GPU runs up to floating-point differences.

### Data (not included; please follow each dataset's terms and cite it)

* Kvasir-SEG (Jha et al., MMM 2020): <https://datasets.simula.no/kvasir-seg/> -> `data/Kvasir-SEG/images`, `data/Kvasir-SEG/masks`
* CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB: the PraNet `TestDataset` release -> `data/TestDataset/<name>/images`, `.../masks`.
  The CVC-300 and Kvasir subsets of that release are not used: CVC-300 duplicates 60 CVC-ColonDB images, and the Kvasir subset
  is drawn from Kvasir-SEG.
* PolypGen (Ali et al., Scientific Data 2023; EndoCV2021, Scientific Reports 2024): from its authors on Synapse (syn45200214).
  It is **not redistributed** here. Run `python revision/prepare_polypgen.py <download folder>` to write
  `data/PolypGen/C1 ... C6/{images,masks}` (frames without a polyp mask removed) and to repeat the duplicate check.

## Recomputing the reported numbers from the included logs (no GPU)

```bash
python revision/setup_results_dirs.py      # copies results/* to results_rev/, results_val/, results_ext/
revision\reproduce_section49.bat           # Windows; recomputes every number of Section 4.9
python revision/analyze.py --dir results_rev   # summary tables of all escalation runs
```

`revision/REPRODUCE_EXPECTED.txt` lists the values reported in the paper for comparison.

## Rerunning the experiments (Windows batch files; each step resumes if interrupted)

| Step | Script | Paper |
|---|---|---|
| 1 | `revision\phase1_train_backbones.bat` | Backbones (3 seeds x 2 augmentations, lambda ablation), gates; Table 8 |
| 2 | `revision\phase2_eval_vlm.bat` | Escalation policies with Llama-3.2-11B and Qwen2.5-VL-7B; Tables 3 and 7 |
| 3 | `revision\phase3_baselines.bat` | Matched baselines and efficiency; Tables 5 and 9 |
| 4 | `revision\phase4_analyze.bat` | Summary tables and figures |
| 5 | `revision\phase5_external_medgemma.bat` | MedGemma-4B; external datasets; Table 10 |
| 6 | `revision\phase6_segformer_external.bat` | SegFormer + MedGemma-4B in-domain and external; Tables 7 and 10 |
| 7 | `revision\phase7_validation_segformer_seeds.bat` | VLM answers on the validation split; SegFormer seeds 1-2; Sections 4.5, 4.8, 5.2 |
| 8 | `revision\phase8_medsam.bat` (with `medsam_refiner.py`) | MedSAM as escalation target; Section 4.9, Table 11 |
| 10 | `revision\phase10_polypgen.bat` | Retrospective external evaluation on PolypGen; Section 4.9, Tables 11-12 |

Phase 9 (`phase9_cvc300_confirm.bat`, `PHASE9_PREREGISTRATION.txt`) is kept for transparency only: it was abandoned because
CVC-300 duplicates CVC-ColonDB images, and its results are not used in the paper.

## Analysis scripts of the revision

| Script | Purpose | Paper |
|---|---|---|
| `predeployment_check.py` | Pre-deployment assessment, harm avoided, confidence intervals, AUROC | Tables 7, 10 and 13 |
| `operator_ceiling.py` | Best of the four edit operations per image (needs masks saved with `--save-masks`) | Section 4.5 |
| `control_sensitivity.py` | Ten random control draws for causal verification | Section 3.5 |
| `learned_acceptance.py` | Acceptance rule fitted on validation only, applied unchanged | Section 4.9, Table 11 |
| `polypgen_confirm.py` | PolypGen primary analysis, per-centre results | Section 4.9, Tables 11-12 |
| `acceptance_baselines.py` | Simpler acceptance rules (change size, faithfulness, confidence) | Supplementary Table S6 |
| `acceptance_population.py`, `acceptance_agreement.py` | Fitting on all vs. gated validation images; decision agreement and calibration | Algorithm 2 notes |
| `acceptance_grouped_cv.py` | Image-grouped cross-validation of the acceptance rule | Algorithm 2 notes |
| `safety_and_clustering.py` | Per-image gains and losses, severe failures, cluster-aware intervals | Table 12, Supplementary Table S7 |
| `test_algorithm2.py` | Checks that the code implements Algorithm 2 (box, empty mask, denominator, area feature) | Algorithm 2 |

`revision/PHASE10_PREREGISTRATION.txt` is our internal analysis plan for PolypGen, written before the download, with the outcome
appended. It keeps its original file name, but its timing rests on our own records, so it is not an externally verifiable
preregistration, as stated in the paper.

## Citation

Citation details will be added on publication.
