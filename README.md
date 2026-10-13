# DocMIL: Weakly-Supervised Document Forgery Detection via Attention-Based Multiple-Instance Learning

Code, split indices, per-document predictions and result tables for the IEEE Access article
*DocMIL: Weakly-Supervised Document Forgery Detection via Attention-Based Multiple-Instance Learning*
(F M Javed Mehedi Shamrat, Syed Muntasin Fayaz, Mostofa Mahmud Hasan, Rayed Ghazawi, Mazen Omar Khairo, Md. Asraf Ali).

DocMIL treats a document image as a bag of multi-scale patches and learns a gated attention pooling from
document-level labels only (no pixel masks, no synthetic pretraining). The evaluation protocol builds every
evaluation bag **without consulting any region annotation**, reports results under exhaustive tiling, and
includes evidence-removal and label-permutation controls.

## Repository layout

```
notebooks/
  docmil_findItAgain.ipynb   # full pipeline, DATASET = "fia"
  docmil_rtm.ipynb           # full pipeline, DATASET = "rtm"
results/
  fia/  rtm/
    splits.json              # document IDs of every fold and of the test set
    manifest.json            # configuration, seeds, software versions, cache counts
    ledger.jsonl             # one line per trained model with its test metrics
    tables/                  # every CSV behind the paper, LaTeX tables, numbers.tex
    preds/                   # per-document test and validation predictions (.npz)
figures/                     # vector PDFs of all figures in the paper
requirements.txt
```

Both notebooks contain the same code; they differ only in the `DATASET` switch in the configuration cell.

## Data

The datasets are not redistributed here. Obtain them from their authors:

- **FindItAgain** (Martínez Tornés et al., ICDAR 2023): https://l3i-share.univ-lr.fr/2023Finditagain/index.html
- **Real Text Manipulation (RTM)** (Luo et al., Pattern Recognition 2025): https://github.com/DrLuo/RTM
  (we use the 9,000-image release with `JPEGImages/`, `SegmentationClass/`, `train.txt` and `test.txt`)

One forged FindItAgain validation image (`X51006619709.png`) listed in `val.txt` is absent from the
distributed archive; the code reports and skips it (987 usable receipts, 162 forged).

## Running

1. Open a notebook (we used Kaggle with one NVIDIA Tesla T4; any CUDA GPU with 16 GB works).
2. In the configuration cell set `DATASET = "fia"` or `"rtm"` and edit `root` in `PRESETS` to your dataset path.
3. Run all cells top to bottom.

Every run is resumable. Per-epoch checkpoints and a JSONL ledger let a restarted session skip finished work,
and the exhaustive-tiling evaluation is written in shards. The tile cache is deterministic (seed 1234), so it
can be rebuilt identically in a new session. Approximate cost on a T4: about 25 GPU-hours for FindItAgain and
35 to 40 for RTM.

Software used for the reported results: Python 3.13, PyTorch 2.11.0, torchvision 0.26.0, CUDA 12.8
(full list in `requirements.txt`). ImageNet weights: torchvision `MobileNet_V3_Small_Weights.IMAGENET1K_V1`.

## Protocol summary

| Item | Setting |
|---|---|
| Tiling | scales {128, 192} px, stride 0.75 × scale, white padding to the largest scale |
| Evaluation bag | first K_max tiles (24 FIA / 16 RTM) of an annotation-blind random list seeded by (1234, document index) |
| Exhaustive tiling | every tile of the full grid, one softmax over the whole document |
| Training bag | K_b tiles (16 / 12) drawn each epoch from the first K_train (48 / 24) |
| Region use (optional, training documents only) | auxiliary patch loss (λ = 0.3, θ_min = 12 px²) and region-aware sampling; both off in the document-label-only variant |
| Cross-validation | 5 folds (FIA) / 3 folds (RTM), split seed 0; official test set held out |
| Seeds | 0, 1, 2 for the principal configurations; 0 for secondary runs |
| Selection | best validation ROC-AUC epoch; F1-optimal validation threshold with 4-view flip TTA |
| Statistics | BCa bootstrap (5,000 class-stratified resamples), inverted BCa p-values, Holm correction, fold random-intercept mixed model |

## Paper to file mapping

Paths are relative to `results/<corpus>/tables/` unless stated.

| Paper | File(s) |
|---|---|
| Table 1 (datasets) | `counts.csv`, `listing.csv`, `cache_report.csv`, `format_audit.csv` |
| Table 2 (hyperparameters) | `hyperparameters_by_corpus.csv`, `../manifest.json` |
| Tables 3 and 4 (main results) | `table_main.csv`, `summary_all.csv`, `classical.csv`, `dct_baseline.csv` |
| Table 5 (ablations and controls) | `summary_all.csv` |
| Table 6 (paired statistics) | `stats_pairs_bag.csv`, `stats_pairs_exh.csv`, `stats_vs_chance_bag.csv` |
| Table 7 (per technique, leave-one-technique-out) | `per_type.csv`, `leave_one_technique_out.csv` |
| Table 8 (attention-annotation agreement) | `localization.csv`, `old_attention_diagnostic.csv`, `old_bag_sizes.csv` |
| Table 9 (threshold calibration) | `threshold_prevalence.csv` |
| Table 10 (sensitivity) | `sensitivity.csv`; data scaling in `learning_curve.csv` |
| Table 11 (efficiency) | `efficiency.csv` |
| Exhaustive vs. bag evaluation | `exhaustive_vs_bag.csv` |
| Benign-annotation false positives (FIA) | `benign_false_positives.csv` |
| Numbers quoted in the text | `numbers.tex` (LaTeX macros) |
| Figures 6 and 8 to 14 | `figures/fig_fia_*.pdf`, `figures/fig_rtm_*.pdf` |

Every table can be regenerated from `ledger.jsonl` and `preds/` by running the analysis cells of the notebook
without retraining.

## Included but not part of the reported results

- `RUN_OCR` cell: OCR, receipt-consistency features, a frozen LayoutLMv3 probe and late fusion with DocMIL.
- `adcd_predict_map` hook: scores any external pixel-level localiser (for example ADCD-Net) zero-shot on the same
  test splits by max-aggregation of its map.
- Cross-corpus transfer cell (`OTHER_OUT_DIR`).

## Citation

```bibtex
@article{shamrat2026docmil,
  author  = {Shamrat, F M Javed Mehedi and Fayaz, Syed Muntasin and Hasan, Mostofa Mahmud and
             Ghazawi, Rayed and Khairo, Mazen Omar and Ali, Md. Asraf},
  title   = {{DocMIL}: Weakly-Supervised Document Forgery Detection via Attention-Based
             Multiple-Instance Learning},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The datasets remain under their original licences.
