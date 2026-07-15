# Jukebox dependency-slice experiment

This branch is an isolated, non-production experiment. It does not change the
`master` branch or claim that the Jukebox runtime dependency can be removed.

## Exact consumed surface

`sheetsage/representations/jukebox.py` consumes four symbols:

- `jukebox_infer.hparams.Hyperparams`
- `jukebox_infer.make_models.make_model`
- `jukebox_infer.prior.conditioners.RangeEmbedding`
- `jukebox_infer.utils.torch_utils.empty_cache`

The transitive model-construction closure is 22 Python modules and about 3,918
lines (including VQ-VAE, transformer, prior, labels, checkpoint and download
code). The closure is not a small utility slice: it contains the model math,
checkpoint loading, remote download, distributed adapter, and data labelers.

## Prototype and compatibility switch

`sheetsage._jukebox` defines a private four-symbol contract. The `slice` path
actually owns the small, dependency-free `Hyperparams` mapping; model
construction, conditioners, and cache release remain external. The default
`SHEETSAGE_JUKEBOX_BACKEND=external` path resolves the published
`jukebox-infer` package. `SHEETSAGE_JUKEBOX_BACKEND=slice` selects the same
symbols through the private seam, allowing import and mock parity checks without
vendoring the 3,918-line closure. Public SheetSage APIs are unchanged.

## Assumptions and ownership

Jukebox checkpoints remain OpenAI-hosted artifacts and are also referenced by
SheetSage assets. `jukebox-infer` is MIT; OpenAI model weights retain their
upstream terms (see `NOTICE`). Copying model code into SheetSage would duplicate
ownership and checkpoint logic, and would require keeping two implementations in
sync.

## Evidence and recommendation

The seam tests prove symbol identity and reversible wiring; full model parity
requires multi-GB checkpoints and a CUDA device, so it was not run here. The
measured closure is too large for a clean private extraction. Keep
`jukebox-infer` as the runtime dependency for now. If dependency removal becomes
important, extract a separately owned, smaller support package from the closure;
do not copy it into SheetSage.

`nav-audit` findings on this experimental shape: the private seam is narrow and
explicit, but the underlying external dependency still owns model architecture,
checkpoint URLs, and license boundaries. No production-readiness claim is made.

## Read-only nav-audit (8 rules)

- **1 Information hiding:** the four-symbol seam is narrow; checkpoint and
  architecture decisions remain owned by `jukebox-infer`. The 502-line
  `representations/jukebox.py` still combines decode, codify, activation hooks,
  conditioning, and singleton compatibility (warning).
- **2 Interface-first:** `_jukebox` is a private package door; no new public
  symbols were added.
- **3 Explicit dependencies:** backend selection is explicit or controlled by
  one documented environment switch; no global model manager was introduced.
- **4 Right grain:** the existing 502-line extractor is a giant-module warning;
  the experiment intentionally does not split production behavior on this
  branch.
- **5 Framework fit:** Python imports are conventional and the backend contract
  is a frozen dataclass.
- **6 Rearrange, don't rewrite:** the change only rewires imports and adds a
  reversible seam; model behavior is delegated unchanged.
- **7 Confidence boundary:** full numerical parity is bounded by unavailable
  multi-GB checkpoints/CUDA, so this remains an experiment rather than a merge
  recommendation.
- **8 Agent navigability:** the new private files are describable in one
  sentence; the legacy extractor requires multiple responsibilities in its
  description, confirming the grain warning above.
