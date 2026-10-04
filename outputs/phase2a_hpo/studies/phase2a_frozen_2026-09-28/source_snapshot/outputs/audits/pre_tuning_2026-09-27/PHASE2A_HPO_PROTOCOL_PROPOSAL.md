# NeuroBridge — Phase 2A validation-only HPO protocol proposal

Frozen design protocol, approved 28 September 2026. **Not implemented: no fit, search, test evaluation, code/config edit, embedding regeneration, or V3 run was performed.** `FACT` means verified in the frozen V2 audit, lockdown, current source, or existing compute sidecars; `INFERENCE` means interpretation; `RECOMMENDATION` records the approved prospective protocol. The binding inputs are [NEUROBRIDGE_PRE_TUNING_AUDIT.md](../pre_tuning_2026-09-26/NEUROBRIDGE_PRE_TUNING_AUDIT.md) and [PHASE2_PRE_TUNING_LOCKDOWN.md](PHASE2_PRE_TUNING_LOCKDOWN.md). V2 remains immutable. Geometry/structural fidelity, information accessibility, and temporal fidelity remain separate outcomes; there is no aggregate “best model” score.

## 1. Executive recommendation

**RECOMMENDATION (approved).** Use the sealed **Sobol low-discrepancy candidate design** in Section 4 over three optimization/objective-scale variables—learning rate, weight decay, and the active temperature—within each architecture × objective cell. Keep architectures and all scientific supervision/data semantics fixed. Include the exact V2 hyperparameter triple as a **new V3-provenance baseline candidate**; do not silently reuse historical V2 checkpoints with mismatched source provenance. Tune Synthetic and Real **separately**, but give the same candidate triple to both Synthetic A/B and, separately, to Real TOTAL65/A/B. Use validation only for checkpoint selection, anti-collapse qualification, and candidate ranking. Do not inspect V2 or Phase-2 test, synthetic Z, lag, decoding, RSA, CKA, or Procrustes as HPO selectors.

**FACT.** The existing [lockdown](PHASE2_PRE_TUNING_LOCKDOWN.md) establishes distinct objective losses and a validation-only geometric near-collapse proposal. Transformer failures vary by seed and population; their cause is unknown. The current real runner files do not match hashes in a representative saved historical fit, so a prospective HPO runner needs its own complete source snapshot. The numeric ranges and budget below are **new protocol proposals**, not V2 bug fixes and not conclusions from held-out performance.

## 2. Frozen quantities

| Quantity | Phase-2A proposal | Rationale / provenance |
|---|---|---|
| Scientific factors | CNN1D vs Transformer; `soft`, supervised multi-positive `infonce`, `time_contrastive_blocks`, `behavior_contrastive_blocks` remain four separate objectives. | No cross-objective raw-loss ranking or objective replacement. `src/neurobridge/losses/infonce.py:14–43,97–162`. |
| Model/data shape | Existing architecture layouts and capacity, 3D unit training embeddings, 21-bin window, stride 1, global batch 1024, actual effective candidate count, input preprocessing/normalization, trial-level split. | Current synthetic config `src/neurobridge/experiments/staged_shared_latent.py:67–123`; canonical real fit guard `src/neurobridge/experiments/real_monkey_validated.py:591–596`. |
| Scientific supervision | `metadata_temperature=0.5`, time/condition soft-target weights, positive offset `+10`, current behavior/time marginal-negative samplers, condition labels, validity masks. | Changing these changes target/data semantics, not merely optimization. See lockdown Sections 2–6. |
| Data lineages | Synthetic generator/dataset realization and split **42**; Real preloaded source and R0 split **42**; exact Real TOTAL65 and fixed random A/B partition with `channel_split_seed=42`. | Training seed is a separate stream. No partition chosen on performance. Existing V2 provenance/splits and `outputs/runs/channel_split_seed_42.json`. |
| Branches | HPO fits are `held_out` **train + validation only**. No full-sample refit, R10 fit, controlled-lag evaluation, scientific Stage-5 test evaluation, or MATLAB export during candidate search. | Full-sample is descriptive; R10 is transform-only downstream of R0. |
| Optimizer/schedule form | AdamW; current optimizer update semantics and best-checkpoint/early-stopping distinction; no scheduler, clipping, AMP, or architecture change introduced as a hidden HPO factor. | `staged_shared_latent.py:824–931`; `real_monkey_validated.py:651–659,762–877`. |

**FACT.** `RealMonkeyConfig` *bare legacy defaults* include batch 256 and cap 2000 (`src/neurobridge/experiments/real_monkey.py:77–80`), but the **current canonical validated path** builds max 4000 and rejects batch ≠1024 (`real_monkey_validated.py:91–117,591–596`). The prospective baseline is the canonical validated V2 setting, not an accidental invocation of bare defaults. Synthetic A/B use base training seed and base+1 in the existing runner (`staged_shared_latent.py:824–831`); this mapping must be retained or explicitly versioned, never confused with dataset seed.

## 3. Tunable quantities

**RECOMMENDATION.** Exactly one active temperature per objective-family fit, plus `learning_rate` and `weight_decay`:

- `soft`: search `embedding_temperature`; keep `metadata_temperature=0.5` fixed because it constructs the target distribution.
- Supervised multi-positive `infonce`: search `embedding_temperature` for the in-batch cosine logits.
- `time_contrastive_blocks` and `behavior_contrastive_blocks`: independently search their *candidate-specific* `cebra_temperature` / public `temporal_objective_temperature`. They share a **range**, not a fitted value, and no universal “Transformer temperature” exists.

There is **no** temperature search across objectives by pooling their validation losses. Architecture width/depth/heads/kernel/FFN are reserved for a conditional Phase-2B decision. Window, stride, offset, soft-target metadata temperature, behavior/time sampling distributions, labels, split, batch, embedding dimension, and normalization are not Phase-2A tunables. Their alteration would require a **new scientific protocol/objective or data comparison**, not be disguised as HPO. Ranges are identical for CNN and Transformer to give both comparable opportunity; independent final selections within each architecture are allowed.

## 4. Proposed numerical search spaces

| Parameter | Current V2 value actually consumed | Proposed Phase-2A range | Sampling | Numerical rationale, **not a V2 causal finding** |
|---|---:|---:|---|---|
| AdamW learning rate, both architectures/objectives | `1e-3` | `[1e-4, 3e-3]` | Sobol unit coordinate mapped logarithmically. | Extends exploration below V2 by a factor 10 while retaining a compact upper bound. Whether V2's optimization scale contributes to any near-collapse is **unknown**. `staged_shared_latent.py:108–110,828`; `real_monkey.py:78–79`; `real_monkey_validated.py:658–660`. |
| AdamW weight decay, both architectures/objectives | `1e-4` | `[1e-5, 1e-3]` | Sobol unit coordinate mapped logarithmically. | One order of magnitude either side, strictly positive for a simple log scale; no zero-decay subprotocol is added. Same source lines. |
| `embedding_temperature` for `soft` | `0.1` | `[0.05, 0.20]` | Objective-specific Sobol coordinate mapped logarithmically. | Normalized cosine is bounded `[-1,1]`; endpoint logit magnitudes at most 20 or 5, versus 10 now, handled by log-sum-exp. Target-kernel temperature stays 0.5. |
| `embedding_temperature` for supervised `infonce` | `0.1` | `[0.05, 0.20]` | Objective-specific Sobol coordinate mapped logarithmically. | Same bounded in-batch logit scale; sampled objectives have a different formula and therefore a different range. |
| `cebra_temperature` for **time** triplets | `1.0` | `[0.1, 2.0]` | Objective-specific Sobol coordinate mapped logarithmically. | Cosine scores lie in `[-10,10]` at the lower endpoint and `[-0.5,0.5]` at the upper. This probes a wider objective-scale regime before considering Phase-2B; it does **not** presume that lower temperature prevents collapse. |
| `cebra_temperature` for **behavior** triplets | `1.0` | `[0.1, 2.0]` | Separate objective-specific Sobol stream, mapped logarithmically. | Same broader numerical range as time because the loss form is shared, but the positive distribution and candidate selection remain distinct. Collapse cause **unknown**. |

**Approved candidate generation and seal.** Candidate 0 is exactly `(lr=1e-3, weight_decay=1e-4, active temperature=0.1 for soft/supervised or 1.0 for time/behavior)`. For each objective independently, use SciPy **1.13.1** `scipy.stats.qmc.Sobol(d=3, scramble=True, seed=20260928+j).random_base2(m=3)`, where `j=0,1,2,3` in objective order `soft`, `infonce`, `time_contrastive_blocks`, `behavior_contrastive_blocks`. Map each unit coordinate `u` to its approved bound `[a,b]` by `x=exp(log(a)+float(u)×(log(b)-log(a)))`, in column order LR, weight decay, active temperature. Take **only the first five** of the eight generated points as candidates 1–5; points 6–8 are **not authorized candidates**. The five-point prefix loses some full-power-of-two Sobol balance guarantees, but provides a prespecified space-filling design without adaptive tuning. Objective-specific scrambles keep time/behavior streams separate. Use the **same sealed numerical triple** for CNN and Transformer and for Synthetic and Real at a given objective/index; domain selections remain separate. Indices **1–3** are initial, **4–5** are used only if the unchanged Section-7 extension trigger fires. The sealed decimal strings below, not runtime regeneration, are authoritative for later implementation. Proposal seeds are separate from training seeds.

```csv
objective,candidate,lr,weight_decay,temperature
soft,1,0.0010717221240559393,0.00036625451895020913,0.11065361153401906
soft,2,0.00019749937882922414,4.8234679132582012e-05,0.092849306559744341
soft,3,0.00032133362426250305,0.00010626568831948351,0.19424030526336192
soft,4,0.0017259009787235053,1.6607005430706566e-05,0.05768593385718452
soft,5,0.0029055050268657467,0.00026962609402288125,0.083978728256776464
infonce,1,0.0019186487924144232,0.0001147040509290496,0.10165049742890192
infonce,2,0.0005372687003035359,2.9623432092428512e-05,0.07553731701182094
infonce,3,0.0001677985560548701,0.00068936488845545226,0.18094199691102594
infonce,4,0.00059424393499990795,4.2684009506751506e-05,0.060339100904803183
infonce,5,0.0011716776180662248,0.0003416774296593524,0.085609803258813358
time_contrastive_blocks,1,0.0019222973497531495,6.6598057836530394e-05,0.78816862651140207
time_contrastive_blocks,2,0.00016375435299976828,0.00011375137967953784,0.1735617792707736
time_contrastive_blocks,3,0.00052740769928552762,2.930753191465412e-05,1.0494153783666149
time_contrastive_blocks,4,0.0005953717238907127,0.0004659247384656004,0.278649686841551
time_contrastive_blocks,5,0.001105113045950303,1.1858717700174355e-05,0.12416733293043147
behavior_contrastive_blocks,1,0.0029564921407279443,0.00090193916481522108,1.0835783554613034
behavior_contrastive_blocks,2,0.00010511370350211826,1.3389195739766575e-05,0.38578572211598844
behavior_contrastive_blocks,3,0.00032432705417700448,0.00022893263429992996,0.66763904993205703
behavior_contrastive_blocks,4,0.00090333630870399794,3.4252126668427683e-05,0.13676827492083402
behavior_contrastive_blocks,5,0.00056419976192669235,0.0001396505805253916,0.25922799348919939
```

Seal checksum: SHA-256 `196d0b24251ed43f2f781af635a3cb1c37abf43e5e6e7d8ed66cd261aa616f6c` over the **20 data lines only**, UTF-8, comma-separated exactly as shown, LF after every line including the last; header excluded. The values were generated for this protocol freeze **without a fit**. A runner must verify this list/hash before use and must not adapt or regenerate candidates based on validation/test outcomes. The bounds and Sobol design do not assert that temperature or LR causes any observed collapse; **CAUSE UNKNOWN**.

## 5. Early-stopping decision

**FACT — two different rules.** In both domains, **best-checkpoint selection** updates the saved state whenever the current validation loss is *strictly below* the lowest raw validation loss so far; this applies from the **first** check, even before minimum updates. **Stopping** starts only at the first check at/after `min_steps`. Its first eligible loss initializes a separate `meaningful_reference`. At each later eligible check, `relative_gain = (meaningful_reference - current_validation_loss) / max(abs(meaningful_reference), 1e-12)`. Patience resets and the reference changes only if `relative_gain ≥ 0.001`; otherwise stale-check count increases. Thus small raw improvements can update the best checkpoint **without** resetting patience. Stop when stale checks reach patience before max; selected checkpoint can greatly precede stopping update. This is not a discrepancy to “fix” during HPO. Evidence: `staged_shared_latent.py:933–968,976–1002`; `real_monkey_validated.py:800–877,879–913`.

| Domain | Checks | First patience-eligible reference | Patience | Relative reset | Max | Earliest possible patience stop; first selectable checkpoint |
|---|---|---:|---:|---:|---:|---|
| Synthetic | Every 400 updates | 800 | 3 subsequent stale checks | `≥0.001` | 4000 | 2000; first best can be 400. |
| Real R0 | Every 150 updates, also max if due | 750 | 5 subsequent stale checks | `≥0.001` | 4000 | 1500; first best can be 150. |

**Alternative 1: preserve the schedules.** This retains V2 optimization exposure/checkpoint semantics for domain-specific comparisons, permits direct within-domain V2 vs prospectively rerun baseline discussion (subject to provenance limitations), and avoids adding a fourth tuned factor. Synthetic and Real are already tuned separately below, so their different validation frequencies do not enter a single pooled loss selection. Drawback: nominal max 4000 does **not** imply equal actual update or validation-query exposure across domains; cross-domain training-efficiency comparisons must report actual updates and check counts.

**Alternative 2: prospectively harmonize V3.** A single check interval/min/patience could simplify cross-domain operational comparison. But it changes the number and timing of validation opportunities and the effective stopping distribution, so V3 outcomes would differ from V2 for at least two simultaneous reasons. A new harmonized baseline would be required and the schedule itself would need a scientific justification independent of results. It would not retroactively repair V2.

**RECOMMENDATION.** Preserve the existing **domain-specific** schedules for Phase-2A, freeze them verbatim before implementation, and report actual update/check count per fit. Do not normalize silently. No checkpoint is reselected using geometry; geometry only decides whether the preselected raw-loss checkpoint is eligible as a *candidate*.

## 6. HPO algorithm

| Method | Advantage | Problem here | Decision |
|---|---|---|---|
| Sealed scrambled-Sobol design with a forced V2-value baseline | Prespecified spread across the 3-D log space; auditable candidate list; simple parallel/resume logic; no surrogate assumptions; fair paired triples. | Five Sobol points plus baseline remain a small design and can miss a narrow stable region; partial eight-point block is not fully balanced. | **Approved** bounded 4→6 design with explicit instability outcome. |
| Pure seeded random search | Simple and reproducible with the same budget. | At only 3–5 nonbaseline draws it can cluster by chance, leaving broad parts of the approved space unvisited. | Superseded by the sealed Sobol list, not executed. |
| Optuna/TPE | Can adapt subsequent samples to promising regions. | Four-to-six candidates per cell/population give a weak surrogate; collapse means censored/ineligible trials; asynchronous scheduling/order can change suggestions; more provenance and complexity. | Not justified initially; “Optuna” is not intrinsically more scientific. |
| Successive halving / early pruning | Saves updates on clearly poor fits. | Current best-checkpoint timing and patience already vary by domain; early loss near `ln B` or early geometry need not predict later recovery. Pruning introduces unequal stopping rules and may systematically hide instability. | Do **not** prune in Phase-2A; let existing early stopping act. |

**Approved exact sequence.** Freeze source/config/data and use the candidate 0 baseline plus the **already sealed Section-4 Sobol triples**; do not redraw them at run time. Use indices 0–3 initially and the identical ordered triples for the two architectures and both domains. Execute **held-out train/validation only** on training seed 1101 for the two Synthetic populations and three Real R0 populations. Each architecture × objective × domain candidate triple yields two or three fits, not one. Qualify each saved **best-validation** checkpoint using the Section-9 validation-only algorithm. After initial four candidates, apply the paired architecture extension rule in Section 7; an extending domain uses its sealed indices 4–5. Re-evaluate the top two eligible candidate triples per cell on training seeds **1201 and 1301**; these are **HPO robustness seeds**, not the later confirmatory multi-seed study. Select a domain-level triple for each architecture × objective from eligible finalists. If none qualifies, report no winner/unstable. Do not fit full-sample or evaluate test during this sequence.

The synthetic runner currently maps base training seed to A and base+1 to B; retain and record that mapping. Real uses the base training seed separately for TOTAL65/A/B. Dataset generator seed 42, split seed 42, and channel-partition seed 42 remain distinct fixed fields; coincident numerical values do not mean one RNG stream. The sampler and loader RNG are coupled in current code, so capture that provenance honestly instead of inventing an independent sampler seed.

## 7. Trial budget

**RECOMMENDATION.** The unit called an “HPO trial” is a **candidate hyperparameter triple within one architecture × objective × domain cell**, evaluated on every population in that domain. A “fit” is one population × candidate × training seed checkpoint. There are 8 architecture × objective cells per domain, **16 domain-cells** total, and five population units (Synthetic A/B; Real TOTAL65/A/B).

| Stage | Candidate trials per domain-cell | Fits per Synthetic domain-cell | Fits per Real domain-cell | Total fits over 8 cells/domain |
|---|---:|---:|---:|---:|
| Initial paired Sobol design | 4, including exact V2-value baseline | `4×2=8` | `4×3=12` | `8×(8+12)=160` |
| Optional paired extension | +2, only for triggered objective×domain pair, **both architectures** | +4 | +6 | At most `8×(4+6)=80` |
| Two finalists, two additional HPO seeds | At most 2 previously evaluated triples; no new configuration drawn | `2×2×2=8` | `2×2×3=12` | At most `8×(8+12)=160` |
| Total | 4–6 distinct candidate triples/cell; 1–3 HPO seeds on finalists | — | — | **Up to 320 fits without extension; 400 fit ceiling** |

**Extension trigger, fixed in advance:** after four search candidates on seed 1101, if **either** architecture in an objective × domain pair has fewer than **two** candidate triples whose best checkpoints pass all-population qualification, activate the **two already sealed Sobol triples 4–5** for **both** architectures in that objective × domain pair. Otherwise no extension. Never extend based on test or scientific outcomes. After six candidates, do not extend again even if none qualify; report instability/no winner. This keeps CNN and Transformer matched within each objective/domain pair. The conditional extension can give different objectives/domains different counts, but the trigger and counts are disclosed rather than treating failed fits as free. Crashes count against the budget except the narrow infrastructure retry rule in Section 10. If fewer than two eligible candidates remain, reevaluate the one available candidate; if zero, stop that cell without finalists.

The equality of candidate counts is a **budget rule**, not a claim that every cell needs the same number to locate a stable region. The 4→6 cap is deliberately small; if failures are pervasive, the correct result may be “unstable under Phase-2A search,” not an unbounded experiment factory.

## 8. Seed strategy

**RECOMMENDATION: staged search with limited finalist re-evaluation.** Search seed **1101** for all candidate triples; use new HPO-only training seeds **1201, 1301** for up to two finalists. Reusing the same root seed across candidates controls a major random variation source, while multiple finalist seeds test whether the apparent gain is tied to one initialization/order. The 100-step separation also keeps Synthetic B's `base+1` seed distinct across stages. Seed 42/123/456 V2 results remain frozen historical evidence and are **not** Phase-2A fit credits; 1101/1201/1301 are not reserved final confirmatory seeds. A later confirmatory stage must use separately declared seeds/data/splits and cannot be folded back into HPO selection.

**FACT.** Synthetic A/B seed handling is base/base+1; current synthetic generator seed and split seed are 42. Real trial split and A/B partition seeds are 42, independently specified. Triplet validation pairs are fixed across checks in the real runner, with validation order seed `training_seed+10000` (`real_monkey_validated.py:651–736`); sampler pairing and batch order share the loader generator. Synthetic validation subset and triplet draws are created before fit from a seeded execution path (`staged_shared_latent.py:824–885`). **RECOMMENDATION.** Save validation candidate indices/pair identities or their hash when building the HPO runner to ensure that a same-seed comparison uses the same validation estimator. This is provenance/qualification, **not** a sampler-objective change. If implementation cannot preserve a fixed estimator across candidates, flag that as a protocol blocker before training, not as an after-the-fact interpretation.

## 9. Validation-only selection algorithm

**RECOMMENDATION — apply independently within each architecture × objective × domain cell; do not compare different objectives' raw losses.**

1. Fit candidate on **train trials only** with the frozen domain-specific schedule, and save full history, stopping state, and the **strict minimum raw validation-loss** checkpoint. A check before `min_steps` can still be the selected checkpoint.
2. Validate artifact shape/cardinality, finite weights/loss/embedding and unit norm; derive validation `N×3` embeddings from the selected checkpoint on the **prespecified valid validation-window scope**. Never draw test windows. Fix the pair diagnostic sampling convention at RNG 100, at most 512 validation rows, and threshold `distance <0.01` from the lockdown. The PI-approved near-collapse rule is **frozen unchanged**: `distance_q50 <0.02` **and** `fraction(distance<0.01) ≥0.25` **and** covariance trace `<0.001`. PR alone never excludes a candidate.
3. Mark a candidate triple **eligible at search seed 1101** only if every population in that domain passes validity and is **not** probable near-collapse. Do not switch to the stopping checkpoint if its best checkpoint fails. Record noncollapsed anisotropy separately; it is not an automatic exclusion.
4. Within eligible triples for the **same objective**, rank by equal-weight mean of the populations' best validation objective losses: `L̄_d(h,s) = (1/|P_d|) Σ_{p∈P_d} L_val^best(h,s,p)`, with `P_synthetic={A,B}`, `P_real={TOTAL65,A,B}`. This is an **operational selection scalar within one objective**, not a composite of geometry/decoding/lag and not an independent-population statistical estimate; Real populations share channels/trials. Population-specific values must always be reported beside the mean. Do not rank `soft` against triplet or supervised loss.
5. Take the two lowest eligible triples to seeds 1201 and 1301. Final eligibility requires **all three HPO seeds × all populations** to pass validity and near-collapse checks. Rank eligible finalists by the equal-weight mean validation objective loss across the three seeds and populations. A finalist that fails a later seed is recorded as unstable and is not substituted post hoc with the stopping state.
6. **Exact tie-breaker:** if eligible finalists' aggregated losses differ by at most `1e-6` in objective-loss units, choose the one with fewer **total actual optimizer updates** over its three seeds/populations; if still tied, the smaller prespecified candidate index. The update tie-breaker applies only in a numerical tie, not as a hidden compute-quality score. If no eligible finalist remains, select **none** and record the cell unstable. Report population/seed losses and failures; never hide them behind the aggregate.

**FACT / boundary.** A low validation objective loss does not imply preservation of Z, downstream accessibility, or temporal fidelity. Geometry here is a coarse safety qualification, not the three-axis scientific evaluation. **Prohibited selection:** test geometry, decoding, lag, CKA/RSA, Procrustes-to-Z, synthetic ground-truth Z or held-out lag, full-sample descriptive outputs, and researcher inspection of test plots while selecting. Test may be read only under a later separately frozen final-evaluation protocol. The V2 test has already been viewed; using the identical split for a new confirmatory claim would carry analyst-level reuse even if runner code never opens it.

## 10. Collapse/failure handling

**RECOMMENDATION.** Every attempted fit gets an immutable trial record and consumes a budget slot. Retain partial logs/checkpoints when recoverable, without overwriting a prior run ID.

| Event | Status and selection effect | Required record |
|---|---|---|
| Crash during optimization or unavailable GPU | `FAILED_RUNTIME`; ineligible; counts against budget. | Error class/message, last update, device, log tail, candidate ID, source/config/parent hashes. Only a clearly external **pre-fit** infrastructure failure with zero updates may be retried once under the **same** candidate and a linked attempt ID; no retry with a new seed/hyperparameter for free. |
| NaN/Inf loss, gradient, weights, validation loss or embedding | `FAILED_NUMERICAL`; ineligible; counts. | First nonfinite stage/update, last finite checkpoint/history, implicated tensor/parameter summary if safe. No silent clipping or LR adjustment. |
| Missing/mismatched checkpoint, wrong `N×3`, row/metadata mismatch, bad unit norm | `FAILED_ARTIFACT`; ineligible; counts. | Invariant failure, exact file/hash/parent; stop downstream use. This is qualification, not a scientific collapse label. |
| Best-checkpoint validation embedding meets all three Section-9 near-collapse conditions | `INELIGIBLE_NEAR_COLLAPSE`; counts. | Loss, selected/stopping updates, covariance trace/eigenvalues, PR/ER, pair quantiles, near fraction; preserve plot and model. |
| Triplet best loss close to `ln(K_actual)` | `BASELINE_PROXIMITY_WARNING` **only**; not exclusion by itself. | Define warning prospectively as `|L_val^best − ln(K_actual)| ≤0.005` when the loss uses fixed K; log K, batch count, and validation estimate. Geometry/invariants decide eligibility; a log-baseline alone is not proof of collapse. |
| Best checkpoint at first validation check or stop at earliest patience-allowed step | `EARLY_SELECTION_WARNING` / `EARLY_STOP_WARNING`, **not** automatic failure. | Both update numbers, validation history, patience reference/stale checks. Synthetic first selected 400/earliest stop 2000; Real 150/1500. |

Report denominators for failure rates at three levels: attempted **fits**; candidate triples eligible across all populations on search seed; and finalists eligible across all three seeds/populations. Preserve each architecture × objective × domain's full attempted set, not just winners. Label a cell `UNSTABLE_PHASE2A` when, after its allowed 4–6 candidate triples and finalist re-evaluation, **zero** finalists pass all-population/all-seed qualification. The exact near-collapse rule and warning thresholds cannot be adjusted after seeing candidate results. An empirical failure may itself be a scientific observation; no cell is held indefinitely until it looks good.

## 11. Synthetic-vs-Real tuning strategy

| Strategy | What it would test / allow | Problem for the initial HPO |
|---|---|---|
| A. Tune Synthetic, transfer one triple to Real | A stringent *zero-real-validation* transfer question, valuable as a later separate test. | Input dimension/statistics and V2 collapse patterns differ; a transferred failure would not isolate architecture from domain mismatch. Synthetic Z must not enter HPO. |
| B. Tune every Synthetic and Real population independently | Best local validation adaptation. | Different A/B triples would confound same-process/cross-population and controlled-lag comparisons; more selection opportunities and provenance complexity. |
| C. **Domain-separated, population-shared hierarchical selection** | One triple per architecture × objective in Synthetic A/B, a separate triple in Real TOTAL65/A/B. Same candidate list and budget in both domains, separate validation ranking. | Does **not** test out-of-domain hyperparameter transfer; Real A/B/total are correlated and share a recording. |

**RECOMMENDATION: C.** It respects the material Synthetic/Real statistical difference while keeping A/B on the same optimization protocol. Use identical candidate proposals in both domains, but select independently from each domain's own validation losses, with all populations required to qualify. This permits the bounded claim “under a fixed data/split and population-shared validation-tuning budget, these representations behaved this way in Synthetic versus this Real recording.” It does **not** permit a claim of zero-shot Synthetic→Real transfer, biological latent-Z recovery in Real, independence of A/B observations, or universal method superiority. A later explicit transfer experiment would keep a Synthetic-selected triple frozen when applied to Real, not be folded into this HPO.

## 12. CNN-vs-Transformer fairness

**RECOMMENDATION.** Fair opportunity means: same three search dimensions and numerical bounds for an objective; identical ordered candidate triples at each index; identical 4 initial candidates and paired optional extension within objective × domain; same three HPO root seeds and population support; same domain-specific update/check/patience cap **for both architectures**; same qualification and ranking; and complete reporting of actual updates, time, memory, candidate failures, and near-collapse rates. It **does not** mean identical hyperparameter values must win, identical runtime, or that compute efficiency equals representation quality.

**FACT.** Current architectures have unequal parameter counts and wall time, and collapse rates differ. That is an outcome to report, not a reason to give the apparently weaker model more candidates. Within Synthetic, both architectures share the 400/800/3 schedule; within Real both share 150/750/5. The search therefore never compares CNN-default to Transformer-heavily-tuned or vice versa. Each fit reports `trainable_parameters`, actual optimizer updates, validation checks, wall time, seconds/100 updates, updates/s, windows/s where meaningful, peak allocated/reserved GPU memory, device/PyTorch/CUDA, and later inference throughput if scientific embedding generation is authorized. Keep computational efficiency in a **separate** results column, never in the HPO geometry/task/temporal score.

## 13. Phase-2B trigger

**RECOMMENDATION.** Phase-2B is **not automatic**. Consider a separate PI-approved architecture-capacity study only if, after the full Phase-2A candidate cap and finalist seed check, a particular architecture × objective × domain cell is `UNSTABLE_PHASE2A` by Section 10 **while the matched other architecture/objective cell on the same populations has at least one all-seed eligible finalist**, and qualification software/data invariants pass. This is evidence of architecture-conditional instability under the *limited* Phase-2A search, **not proof** that width/depth caused it. First check recorded gradients, losses, validation geometry and provenance without test selection; do not silently widen Phase-2A. If both architectures fail, suspect objective/data/protocol interaction rather than reflexively tuning capacity. Low PR with broad distances, poor held-out Z/decoding/lag alone, or a single near-collapsed V2 seed is **not** a trigger.

If separately approved, eligible Phase-2B parameters are Transformer `d_model`, number of heads/layers and feed-forward width, or CNN hidden width/layers/kernel, each with a declared hypothesis and matched budget. Pooling/readout redesign, sampler semantics and target construction would be separate scientific changes, not automatic capacity HPO. No Phase-2B ranges or fits are authorized here.

## 14. Provenance schema

**RECOMMENDATION — one immutable record per fit plus a parent record per candidate triple.** Minimum fields:

| Level | Required fields |
|---|---|
| Study/protocol | Frozen Phase-2A approval/version/hash; source **snapshot and SHA-256 of every relevant file including untracked/dirty runners**; Git HEAD and dirty diff/status separately; objective semantics and validation/anti-collapse rule version; ordered **sealed candidate list and its SHA-256**, Sobol algorithm/scramble/objective-specific seeds/SciPy version and numerical ranges; fixed schedule and budget rule. |
| Data parents | Raw source/preprocessing version and SHA; synthetic generated realization SHA plus generator seed; Stage-2 windows hash/valid-mask/window metadata; trial split IDs, seed and hash; Real exact 65-channel list or fixed A/B indices, channel-partition seed/hash; imposed shift `0` for HPO; parent artifact IDs and paths. |
| Candidate/fit identity | Unique study/candidate/attempt/run ID; domain, population, branch, architecture, objective; active LR/weight decay/temperature and all frozen resolved config fields plus canonical config SHA; training seed and its Synthetic A/B offset; data/generator seed, split seed, channel seed; sampler/loader RNG seed and coupling or separate state; fixed validation order/triplet-pair identities/hash. |
| Training environment | Python/PyTorch/CUDA/cuDNN versions, GPU model, device, deterministic flags, optimizer implementation, batch/effective candidate counts, number of trainable parameters; start/end timestamps; optimizer updates, validation checks, train/validation history, stopping reason, best raw-validation update/loss, meaningful-reference trajectory, stopping update, total time and GPU memory. |
| Artifact/status | Best and stopping checkpoint paths/SHA/parent IDs; validation-only unit-embedding artifact or ephemeral diagnostic source hash (no test); `N×3`/cardinality/norm checks; pair-sampling seed and full anti-collapse statistics; eligible/ineligible reason and warnings; crash/nonfinite details, retry link if allowed; downstream artifacts must name parent checkpoint/config/split hashes. |

**FACT.** Historical real provenance already has many such fields, but the saved `real_monkey.py` and `real_monkey_validated.py` SHA-256s do not match current files; Git HEAD alone is insufficient (lockdown Section 11). The V2 checkpoint payloads do not prove restorable optimizer/RNG state or bitwise GPU replay. **RECOMMENDATION.** A new runner must record source content and RNG provenance honestly; it must not claim bitwise reproducibility unless separately tested. Trial IDs must never overwrite V2 or another candidate attempt. Resume/invalidation must be content-hash dependent and preserve independent candidate/population/seed branches.

## 15. Estimated total number of fits / compute implications

**FACT.** Read-only medians of existing V2 held-out `training_seconds` in the compute sidecars indexed by the diagnostic CSV are: Synthetic CNN **88 s** (24 fits), Synthetic Transformer **101 s** (24); Real A/B/TOTAL65 CNN approximately **67–74 s** (12 each), Transformer **75–78 s** (12 each). These are historical medians at V2 hyperparameters, not a measured runtime for the proposed V3 candidate space. An existing Synthetic A Transformer-time fit took **1554 s**, far above the median; scheduling must tolerate outliers. Source: [EXISTING_EMBEDDING_DIAGNOSTICS.csv](../pre_tuning_2026-09-26/EXISTING_EMBEDDING_DIAGNOSTICS.csv) `compute_artifact` pointers to per-fit `compute.json` files; e.g. `outputs/runs/v2_corrected_2026-09-23_seed123_real_total_65/stage03_models/held_out/transformer_behavior_contrastive_blocks/compute.json` records 94.9 s/1500 updates on an RTX 3080.

**INFERENCE / planning estimate.** Applying those medians to the **160 initial fits** gives approximately 3.5–4 GPU-hours if run serially, **plus** ~3.5–4 GPU-hours for the maximum 160 finalist-seed fits, and up to ~2 GPU-hours for the 80 optional extension fits: roughly **7–10 serial GPU-hours at historical medians**, not a wall-clock guarantee. The wider LR/triplet-temperature intervals do **not** change candidate or maximum fit counts; they may change convergence, actual early-stopping updates, numerical-failure rates, and therefore runtime in an **unknown direction**. Allow materially more for 4000-update candidates, diagnostic embedding passes, disk I/O, altered LR/temperature, and outliers; a conservative reservation remains **one to two GPU-days**, then revise from actual first-stage throughput without changing trial count or selection rules. Formula: `initial 4 candidates × 4 objectives × [2 Synthetic populations × (88+101) s + 3 Real populations × (~70+~75) s] ≈ 3.6 h`; finalist stage has the same worst-case fit count; full paired extension adds half that. Early stopping may save updates but **is not an extra pruning policy** and can also select an early checkpoint while continuing to the patience stop. Record actual cost separately from scientific representation quality.

No full-sample refits, Stage-5 tests, R10 transformations, independent dataset realizations, alternative splits/partitions, or Phase-2B fits are included in these counts. They would require separate approval and compute estimates. Existing V2 artifacts may inform data-parent hashes and resource planning but are not silently counted as new HPO trials or overwritten.

## 16. PI-approved freeze decisions

The PI approved all decisions formerly listed here, with the final amendment replacing pure random candidate draws by the sealed Section-4 Sobol design. They are frozen for Phase-2A and do **not** authorize fits:

1. **Domain-shared population policy:** one Synthetic A/B triple and one Real TOTAL65/A/B triple per architecture × objective. Population-specific tuning or zero-shot transfer is a different scientific question.
2. **Numerical/search protocol:** the Section-4 ranges and sealed candidate 0–5 list, 4→6 paired extension trigger, HPO seeds 1101/1201/1301, strict all-population/all-seed eligibility, finalist ranking and tie-break rules. These are new prospective settings, **not** retroactive V2 changes.
3. **Geometry qualification:** the lockdown's validation-only thresholds, cohort/pair-sampling convention, and descriptive-only treatment of severe anisotropy without near-collapse.
4. **Early stopping:** retain Synthetic versus Real domain-specific schedules and the separation of raw best-checkpoint selection from meaningful-improvement patience. A harmonized schedule would require a separately approved protocol and baseline.
5. **Provenance and later confirmation:** preserve a source snapshot including untracked runners and plan independent confirmation. The V2 held-out test has already been viewed; exact historical real-run source replay is not established.

## 17. GO / NO-GO for implementing the HPO runner

**GO to implementing a dry-run-capable, provenance-checked HPO runner under this now-frozen protocol only.** Implementation must verify the sealed candidate list/hash, validation/test firewall, resume/invalidation, trial accounting, and qualification logic without training or touching V2. **NO-GO to training, Optuna/TPE/pruning launch, V3 runs, test/scientific-outcome inspection for selection, architecture/objective/sampler change, or any V2 overwrite under this prompt. STOP.**
