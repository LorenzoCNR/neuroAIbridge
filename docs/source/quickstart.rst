Quickstart
==========

Open the primary documented experiment:

.. code-block:: text

   notebooks/experiment_05_shared_latent_staged.py

Run its cells in order in VS Code or Jupyter. The notebook performs the
complete controlled pipeline without hiding it behind one experiment-wide
function:

1. generate 200 trials of a three-dimensional circular task latent;
2. inspect ``Z`` and the task-state variables;
3. construct ``B`` and inspect the neural tuning mixture;
4. calculate ``u``, ``lambda``, and stochastic counts ``X``;
5. construct one centered 21-bin window per trial time;
6. split complete trials into train, validation, and test sets;
7. fit PCA directly;
8. inspect one batch-wise soft target matrix;
9. construct and train CNN1D and Transformer encoders with four objectives;
10. compare embeddings with the known latent and estimate the inter-subject lag;
11. save models, figures, metrics, and arrays.

Artifacts are written to:

.. code-block:: text

   outputs/output_2026-08-27_comparison_2000/
   |-- stage01_data/
   |-- stage03_models/
   |-- stage05_metrics/
   `-- stage06_figures/

The separate real-data branch is run from
``src/neurobridge/experiments/real_monkey.py``. Its current cached output is:

.. code-block:: text

   outputs/real_monkey_area2_active_staged_2026-09-22/
   |-- stage01_data/
   |-- stage02_windows/
   |-- stage03_models/
   |-- stage04_embeddings/
   |-- stage05_metrics/
   `-- stage06_figures/

It uses one Area-2 session (193 trials, 65 channels), and reports observed-task
decoding/regression, input CKA, robustness, and efficiency. It has no known
latent ``Z`` or biological lag target.

``outputs`` is intentionally ignored by Git because all artifacts can be
regenerated from the experiment configuration.

The equivalent non-interactive command is:

.. code-block:: console

   python notebooks/experiment_05_shared_latent_staged.py
