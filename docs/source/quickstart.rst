Quickstart
==========

Inspect the current frozen study
--------------------------------

The result artifacts are already versioned; reading them does not require
training a model. Start with the `results and limitations guide <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/06_EXPERIMENTS_AND_LIMITATIONS.md>`_
and then use the `Python pipeline map <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/07_PYTHON_PIPELINE_MAP.md>`_
to follow code and artifact dependencies. The compact metrics and provenance
are under ``outputs/final_thesis_v1/final_evaluation/``.

Run a simulator tutorial
------------------------

The staged shared-latent notebook is an executable, cell-by-cell reference
for the original Synthetic workflow:

.. code-block:: text

   notebooks/experiment_05_shared_latent_staged.py

Open it in VS Code or Jupyter and run cells in order. It demonstrates latent
generation, neural mapping, windowing, trial splitting, fitting, embedding,
and evaluation. It is not the final Phase-2A HPO/Real protocol and should not
be used to infer the final Real settings. For a smaller first example, use
``notebooks/experiment_01_circular_3d.ipynb``.

The final Real study is already represented by its frozen spec, fit records,
embeddings, metrics, and provenance; do not rerun its training merely to
inspect the results. Its selected window, seeds, populations, and limits are
summarized in the current-results guide.

Output and Git boundary
-----------------------

The repository versions selected compact reports, tables, provenance, and
figures under ``outputs/``. Large local neural arrays, checkpoints, caches,
interactive HTML, and designated high-volume replicate files may be excluded
by ``.gitignore``; ignored local files are not removed. See the root
``README.md`` and the pipeline map for the current publication boundary.
