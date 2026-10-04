NeuroBridge documentation
=========================

NeuroBridge is a research package for controlled neural time-series
simulation and representation learning. It generates an interpretable latent
task process, maps that process into neural population activity, emits sparse
spike counts, and evaluates whether an encoder recovers the known geometry.

The package currently provides circular and linear motor-task simulations,
localized place fields, centered temporal windows, PCA and temporal neural
encoders, structured contrastive objectives, and latent-recovery metrics.

.. note::

   NeuroBridge is an active research prototype. Reference results are useful
   for reproducibility and debugging, but are not yet multi-seed held-out
   benchmarks.

Start here
----------

* :doc:`installation` explains editable and documentation installations.
* :doc:`quickstart` runs the first controlled experiment.
* :doc:`concepts` defines the generative and learning objects.
* :doc:`experiments` presents the four baseline experiments.
* :doc:`demos` links each executable Jupyter notebook.
* :doc:`api` is generated from the package docstrings.

Canonical repository documents
------------------------------

The complete scientific documents are also readable directly on GitHub,
without building or entering this website:

* `Documentation index <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/README.md>`_
* `Generative model <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/01_GENERATIVE_MODEL.md>`_
* `Data and temporal windows <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/02_DATA_AND_WINDOWS.md>`_
* `Learning objectives <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/03_LEARNING_OBJECTIVES.md>`_
* `Encoders <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/04_ENCODERS.md>`_
* `Evaluation and multiple subjects <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/05_EVALUATION_AND_MULTISUBJECT.md>`_
* `Experiments and limitations <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/06_EXPERIMENTS_AND_LIMITATIONS.md>`_

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   quickstart
   concepts
   experiments
   demos
   api
   contributing
