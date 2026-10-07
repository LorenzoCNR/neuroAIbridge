NeuroBridge documentation
=========================

NeuroBridge is a research package for controlled neural time-series
simulation and representation learning, with separate frozen Synthetic and
Real-data evaluation branches. The simulator supplies a known task latent;
the single-session Area-2 recording does not have a known biological latent.

The package provides circular and linear motor-task simulations, localized
place fields, centered temporal windows, PCA and temporal neural encoders,
structured objectives, and staged downstream evaluation.

.. note::

   The repository includes frozen multi-seed Synthetic and Real study
   outputs. Their scope is still limited: one Real recording/session, three
   training seeds, and no known Real latent ground truth. See the current
   results guide for what each branch can support.

Start here
----------

* :doc:`installation` explains editable and documentation installations.
* :doc:`quickstart` shows how to inspect frozen results and run a simulator tutorial.
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
* `Python pipeline map <https://github.com/LorenzoCNR/neuroAIbridge/blob/main/docs/07_PYTHON_PIPELINE_MAP.md>`_

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
