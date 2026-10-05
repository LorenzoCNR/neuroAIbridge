# Balanced-accuracy bootstrap decision

Use the class-stratified complete-trial bootstrap as the preferred sampling-uncertainty interval for Balanced Accuracy. Each observed test class contributes exactly its original number of held-out trials to each replicate; whole trials, not windows, are resampled. This matches balanced accuracy's equal-class-recall estimand and avoids undefined replicates with an absent class.

The interval is conditional on observed held-out class trial counts; it does not quantify uncertainty in class prevalence. The unstratified intervals in `gap_closure_v1` remain preserved as sensitivity results. The frozen probe is reproduced using its existing train/validation-selected settings and test predictions; probes and encoders are not refit per bootstrap replicate. Training-seed variability remains separate.
