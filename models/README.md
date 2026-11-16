# Model weights

Model binaries are not included in this source repository. The recorded demo runs without them.

For live inference, provide these compatible model files here:

- pole_best6.pt
- channel_12class_v2.pt
- insulator_new.pt
- shed_model.pt
- cable_best.pt

Expected checkpoint hashes and saved validation scores are in `static/model_evaluation.json`. Use the full runtime dependencies and enable live inference only after installing the matching weights. `scripts/record_samples.py` and `scripts/export_checkpoint_metrics.py` require them. The checkpoint-integrity test in `tests/test_demo_evidence.py` also requires these local files.
