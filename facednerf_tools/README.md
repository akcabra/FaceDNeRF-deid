# Original FaceDNeRF and EG3D utilities

This directory contains command-line utilities inherited from the original
FaceDNeRF/EG3D repository. They are retained for reference and compatibility,
but they are not part of the de-identification or evaluation workflow used in
this project.

The active project entry point remains at the repository root (`run.py`), with
evaluation and data-preparation utilities under `scripts/`.

Run an inherited utility from the repository root as a module so that imports
of the shared EG3D packages resolve correctly. For example:

```bash
python -m facednerf_tools.gen_samples --help
```
