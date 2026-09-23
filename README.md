# torch-native-dropout-train-none-guard

This project has been consolidated into [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards).

Use the umbrella package instead:

```bash
python -m pip install torch-correctness-guards[torch]
torch-guard run native-dropout-train-none
```

Python API:

```python
from torch_correctness_guards import safe_native_dropout, diagnose_native_dropout_train_none
```

The original functionality is now maintained as:

- module: `torch_correctness_guards.guards.native_dropout_train_none`
- CLI: `torch-guard run native-dropout-train-none`
- exported API: `safe_native_dropout`, `diagnose_native_dropout_train_none`

This repository is archived as a read-only pointer to the maintained umbrella package.
