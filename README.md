# torch-native-dropout-train-none-guard

Guards a real, still-open PyTorch defect: `torch.native_dropout(input, p,
train)` accepts `train=None` (the operator's C++ schema has no default,
and the Python docs never describe what `None` means), and every
execution path currently disagrees about what it does.

Upstream: [pytorch/pytorch#197846](https://github.com/pytorch/pytorch/issues/197846)
(open). Fix PR: [#197854](https://github.com/pytorch/pytorch/pull/197854)
(open, not yet merged as of this tool's last check).

## The defect

| Path | `train=None` behavior |
|---|---|
| CPU eager | Applies dropout (acts like `train=True`) |
| MPS eager | Returns input unchanged (acts like `train=False`) |
| `torch.compile` (Inductor), any device | Skips dropout — **disagrees with eager on the same backend**, not just across devices |

No error, no warning. A caller who omits the `train` argument in a way
that resolves to `None` (easy to do — it is the schema's only way to
leave it out) silently gets three different numeric behaviors depending
on device and whether `torch.compile` is active.

### Independently reproduced on this host (torch 2.14.0)

- CPU eager `train=None`: dropout applied (mask not all-true).
- MPS eager `train=None`: dropout **not** applied (mask all-true) —
  byte-exact match to the upstream issue's own reported numbers.
- CPU eager vs. CPU `torch.compile(backend="inductor")` for
  `train=None`: **diverge** (eager applies dropout, compiled does not).
- Control: explicit `train=True` and `train=False` agree perfectly
  between eager and `torch.compile` in both cases — isolating the
  defect to the `None`-specific path only, not a general
  `native_dropout`/Inductor correctness problem.

## What this tool does

```bash
pip install torch-native-dropout-train-none-guard[torch]
torch-native-dropout-train-none-guard
```

`diagnose()` / the CLI re-runs the eager-vs-compiled comparison **live**,
on whatever torch build is currently installed, for `p` in `{0.0, 0.3,
0.5, 0.9, 1.0}` crossed with `train` in `{None, True, False}` — never
trusting a cached or previously-reported result. Each fixture runs in
its own subprocess so no `torch.compile`/dynamo cache state can leak
between fixtures.

`safe_native_dropout(fn)` wraps a 3-argument
`fn(input, p, train)` call so that `train=None` is coerced to
`train=True` **before** the call reaches `torch.compile`'s
decomposition — forcing the one behavior every backend currently
agrees on for an explicit value. This is a compatibility shim for an
unresolved upstream ambiguity, not a claim that `True` is "the correct"
meaning of `None` — the upstream issue itself does not define what
`None` should mean, only that today's disagreement is unintentional.

```python
import torch
from torch_native_dropout_train_none_guard import safe_native_dropout

def raw(input_tensor, p, train):
    return torch.native_dropout(input_tensor, p, train)

guarded = safe_native_dropout(raw)
guarded(x, 0.5, None)  # now consistently applies dropout, matching CPU eager
```

## Honest limitations

- **MPS-vs-CPU eager divergence is NOT covered by this project's CI.**
  GitHub Actions has no Apple Silicon / MPS runner. That part of the
  upstream issue was reproduced once, manually, on this contributor's
  real M-series hardware (see `scratch/_repro_197846_mps.py` in this
  fleet's development history) — it is documented here, not silently
  dropped, but it has no regression test and is not re-verified by CI.
  Only the CPU-eager-vs-CPU-compiled divergence is guarded and tested
  in this repository's CI (ubuntu-latest + macos-latest, both running
  CPU-only torch).
- This tool does not fix PyTorch. It is a userspace workaround. If/when
  upstream PR #197854 merges and ships in a released torch version, the
  underlying divergence may disappear and this guard becomes a
  (harmless) no-op — re-run `diagnose()` against your installed torch
  version to check before assuming the guard is still needed.
- No native-language (Chinese/Japanese) community discussion of this
  specific `train=None` semantics gap was found during this project's
  research phase; evidence is English-only (the upstream issue itself
  plus this tool's own independent reproduction).

## 中文说明 (Chinese summary)

`torch.native_dropout(input, p, train)` 的 `train` 参数在传入
`train=None` 时，CPU eager 模式会应用 dropout，MPS eager 模式则不会
（等同于 `train=False`），而 `torch.compile`（Inductor）在**同一个
CPU 后端**上也不会应用 dropout —— 三者互不一致，且没有任何报错或警告。
本工具在本机（torch 2.14.0）独立复现了这一行为差异，并提供
`safe_native_dropout()`，在调用 `torch.compile` 之前将 `train=None`
强制转换为 `train=True`，以恢复与 CPU eager 一致的行为。**注意**：
MPS 与 CPU 之间的差异未被本项目的 CI 覆盖（GitHub Actions
没有 MPS 运行环境），仅在开发过程中于真实 Apple Silicon 硬件上手动复现
并如实记录，未纳入自动回归测试。

## License

MIT
