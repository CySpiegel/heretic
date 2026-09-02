# Intel XPU support audit (branch feat/intel-xpu)

Date: 2026-09-01. Multi-agent audit (5 lens finders, 1 refuter per findings list, completeness critic + refuter) run against the heretic codebase on a machine with two Intel Arc Pro B70 GPUs, torch 2.13.0+xpu. Adjudicated by the session lead; task IDs refer to the wave-1 implementation spec.

## Findings and verdicts

### `xpu-heterogeneous-gpu-warning-cuda-only` — Heterogeneous-GPU reproducibility warning only checks torch.cuda, never torch.xpu

- **Severity (finder):** major  ·  **File:** `src/heretic/utils.py:387`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted — T3 (deduplicated with xpu-mgpu-2; backend-agnostic variant chosen)

**Description.** In generate_reproduce_readme(), the check that warns about non-deterministic behavior from mixing non-identical GPUs under device_map='auto' is gated on `if torch.cuda.is_available():` only, with no corresponding branch for XPU (or MLU/SDAA/MUSA/NPU). On a two-XPU machine, this block never executes, so `heterogeneous_warning` stays empty regardless of whether the two XPU devices are identical or different models. Every other accelerator-aware function in this codebase (system.py's empty_cache and get_accelerator_info_dict, utils.py's own print_memory_usage a few dozen lines above at line 86) already has a matching `elif is_xpu_available(): ...` branch using torch.xpu.device_count()/get_device_name(i) -- this one code path was missed, making it device-parity-inconsistent with the rest of the file it lives in.

**Evidence.**

```
src/heretic/utils.py lines 384-400:
    heterogeneous_warning = ""

    if include_system_information:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 1:
                device_names = {torch.cuda.get_device_name(i) for i in range(count)}
                if len(device_names) > 1:
                    heterogeneous_warning = """
> [!WARNING]
> **Heterogeneous GPUs**
...
"""
No `elif is_xpu_available():` follows, unlike the pattern already used at utils.py:80-94 (print_memory_usage) and system.py:34-45 (empty_cache), both of which do have `elif is_xpu_available(): ...` branches. Confirmed live on this machine: `uv run --extra xpu python -c "from accelerate.utils import is_xpu_available; import torch; print(is_xpu_available(), torch.xpu.device_count(), [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])"` -> `True 2 ['Intel(R) Graphics [0xe223]', 'Intel(R) Graphics [0xe223]']`, i.e. torch.xpu.get_device_name is available and returns per-device names exactly like torch.cuda.get_device_name, so nothing blocks adding the same check for XPU.
```

**Proposed change.** Add an `elif is_xpu_available():` branch mirroring the CUDA one (is_xpu_available is already imported into utils.py from .system at line 38):

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 1:
            device_names = {torch.cuda.get_device_name(i) for i in range(count)}
            if len(device_names) > 1:
                heterogeneous_warning = """ ... """
    elif is_xpu_available():
        count = torch.xpu.device_count()
        if count > 1:
            device_names = {torch.xpu.get_device_name(i) for i in range(count)}
            if len(device_names) > 1:
                heterogeneous_warning = """ ... """

Better still, factor the duplicated body into a small helper that takes (device_count, get_device_name_fn) so CUDA/XPU/MLU/SDAA/MUSA share one implementation and future backends can't be missed the same way again; or reuse the already-generic `get_accelerator_info_dict()['devices']` list (computed a few lines below at line 405) by moving that call above this block and driving the heterogeneous check off `{d['name'] for d in devices}` instead of re-querying torch.cuda directly, which would fix this for every backend at once including the two-B70 case this branch targets.

**Refuter reasoning.** Read src/heretic/utils.py lines 376-434 directly: generate_reproduce_readme()'s heterogeneous_warning block (lines 386-400) is gated solely on `if torch.cuda.is_available():` with no elif for XPU (or any other backend). This is inconsistent with the exact same check pattern used a few dozen lines above in print_memory_usage (lines 80-94), which does have `elif is_xpu_available(): ...`, and with system.py's get_accelerator_info_dict (lines 240-330), which has full XPU/MLU/SDAA/MUSA/MPS branches each populating a 'name' field per device (e.g. line 273: `[{'name': torch.xpu.get_device_name(i)} for i in range(count)]`). is_xpu_available is confirmed importable and already imported into utils.py (line 38, sourced from .system which itself imports it from accelerate.utils). On a real two-XPU (B70) machine this means the reproduce README's heterogeneous-GPU reproducibility warning silently never fires no matter what XPU hardware is used, which is a genuine parity gap and plausibly misleading (a major-severity UX defect, not a crash). The proposed fix (add a mirroring `elif is_xpu_available(): count = torch.xpu.device_count(); ...` branch, or better, drive the check off the already-computed `get_accelerator_info_dict()['devices']` list which carries a 'name' key for every backend including XPU) is correct, minimal, idiomatic for this codebase, and verified feasible since accelerator_info's devices dicts already contain 'name' for XPU.

### `xpu-diag-1` — get_accelerator_info_dict() XPU branch reports no VRAM and 'Unknown' driver version even though torch exposes both directly

- **Severity (finder):** major  ·  **File:** `src/heretic/system.py:266`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted — T1 (deduplicated with xpu-mgpu-1 and xpu-accelerator-info-dict-missing-vram-api-driver)

**Description.** Unlike the CUDA/ROCm branch (lines 243-264), which populates per-device vram_gb via torch.cuda.mem_get_info and api_version via torch.version.cuda/hip, the XPU branch (lines 266-274) hardcodes api_name/api_version to None, never reports any vram_gb per device, and gets driver_version exclusively by shelling out to the external 'xpu-smi discovery' binary (get_xpu_driver_version(), line 105). On the target machine (2x Arc Pro B70, no xpu-smi installed) this makes the startup banner both incomplete and actively misleading: it omits the '(NN.NN GB total VRAM)' summary entirely (because get_accelerator_info() only adds that suffix when total_vram>0, system.py:340-342) and prints 'Driver Version: Unknown' even though the driver version is trivially available. All of this information -- name, driver_version, total_memory (bytes) -- is exposed for free by torch.xpu.get_device_properties(i), with no dependency on xpu-smi, clinfo, or any subprocess call.

**Evidence.**

```
Live probe on the target 2xB70 machine:
>>> torch.xpu.get_device_properties(0)
_XpuDeviceProperties(name='Intel(R) Graphics [0xe223]', platform_name='Intel(R) oneAPI Unified Runtime over Level-Zero V2', ..., driver_version='1.14.37020', total_memory=31023MB, ...)
>>> p.total_memory  # bytes
32530182144  (=30.30 GB, matches the known 30.3GB/GPU spec)

Actual heretic output captured by running get_accelerator_info() on this machine:
  Detected 2 XPU device(s)
  Driver Version: Unknown
  * XPU 0: Intel(R) Graphics [0xe223]
  * XPU 1: Intel(R) Graphics [0xe223]
(no VRAM figure anywhere, despite 60+ GB of VRAM being present and load-planning-relevant)

Current code (system.py:266-274):
    if is_xpu_available():
        count = torch.xpu.device_count()
        return {
            "type": "XPU",
            "api_name": None,
            "api_version": None,
            "driver_version": get_xpu_driver_version(),
            "devices": [{"name": torch.xpu.get_device_name(i)} for i in range(count)],
        }
```

**Proposed change.** Populate driver_version, api info, and vram_gb from torch.xpu.get_device_properties(i) instead of relying solely on the absent xpu-smi binary:

    if is_xpu_available():
        count = torch.xpu.device_count()  # ty:ignore[unresolved-attribute]
        props = [torch.xpu.get_device_properties(i) for i in range(count)]  # ty:ignore[unresolved-attribute]
        return {
            "type": "XPU",
            "api_name": "XPU Runtime Version",
            "api_version": torch.version.xpu,  # ty:ignore[unresolved-attribute]
            "driver_version": get_xpu_driver_version()
            or (props[0].driver_version if props else None),
            "devices": [
                {
                    "name": p.name,
                    "vram_gb": round(p.total_memory / (1024**3), 2),
                }
                for p in props
            ],
        }

This keeps xpu-smi as a first-choice source (in case it ever reports something more precise/updated) but falls back to the property torch already has, matching the shape (vram_gb key) the CUDA branch and the downstream formatters (get_accelerator_info(), generate_reproduce_readme() in utils.py) already expect. It also fixes a secondary bug: because api_name was hardcoded None, reproduce.py's check_environment() (lines 258-268) silently skips verifying the XPU runtime/oneAPI version on reproduction, since the 'if accelerators.get("api_name") and accelerators.get("api_version")' guard in get_accelerator_info() (system.py:345) and the analogous check in reproduce.py never fire for XPU today.

**Refuter reasoning.** Read system.py:266-274 verbatim as quoted; the XPU branch does hardcode api_name/api_version to None and get driver_version solely from the xpu-smi subprocess wrapper (get_xpu_driver_version, line 105-119), which returns None whenever the xpu-smi binary is missing (FileNotFoundError caught). Live probe on the actual 2xB70 machine confirms torch.xpu.get_device_properties(i) exposes name, driver_version ('1.14.37020') and total_memory (32530182144 bytes = 30.30 GB) with zero dependency on xpu-smi/clinfo. Confirmed downstream: get_accelerator_info() (system.py:340-353) sums d.get('vram_gb',0) and only appends the VRAM suffix when >0, and only prints the api_name/api_version line when both are truthy -- so today's None/None values silently suppress both. The secondary reproduce.py claim is mechanistically mis-stated (there is no literal 'if api_name and api_version' guard in check_environment at reproduce.py:258-284 -- only a type-match guard), but the functional consequence described is still real: verify() at reproduce.py:262-268 compares accelerators['api_version'] (always None today) against the saved JSON value (also always None, since the same bug affects the run that generated it), so this==original is trivially true and a real XPU/oneAPI runtime version drift between the original and reproduction run can never be flagged. The proposed fix (populate driver_version/api_version/vram_gb from torch.xpu.get_device_properties(i), keeping xpu-smi as first choice) directly fixes both the display bug and this reproduce.py side-effect, and matches the vram_gb dict shape the CUDA branch and downstream formatters already expect.

### `xpu-diag-2` — "No GPU or other accelerator detected" message gives no hint when Intel GPU hardware is present but torch has no XPU support (forgot --extra xpu)

- **Severity (finder):** major  ·  **File:** `src/heretic/system.py:332`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted — T2

**Description.** When get_accelerator_info_dict() falls through every backend check and returns {"type": None}, get_accelerator_info() (lines 327-336) prints only 'No GPU or other accelerator detected. Operations will be slow.' with no diagnosis. This is exactly the failure mode the project's own environment notes call out as the most common footgun on this hardware: running `uv run python ...` instead of `uv run --extra xpu python ...` silently reinstalls the CUDA torch build, which then reports torch.version.xpu is None and no CUDA device either, so heretic falls through to this generic message even though two Arc Pro B70s are physically present and enumerable at /sys/class/drm. Nothing in the code distinguishes 'no accelerator hardware at all' from 'accelerator hardware present but this torch build can't see it', so the user gets no actionable signal and may not realize their environment silently reverted to a broken CUDA install.

**Evidence.**

```
Confirmed on this machine that vendor detection works without any extra tooling:
$ for f in /sys/class/drm/renderD*/device/vendor; do echo "$f: $(cat $f)"; done
/sys/class/drm/renderD128/device/vendor: 0x8086
/sys/class/drm/renderD129/device/vendor: 0x8086
and torch.version.xpu is a documented, always-present attribute (None on a non-XPU build, '20260000' here). Current code:
    if info["type"] is None:
        suffix = " Operations will be slow." if include_warnings else ""
        return (
            f"[bold yellow]No GPU or other accelerator detected.{suffix}[/]\n".strip()
        )
No check anywhere in system.py inspects torch.version.xpu or /sys/class/drm/*/device/vendor.
```

**Proposed change.** Add a best-effort Linux-only detector and use it to append an actionable hint in the info["type"] is None branch of get_accelerator_info():

    def _intel_gpu_present_without_xpu_torch() -> bool:
        """True when an Intel GPU is visible in /sys but the installed torch has no XPU backend."""
        if getattr(torch.version, "xpu", None) is not None:
            return False
        if platform.system() != "Linux":
            return False
        try:
            import glob
            for vendor_path in glob.glob("/sys/class/drm/renderD*/device/vendor"):
                with open(vendor_path, encoding="utf-8") as f:
                    if f.read().strip().lower() == "0x8086":
                        return True
        except OSError:
            pass
        return False

    # in get_accelerator_info():
    if info["type"] is None:
        suffix = " Operations will be slow." if include_warnings else ""
        hint = ""
        if include_warnings and _intel_gpu_present_without_xpu_torch():
            hint = (
                "\n[yellow]An Intel GPU was detected on this system, but this PyTorch "
                "install has no XPU support. Did you forget the `xpu` extra? "
                "Try: [bold]uv run --extra xpu heretic[/].[/]"
            )
        return (
            f"[bold yellow]No GPU or other accelerator detected.{suffix}[/]{hint}\n".strip()
        )

**Refuter reasoning.** Read system.py:327-336 verbatim: the info['type'] is None branch prints only the generic 'No GPU or other accelerator detected' message with no distinction between 'no hardware' and 'hardware present but torch build lacks XPU support'. Confirmed platform and os are already imported in system.py (lines 7-8), so the proposed detector needs only a local 'import glob'. Confirmed on this machine that /sys/class/drm/renderD128/device/vendor and renderD129/device/vendor both read '0x8086', validating the detection mechanism works without xpu-smi/clinfo. Confirmed torch.version.xpu is a stable, always-defined attribute (present in __all__ of torch/version.py, default None on non-XPU builds) so getattr(torch.version, 'xpu', None) is a safe, low-risk check. Confirmed pyproject.toml's project.scripts defines 'heretic = heretic.main:main', so the suggested remediation message 'uv run --extra xpu heretic' is accurate CLI usage. The proposed function and integration point are sound and non-invasive.

### `xpu-diag-3` — Raw, unstyled 'Aten Op fallback from XPU to CPU' UserWarning bleeds into heretic's Rich-formatted console output

- **Severity (finder):** minor  ·  **File:** `src/heretic/main.py:307`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted with narrowing — T4: filter only warnings attributed to torch._lowrank so other fallbacks stay visible

**Description.** torch.svd_lowrank (used in model.py:599 during residual/direction analysis) internally calls torch.linalg.qr, which has no XPU kernel and silently falls back to CPU, emitting a one-time raw Python UserWarning via the default warnings.showwarning printer straight to stderr -- unstyled, with an internal PyTorch build file:line reference, breaking out of heretic's Rich console formatting mid-run. heretic already suppresses other libraries' warning noise it considers non-actionable (transformers.logging.set_verbosity_error(), lm_eval logger level, ExperimentalWarning filter at main.py:307), but has no equivalent filter for this expected, already-self-explanatory XPU fallback warning, so on every XPU run it will surface once as raw unstyled text interleaved with heretic's own progress/status output.

**Evidence.**

```
Reproduced verbatim on the target machine:
$ uv run --extra xpu python -c "import torch; x=torch.randn(8,8,device='xpu'); torch.svd_lowrank(x, q=4, niter=2)"
/home/spiegel/Projects/heretic/.venv/lib/python3.12/site-packages/torch/_lowrank.py:72: UserWarning: Aten Op fallback from XPU to CPU happends. This may have performance implications. If need debug the fallback ops please set environment variable `PYTORCH_DEBUG_XPU_FALLBACK=1`  (Triggered internally at /__w/pytorch/pytorch/build/xpu/ATen/RegisterXPU_0.cpp:51246.)
  Q = torch.linalg.qr(X).Q
main.py currently only filters ExperimentalWarning (line 307) and sets logger verbosity for transformers/lm_eval/optuna (lines 297-304); nothing filters this UserWarning.
```

**Proposed change.** In main.py's run(), alongside the existing warnings.filterwarnings("ignore", category=ExperimentalWarning) call, add a targeted filter (guarded to XPU runs, or unconditional since the message text is torch-fallback-specific and harmless to filter on other backends):

    # torch.linalg.qr (used by torch.svd_lowrank in the residual analyzer) has no XPU
    # kernel and falls back to CPU; this is expected and already self-documenting, so
    # suppress the raw unstyled warning instead of letting it interleave with our
    # Rich-formatted progress output.
    warnings.filterwarnings(
        "ignore",
        message=r".*Aten Op fallback from XPU to CPU.*",
        category=UserWarning,
    )

**Refuter reasoning.** Reproduced the exact warning verbatim on the target machine by calling torch.svd_lowrank on an XPU tensor. Confirmed model.py:599 calls torch.svd_lowrank(W, q=2*r+4, niter=6) inside the RowNormalization.FULL branch (model.py:580-599), and confirmed config.default.toml:106 sets row_normalization = "full" as the shipped default -- so this warning fires on essentially every default XPU run, not an edge case. Confirmed main.py:307 is exactly `warnings.filterwarnings("ignore", category=ExperimentalWarning)` with no filter for this UserWarning, and confirmed `warnings` and `ExperimentalWarning` are already imported in main.py, so the proposed one-line addition fits the existing pattern precisely. Python's default warning filter shows a given (message, category, module, lineno) combination once per process, matching the 'one-time' framing.

### `xpu-repro-requirements-txt-triton-xpu-unresolvable` — Generated reproduce/requirements.txt lists triton-xpu, which has no PyPI distribution, breaking the documented `pip install -r requirements.txt` reproduction step for XPU-generated models

- **Severity (finder):** blocker  ·  **File:** `src/heretic/utils.py:542`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted — T5

**Description.** When a model is abliterated on the XPU backend, `create_reproduce_folder` writes a `reproduce/requirements.txt` (via `generate_requirements_txt` -> `get_requirements_dict`, src/heretic/system.py:423-478) that includes `triton-xpu==<version>`, because `get_requirements_dict()` walks torch's transitive dependency graph starting from `packages_to_check = ["heretic-llm", "torch", "torchaudio", "torchvision"]` and torch+xpu depends on triton-xpu. `triton-xpu` is only published on the custom PyTorch XPU wheel index (`https://download.pytorch.org/whl/xpu`), never on PyPI -- confirmed both by the pyproject.toml comment at line 105 ('not published on PyPI') and by uv.lock, which sources it exclusively from `source = { registry = "https://download.pytorch.org/whl/xpu" }`. `[tool.uv.sources]` and `[[tool.uv.index]]` are uv-only concepts; plain `pip` has no knowledge of them. The generated reproduce/README.md's step 2 literally instructs `pip install -r requirements.txt` (utils.py:542) with no `--extra-index-url` and no uv alternative, so following the documented XPU reproduction procedure with plain pip fails outright with a 'No matching distribution found for triton-xpu' error before even reaching step 3 (the PyTorch-specific install command). This is precisely the machine/workflow this branch targets (2x Arc Pro B70, XPU wheel), so any XPU-generated model's reproduce/ folder currently ships an unusable requirements.txt for pip users.

**Evidence.**

```
Confirmed on this machine: `uv run --extra xpu python -c "from heretic.utils import generate_requirements_txt; print(generate_requirements_txt())"` includes the line `triton-xpu==3.7.2`. uv.lock: `name = "triton-xpu" / version = "3.7.2" / source = { registry = "https://download.pytorch.org/whl/xpu" }`. pyproject.toml:104-105: `# Transitive dependency of the XPU torch wheel that is not published on PyPI.\ntriton-xpu = [{ index = "pytorch-xpu", extra = "xpu" }]`. Generated README instruction at utils.py:542: `1. Install the packages listed in requirements.txt: pip install -r requirements.txt`.
```

**Proposed change.** In get_requirements_dict() (src/heretic/system.py), exclude packages that are known to be unavailable on PyPI (e.g. triton-xpu, or more generally any package whose only installed distribution source is a non-PyPI index) from requirements.txt, and instead fold their install command into the same hardware-specific step that already builds `pytorch_install_command` in generate_reproduce_readme (utils.py:485-492) -- e.g. append `--extra-index-url https://download.pytorch.org/whl/xpu` and also install `triton-xpu==<version>` there, or simply detect PyTorch's local version suffix ('+xpu') and switch reproduce/README.md's instructions to recommend `uv sync --extra xpu` for reproduction on non-PyPI accelerator backends instead of raw pip.

**Refuter reasoning.** Verified directly: get_requirements_dict() includes triton-xpu==3.7.2 in its output on this machine, and pyproject.toml/uv.lock confirm triton-xpu is sourced exclusively from the custom pytorch-xpu index (never PyPI). The generated reproduce/README.md's step 2 (utils.py ~line 542, 'pip install -r requirements.txt') has no --extra-index-url and precedes the PyTorch-specific step 3, so a literal pip-based reproduction attempt fails on the triton-xpu line before ever reaching the correct torch install command. The proposed fix (exclude non-PyPI-sourced transitive deps like triton-xpu from requirements.txt and fold their install into the hardware-specific step, or point users at 'uv sync --extra xpu') is appropriate and mirrors the existing precedent of excluding heretic-llm from requirements.txt under analogous non-standard-source conditions (system.py ~470-474). One caveat: 'blocker' severity is debatable since heretic itself still runs fine on XPU -- only the generated reproduce/ pip workflow is broken -- 'major' would arguably fit the stated severity rubric better, but this doesn't change confirmation of the defect or the proposal's appropriateness.

### `xpu-repro-requirements-strip-plus-suffix` — requirements.txt strips the '+xpu' local version segment from torch/torchvision, so `pip install -r requirements.txt` fetches the wrong (CUDA/CPU) wheel before the README's later step silently overwrites it

- **Severity (finder):** major  ·  **File:** `src/heretic/system.py:414`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted (partial) — T5: torch/torchvision stay in requirements.txt for reproduce.py comparisons; the PyTorch install step now pins torchvision/torchaudio with the matching +suffix

**Description.** get_package_version() (system.py:414-420) intentionally strips everything after '+' from installed package versions ('stripping local suffixes like +cu128'), and get_requirements_dict() applies this to every discovered package including torch/torchvision themselves (they are seeded directly into packages_to_check at line 429 and never excluded). On this XPU machine this produces `torch==2.13.0` / `torchvision==0.28.0` in reproduce/requirements.txt, with no indication these came from the XPU wheel index. Per the generated README (utils.py:542-543), a reproducer runs `pip install -r requirements.txt` first (installing plain PyPI torch/torchvision, i.e. the CUDA-bundled build, downloading multiple GB of unrelated nvidia-* wheels) and only afterward runs the separate `pytorch_install_command` (utils.py:485-492) that correctly reinstalls the '+xpu' build with `--index-url https://download.pytorch.org/whl/xpu`. The end state is correct only because step 3 clobbers step 2, but the process wastes bandwidth/time and gives no signal to the user that the origin accelerator was XPU at all when just skimming requirements.txt.

**Evidence.**

```
Probe output on this machine: `torch==2.13.0` / `torchvision==0.28.0` inside generate_requirements_txt() output (no '+xpu' suffix), while `torch.__version__` is confirmed to be `2.13.0+xpu`. get_package_version(): `return version_str.split("+")[0] if "+" in version_str else version_str`.
```

**Proposed change.** Exclude torch/torchvision (and any other accelerator-index-sourced packages) from get_requirements_dict()'s output, the same way heretic-llm is already conditionally excluded (system.py:470-474), since their install is already handled by the dedicated pytorch_install_command step; or keep them but use the un-stripped version string for these specific packages so the '+xpu' tag survives into requirements.txt and readers can see the origin backend at a glance.

**Refuter reasoning.** Verified directly: get_requirements_dict() output contains torch==2.13.0 and torchvision==0.28.0 (no '+xpu' suffix) while torch.__version__ is actually '2.13.0+xpu', because get_package_version() unconditionally strips everything after '+' and torch/torchvision are seeded into packages_to_check with no exclusion. This means step 2 of the generated README installs the wrong (PyPI default/CUDA-bundled) torch/torchvision build before step 3's pytorch_install_command correctly reinstalls the '+xpu' build, wasting bandwidth and time and giving no visible signal of the origin backend when skimming requirements.txt. The primary proposed fix -- excluding torch/torchvision from requirements.txt output, mirroring the existing conditional exclusion of heretic-llm at system.py:470-474 -- is sound and low-risk. The offered alternative (keep them but preserve the '+xpu' suffix in requirements.txt) is flawed on its own: pip has no route to the custom index from a bare requirements.txt line, so 'torch==2.13.0+xpu' with no --extra-index-url would simply fail to resolve rather than working, making that specific alternative worse than the current behavior. The primary proposal stands regardless.

### `xpu-accelerator-info-dict-missing-vram-api-driver` — get_accelerator_info_dict()'s XPU branch hardcodes api_name/api_version to None and gets driver_version only from the (commonly absent) xpu-smi CLI, even though torch.xpu.get_device_properties() exposes total_memory, driver_version, and the oneAPI runtime version directly -- so reproduce.json/README for XPU runs silently omit VRAM, API version, and (on machines without xpu-smi) driver version, unlike the CUDA/ROCm branch

- **Severity (finder):** major  ·  **File:** `src/heretic/system.py:266`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted — T1 (duplicate)

**Description.** The CUDA/ROCm branch (system.py:243-264) populates `api_name`/`api_version` from `torch.version.cuda`/`torch.version.hip`, and per-device `vram_gb` from `torch.cuda.mem_get_info(i)[1]`. The XPU branch (system.py:266-274) hardcodes `api_name: None, api_version: None`, never sets any `vram_gb` key on device dicts, and gets `driver_version` exclusively via `get_xpu_driver_version()` (system.py:105-119), which shells out to the external `xpu-smi discovery` binary and returns None if it isn't installed. On the target 2xArc-B70 machine (xpu-smi NOT installed per task setup), `torch.xpu.get_device_properties(i)` already returns `total_memory=32530182144` (~30.3GB) and `driver_version='1.14.37020'` directly from Level Zero with zero external dependencies, and `torch.version.xpu == '20260000'` is the direct oneAPI/SYCL compiler-version analog of `torch.version.cuda`. Because none of this is wired up, `generate_reproduce_readme()`'s Accelerators section (utils.py:405-432) for an XPU run renders with no VRAM figure (the `vram_suffix`/per-device `vram` computations are always empty since `device.get('vram_gb')` is never set), no API-version line at all (the `if accelerators.get('api_name') and accelerators.get('api_version')` guard at utils.py:416 is always false), and no Driver Version line on any machine lacking xpu-smi. The same gaps propagate into reproduce.json's `system.accelerators` block, and into check_environment()'s api_name/api_version reproducibility check (reproduce.py: verify() call using accelerators['api_name']/['api_version']), which is permanently a no-op for XPU since both sides are always None -- so reproduction on XPU never actually verifies oneAPI/compute-runtime compatibility, unlike CUDA which verifies CUDA Version.

**Evidence.**

```
Probe on this machine: `get_accelerator_info_dict()` returns `{'type': 'XPU', 'api_name': None, 'api_version': None, 'driver_version': None, 'devices': [{'name': 'Intel(R) Graphics [0xe223]'}, {'name': 'Intel(R) Graphics [0xe223]'}]}`. Separately: `torch.xpu.get_device_properties(0)` returns `_XpuDeviceProperties(..., driver_version='1.14.37020', total_memory=31023MB, ...)` and `torch.xpu.mem_get_info(0) == (32530182144, 32530182144)`, and `torch.version.xpu == '20260000'`, all obtainable without xpu-smi.
```

**Proposed change.** In the XPU branch of get_accelerator_info_dict(): set `"api_name": "oneAPI Version", "api_version": torch.version.xpu`; populate each device dict with `"vram_gb": round(torch.xpu.get_device_properties(i).total_memory / (1024**3), 2)` (or `torch.xpu.mem_get_info(i)[1]`); and change `driver_version` to read `torch.xpu.get_device_properties(0).driver_version` first, falling back to `get_xpu_driver_version()` (xpu-smi) only if that attribute is unavailable, so the reproduce README/json get complete, self-contained accelerator metadata on XPU exactly as they already do on CUDA/ROCm.

**Refuter reasoning.** Verified directly: get_accelerator_info_dict() on this machine returns api_name=None, api_version=None, driver_version=None (xpu-smi absent), and device dicts with no vram_gb key -- exactly as claimed -- while torch.xpu.get_device_properties(i) already exposes driver_version='1.14.37020' and total_memory=32530182144 with zero external dependencies, torch.xpu.mem_get_info(i) returns (32530182144, 32530182144), and torch.version.xpu == '20260000' with zero external dependencies. Also independently confirmed the downstream consequence claimed in the finding: reproduce.py's verify() call for api_name/api_version compares None != None -> never records a mismatch, i.e. it is a permanent no-op for XPU, exactly as described. The proposed fix (populate api_version from torch.version.xpu, vram_gb from get_device_properties()/mem_get_info(), and prefer get_device_properties().driver_version over the xpu-smi shellout) is directionally and mechanically correct and mirrors the existing CUDA/ROCm branch pattern one-for-one. Minor nitpick only: torch's own collect_env.py labels torch.version.xpu as 'XPU used to build PyTorch' rather than a strict 'oneAPI Version', so the exact api_name label in the proposal is a cosmetic quibble, not a correctness problem with the fix.

### `readme-no-intel-xpu-install-guidance` — README.md's Usage/install instructions contain zero mention of Intel GPU/XPU, the `xpu` extra, or that plain `pip install` cannot select the XPU torch build (only `uv sync --extra xpu` / `uv run --extra xpu` can, since pip does not honor pyproject.toml's [tool.uv.sources]/[[tool.uv.index]])

- **Severity (finder):** major  ·  **File:** `README.md:86`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted — T8

**Description.** The Usage section (README.md:84-128) says only 'Prepare a Python 3.10+ environment with PyTorch 2.2+ installed as appropriate for your hardware' and shows `pip install -U heretic-llm`, with no hardware-specific guidance anywhere in the file (confirmed: zero occurrences of 'xpu', 'intel', 'arc', 'battlemage', 'level zero', 'oneapi' in README.md). This is a real functional trap, not just a documentation nicety: pyproject.toml's `xpu` extra (lines 63-67) and its custom index/source redirection (`[[tool.uv.index]]` / `[tool.uv.sources]`, lines 96-105) are uv-specific mechanisms that plain pip does not understand. A user who runs `pip install -U 'heretic-llm[xpu]'` -- the natural action given the existing README's pip-first framing -- will NOT get the XPU torch wheel or triton-xpu at all (pip has no route to the pytorch-xpu index or the source override), and will either silently install a broken/irrelevant combination or fail outright. The only working install path for Intel GPU users is uv-based (`uv sync --extra xpu` then `uv run --extra xpu heretic`, or `uv run --extra xpu heretic ...` directly against the repo), and this is nowhere stated for a project whose branch's whole purpose is first-class Intel GPU support.

**Evidence.**

```
`grep -ni "xpu\|intel\|arc\b\|battlemage\|level.zero\|oneapi" README.md` returns nothing. pyproject.toml:56-58 comment: 'Enable XPU with `uv sync --extra xpu` or `uv run --extra xpu heretic`. It requires the Intel GPU compute runtime (Level Zero) to be installed on the system.' -- this instruction exists only as a pyproject.toml comment, never surfaced in user-facing README.md.
```

**Proposed change.** Add a short hardware note to README.md's Usage section analogous to the existing PyTorch-version [!IMPORTANT] callout, e.g.: 'On Intel Arc / Battlemage GPUs, install with `uv sync --extra xpu` (or `uv run --extra xpu heretic ...`); plain `pip install` cannot select the correct PyTorch XPU build because it relies on a custom package index that only uv understands. Requires the Level Zero compute runtime.' Mirror the existing RTX 3090 batch-size/timing example with an Arc-equivalent note if timing data becomes available.

**Refuter reasoning.** Verified directly: README.md contains zero real mentions of xpu/Intel-GPU/arc/battlemage/level-zero/oneapi (the only grep hits for 'intel' are the substring inside 'intelligence', not actual Intel references). The Usage section (README.md:84-92) only says 'Prepare a Python 3.10+ environment with PyTorch 2.2+ installed as appropriate for your hardware' and shows plain `pip install -U heretic-llm`. Since pyproject.toml's XPU routing lives entirely in uv-only constructs ([tool.uv.sources], [[tool.uv.index]], extras conflicts) that pip does not understand, a user following the README's pip-first framing with `pip install -U 'heretic-llm[xpu]'` will not get a working XPU torch build (plain pip has no route to the pytorch-xpu index, and would separately fail on triton-xpu as in the related finding). The proposed fix -- adding a short hardware note pointing to `uv sync --extra xpu` / `uv run --extra xpu heretic`, styled like the existing PyTorch-version IMPORTANT callout -- is a minimal, appropriate, low-risk documentation change.

### `xpu-mgpu-1` — get_accelerator_info_dict() reports no VRAM and 'Unknown' driver version for XPU, despite the data being trivially available via torch.xpu itself

- **Severity (finder):** major  ·  **File:** `src/heretic/system.py:266`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted — T1 (duplicate)

**Description.** The CUDA branch (lines 243-264) computes per-device VRAM via torch.cuda.mem_get_info(i)[1] and gets the driver version from nvidia-smi. The XPU branch (lines 266-274) does neither correctly: it never populates 'vram_gb' for any device, and it obtains 'driver_version' exclusively from get_xpu_driver_version() (system.py:105-119), which shells out to the external 'xpu-smi' binary. On the target machine xpu-smi is not installed, so this always returns None, and the whole 'api_name'/'api_version' pair is also hardcoded to None even though torch.version.xpu is populated. The result is that get_accelerator_info(), and the '## System / Accelerators' section that generate_reproduce_readme() builds from the same dict (utils.py:405-432), both silently show 'Driver Version: Unknown' and omit VRAM entirely for a two-GPU box, even though torch.xpu.get_device_properties(i).driver_version and torch.xpu.mem_get_info(i) return the exact same information CUDA gets, with no subprocess call needed at all. This directly undermines the purpose of the report (helping the user judge per-device VRAM for e.g. max_memory settings) and the reproduce-README's reproducibility guarantees.

**Evidence.**

```
Live run against the actual XPU hardware (no xpu-smi installed):
```
$ which xpu-smi ; echo exit:$?
exit:1
$ uv run --extra xpu python -c "from heretic.system import get_accelerator_info; print(get_accelerator_info())"
Detected 2 XPU device(s)
Driver Version: Unknown
* XPU 0: Intel(R) Graphics [0xe223]
* XPU 1: Intel(R) Graphics [0xe223]
```
But the data is directly available from torch itself (probe against both real devices):
```
>>> props = torch.xpu.get_device_properties(0)
>>> props.driver_version
'1.14.37020'
>>> torch.xpu.mem_get_info(0)
(32530182144, 32530182144)   # (free, total) bytes == 30.30 GB, matches the box's known 30.3GB cards
```
Current code (system.py:266-274):
```python
    if is_xpu_available():
        count = torch.xpu.device_count()
        return {
            "type": "XPU",
            "api_name": None,
            "api_version": None,
            "driver_version": get_xpu_driver_version(),
            "devices": [{"name": torch.xpu.get_device_name(i)} for i in range(count)],
        }
```
```

**Proposed change.** Populate driver_version and vram_gb directly from torch.xpu, mirroring the CUDA branch, and only fall back to the xpu-smi subprocess (or drop it) if the torch API is unavailable:
```python
    if is_xpu_available():
        count = torch.xpu.device_count()
        devices = []
        driver_version = None
        for i in range(count):
            props = torch.xpu.get_device_properties(i)
            driver_version = driver_version or getattr(props, "driver_version", None)
            total_vram = torch.xpu.mem_get_info(i)[1] / (1024**3)
            devices.append({"name": torch.xpu.get_device_name(i), "vram_gb": round(total_vram, 2)})
        return {
            "type": "XPU",
            "api_name": "oneAPI Version",
            "api_version": torch.version.xpu,
            "driver_version": driver_version or get_xpu_driver_version(),
            "devices": devices,
        }
```

**Refuter reasoning.** Read src/heretic/system.py:266-274 directly: the XPU branch hardcodes api_name/api_version to None and never populates vram_gb, unlike the CUDA branch at 243-264 which does both. get_xpu_driver_version() (105-119) shells out to the 'xpu-smi' binary, which is confirmed absent on this machine ('which xpu-smi' exits 1). Live probe of get_accelerator_info()/get_accelerator_info_dict() reproduces exactly the cited output: 'Driver Version: Unknown' and devices with no vram_gb key. A direct torch.xpu probe confirms the proposed alternative data source works: torch.xpu.get_device_properties(0).driver_version == '1.14.37020' and torch.xpu.mem_get_info(0) == (32530182144, 32530182144) (~30.30GB, matching the known 30.3GB cards), with torch.version.xpu == '20260000' also populated. The proposed fix (populate driver_version/vram_gb/api_version from torch.xpu.get_device_properties and torch.xpu.mem_get_info, mirroring the CUDA branch, falling back to get_xpu_driver_version() only if unavailable) is appropriate and directly mirrors the existing CUDA code pattern already in the same function.

### `xpu-mgpu-2` — Heterogeneous-GPU reproducibility warning in generate_reproduce_readme() only fires for CUDA, never for multi-XPU

- **Severity (finder):** major  ·  **File:** `src/heretic/utils.py:387`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** accepted — T3

**Description.** generate_reproduce_readme() warns the reader that reproducibility cannot be guaranteed when the model was generated on non-identical GPUs and device_map='auto' spread layers across them -- but the check is gated behind `if torch.cuda.is_available()` (line 387) and unconditionally inspects torch.cuda.device_count()/get_device_name(). On an XPU-only machine torch.cuda.is_available() is False, so this block is skipped entirely regardless of how many XPU devices are present or whether they differ. Since this branch's whole purpose is first-class support for a dual-GPU (2x Arc Pro B70) machine using the same device_map='auto' pipeline-parallel mechanism as CUDA, a user who abliterates on two non-identical Intel GPUs (e.g. a B70 + a different Arc/Data Center GPU) gets no warning at all, silently claiming reproducibility guarantees the code cannot back up. (On this specific two-identical-B70 test box the warning would correctly stay empty either way, but the code path itself has no XPU branch, so it can never fire for any heterogeneous-XPU user.)

**Evidence.**

```
utils.py:384-400:
```python
    heterogeneous_warning = ""

    if include_system_information:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 1:
                device_names = {torch.cuda.get_device_name(i) for i in range(count)}
                if len(device_names) > 1:
                    heterogeneous_warning = """
> [!WARNING]
> **Heterogeneous GPUs**
...
"""
```
No `elif is_xpu_available(): ...` branch exists anywhere in this function; get_accelerator_info_dict() (called a few lines later at utils.py:405) is accelerator-agnostic and already returns a normalized `devices` list with a `name` per XPU device, but it is not consulted for the heterogeneous check.
```

**Proposed change.** Derive the heterogeneous check from the same accelerator-agnostic accelerators dict already computed just below, instead of hardcoding torch.cuda:
```python
    heterogeneous_warning = ""

    if include_system_information:
        accelerators = get_accelerator_info_dict()
        devices = accelerators.get("devices", [])
        if len(devices) > 1:
            device_names = {d["name"] for d in devices}
            if len(device_names) > 1:
                heterogeneous_warning = """
> [!WARNING]
> **Heterogeneous GPUs**
...
"""
```
(then reuse `accelerators` for the block currently at utils.py:405 instead of calling get_accelerator_info_dict() a second time).

**Refuter reasoning.** Read src/heretic/utils.py:384-432 directly: the heterogeneous-GPU warning block at line 387 is gated behind `if torch.cuda.is_available():` and only ever inspects torch.cuda.device_count()/get_device_name(), exactly as cited. On this XPU-only box torch.cuda.is_available() is False, confirmed live, so the block is always skipped regardless of XPU device count or heterogeneity. Meanwhile get_accelerator_info_dict() is called separately a few lines later (line 405) and already returns an accelerator-agnostic 'devices' list with per-device 'name', which the heterogeneous check could reuse instead of hardcoding torch.cuda. The proposed fix (derive the check from get_accelerator_info_dict()'s devices list, computed once and reused for the section below instead of calling it twice) is correct, accelerator-agnostic, and eliminates a redundant duplicate call to get_accelerator_info_dict() that exists in the current code (called once implicitly if we added an elif, but here explicitly avoids double-computation) - a genuine minor improvement bundled with the fix.

### `xpu-lmeval-clear-cache` — lm_eval's HFLM batch-size auto-detection never frees XPU cache (torch.cuda.empty_cache() hardcoded)

- **Severity (finder):** minor  ·  **File:** `src/heretic/main.py:1372`  ·  **Refuter verdict:** modified  ·  **Adjudication:** rejected — accelerate.find_executable_batch_size already clears the XPU cache during detection; patching lm_eval internals for a leftover cleanup is not worth the fragility

**Description.** The interactive 'benchmark' REPL command constructs `HFLM(pretrained=model.model, tokenizer=model.tokenizer, batch_size="auto")` (main.py:1372-1376) and calls `lm_eval.simple_evaluate(model=hflm, tasks=[...])` (main.py:1396). Because batch_size is 'auto', lm-eval's `HFLM._detect_batch_size()` (installed at .venv/lib/python3.12/site-packages/lm_eval/models/huggingface.py:797-855) runs `accelerate.find_executable_batch_size`, probing exponentially larger batches with real forward passes and calling `clear_torch_cache()` after each detection pass (huggingface.py:852 and :855). That helper is defined in .venv/lib/python3.12/site-packages/lm_eval/models/utils_hf.py:59-61 as `gc.collect(); torch.cuda.empty_cache()` with no XPU (or any non-CUDA) branch. On this machine `torch.cuda.is_available()` is False, so `torch.cuda.empty_cache()` silently no-ops, meaning cached-but-unused XPU allocator blocks from failed/oversized probe batches are never released during batch-size detection. This is exactly the gap heretic's own `src/heretic/system.py:empty_cache()` was already written to close for the rest of the codebase (it has an explicit `elif is_xpu_available(): torch.xpu.empty_cache()` branch), but that fix does not reach lm-eval's internal cache-clearing calls, which are third-party code invoked only through HFLM. Effect: on the two-XPU Battlemage machine, every time the user picks 'benchmark' in the interactive menu (and again for each additional benchmark run in the same session, doubly if 'Benchmark both models' is chosen), the exponential batch-size probe can leave stale cached blocks fragmenting the XPU allocator's pool, which can make later, differently-shaped probe/eval allocations spuriously fail to find a large batch size (falling back to a smaller batch size, or in the worst case, batch_size=1) purely because of unreclaimed cache -- a degradation that does not occur on CUDA, where the same code path actually frees the cache each time.

**Evidence.**

```
lm_eval/models/utils_hf.py:59-61: `def clear_torch_cache() -> None:\n    gc.collect()\n    torch.cuda.empty_cache()` -- no xpu/mlu/npu/mps branch. Confirmed torch.cuda.empty_cache() is a silent no-op on this XPU-only build: `torch.cuda.is_available()` -> False, `torch.backends.cuda.is_built()` -> False, calling it raises no exception and has no effect (verified via a direct probe). Contrast with accelerate's own device-agnostic helper already present in this venv, accelerate/utils/memory.py:39-49 `clear_device_cache()` (`if is_xpu_available(): torch.xpu.empty_cache()`), and with heretic's own src/heretic/system.py:26-37 `empty_cache()`, which already special-cases XPU the same way. HFLM is constructed at src/heretic/main.py:1372-1376 with `batch_size="auto"`, which is what triggers `_detect_batch_size()` and its `clear_torch_cache()` calls (huggingface.py:797, 852, 855).
```

**Proposed change.** Before constructing `hflm` in main.py (around line 1372), monkeypatch lm-eval's cache clearer to a device-agnostic one, e.g.: `import lm_eval.models.utils_hf as lm_eval_utils_hf` then `lm_eval_utils_hf.clear_torch_cache = heretic_empty_cache` where `heretic_empty_cache` is `from .system import empty_cache`. heretic's `system.empty_cache()` already does `gc.collect()` plus the correct per-backend `empty_cache()`, including the XPU branch, so it's a drop-in replacement for lm_eval's CUDA-only version. Alternatively/additionally, call `system.empty_cache()` explicitly after each `get_results()` call in the benchmark loop (main.py ~1390-1405) so XPU VRAM is reliably reclaimed between benchmark tasks regardless of what lm-eval does internally.

**Refuter reasoning.** The core defect is real: lm_eval/models/utils_hf.py:59-61 `clear_torch_cache()` is hardcoded to `torch.cuda.empty_cache()` with no XPU branch, and this is confirmed a silent no-op on the XPU-only build (torch.cuda.is_available() is False, empty_cache() raises nothing, does nothing). HFLM is indeed constructed with batch_size='auto' at main.py:1372-1376, which triggers `_detect_batch_size()`. However, the finding's central mechanism claim is wrong and its severity/impact narrative is built on that wrong premise. `_detect_batch_size()` (huggingface.py:797-855) wraps its inner forward-probe function with accelerate's `@find_executable_batch_size` decorator (imported from accelerate.utils.memory, line 16). That decorator ALREADY calls `clear_device_cache(garbage_collection=True)` (accelerate/utils/memory.py:162) before the first attempt AND after every OOM-triggered batch-size reduction (memory.py:178), and `clear_device_cache` (memory.py:40-49) already has a correct XPU branch (`if is_xpu_available(): torch.xpu.empty_cache()`) plus recognizes XPU OOM messages via `should_reduce_batch_size` (memory.py:100-116, explicitly lists ' out of memory.' as covering 'CUDA, HIP, XPU'). So the actual exponential-search/OOM-retry loop that the finding describes as leaking cache on every failed probe is, in fact, already device-agnostic and correctly frees XPU cache on every reduction step -- this is NOT the gap. lm_eval's own `clear_torch_cache()` at huggingface.py:852/855 (only one of the two ever executes per call, not both as the evidence implies, since world_size is 1 in heretic's single-process interactive session) runs exactly ONCE, only AFTER the batch size search has already concluded successfully -- it is a final post-detection cleanup, not a per-probe cache clearer. So the true residual gap is much narrower than described: after `_detect_batch_size()` successfully settles on a batch size (which may be called more than once per benchmark, e.g. huggingface.py:1039, 1131, 1397), one final non-essential `gc.collect()+cache-clear` fails to release XPU cache. This does not cause the 'fragmentation → spuriously smaller batch size on later probes' failure mode claimed, since the actual OOM-recovery clearing during search is unaffected. The defect and proposed monkeypatch fix are still reasonable engineering (it is a real, if minor/nice-to-have, gap and the fix is a correct drop-in), so this is not a full refutation -- but the description overstates severity and misattributes the mechanism.

**Refuter correction.** Keep the monkeypatch idea (it's a correct, low-risk drop-in fix) but scope the justification correctly: this only matters for the single post-detection cleanup call in `_detect_batch_size` (huggingface.py:852/855), not for cache reclamation during the actual OOM-driven exponential search, which accelerate's `find_executable_batch_size`/`clear_device_cache` already handles correctly for XPU. Before constructing `hflm` in main.py (~line 1372), patch lm_eval's cache clearer: `import lm_eval.models.utils_hf as lm_eval_utils_hf; from .system import empty_cache as heretic_empty_cache; lm_eval_utils_hf.clear_torch_cache = heretic_empty_cache`. Downgrade severity to nice-to-have (or keep minor but note the impact is a small residual leftover-cache cleanup after each successful batch-size detection, not allocator fragmentation causing spurious smaller batch sizes during the search itself, since that part is already device-agnostic).

### `xpu-multigpu-residual-stack-cross-device-crash` — get_residuals() unconditionally torch.stacks per-layer hidden states without normalizing devices, crashing residual analysis whenever device_map="auto" actually splits a model across the two Battlemage GPUs

- **Severity (finder):** blocker  ·  **File:** `src/heretic/model.py:725`  ·  **Refuter verdict:** confirmed  ·  **Adjudication:** rejected — lead reproduced the opposite on hardware: with an explicit device map placing layers 0-1 on xpu:0 and layers 2-3/norm/lm_head on xpu:1, transformers 5.6.2 returned every hidden state on xpu:0 and heretic-style torch.stack succeeded; accelerate/transformers normalize captured outputs to the root device

**Description.** Model.get_residuals() calls model.generate(..., output_hidden_states=True) and then does `torch.stack([layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states], dim=1)` over the full tuple of per-layer hidden states, with no `.to(...)` device normalization. With transformers 5.6.2's `@capture_outputs` mechanism (src/heretic/... via transformers/utils/output_capturing.py), each decoder layer's hidden state is captured via a forward hook installed directly on that layer module (`install_output_capturing_hook`), so the captured tensor for layer i lives wherever layer i actually executed. When `device_map="auto"` genuinely splits the model across the two 30.3 GB Arc Pro B70 cards (the whole point of having two GPUs -- e.g. any model whose bf16 weights don't fit in a single 30 GB card), accelerate's `dispatch_model` only sets `io_same_device=True` on the hook for the *top-level* model (`io_same_device=(module_name == "")` in accelerate/hooks.py); per-layer hooks do not move outputs back to a common device. So hidden_states[i] legitimately ends up on xpu:0 for layers assigned to card 0 and on xpu:1 for layers assigned to card 1, and the unconditional torch.stack raises a device-mismatch RuntimeError. This happens before optuna, before LoRA abliteration, before any offload_outputs_to_cpu logic runs (that .cpu() call is at line 750, *after* the crashing stack at line 725) -- i.e. the very first residual-analysis step of the pipeline (main.py calls model.get_residuals_batched()/get_residuals_mean() right after prompts are loaded) fails outright for any model that needs both GPUs. Note that model.py already knows layers can live on different devices elsewhere -- `_apply_lora()` does `v = layer_residual_direction.to(module.weight.device)` at line 531 -- so this is an inconsistency, not a fundamental design gap.

**Evidence.**

```
src/heretic/model.py:725: `residuals = torch.stack([layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states], dim=1)` with no `.to()` beforehand. transformers/utils/output_capturing.py:98-115 installs a forward hook per decoder-layer module that appends that module's own output tensor. accelerate/hooks.py:648,672: `io_same_device=(module_name == "")` -- only the top-level model forward gets device normalization, not individual layers. Confirmed on this machine's actual XPU runtime: `torch.stack([torch.randn(2,3, device='xpu:0'), torch.randn(2,3, device='xpu:1')], dim=1)` raises `RuntimeError: Expected all tensors to be on the same device, but got tensors is on xpu:1, different from other tensors on xpu:0 (when checking argument in method wrapper_XPU_cat)`. `git diff master...feat/intel-xpu -- src/heretic/model.py` is empty, confirming this code path is untouched by the XPU branch and was never adapted for the two-GPU target hardware.
```

**Proposed change.** In get_residuals() (src/heretic/model.py:703-731), move every per-layer hidden-state slice to a single common device before stacking, e.g. `target_device = hidden_states[-1].device` (or `self.model.device` / the device of the final norm/lm_head, which is where generation's output already lives) and change the list comprehension to `[layer_hidden_states[:, -1, :].to(target_device) for layer_hidden_states in hidden_states]`. Equivalently, move each slice to CPU immediately if `settings.offload_outputs_to_cpu` is set, before stacking, since that already the eventual destination and avoids an extra XPU-to-XPU copy. Add a regression test (or at least a manual multi-device dtype test) that fakes a >1-device hidden_states tuple to catch this class of bug for any future accelerator backend.

**Refuter reasoning.** Verified end-to-end against the actual installed sources and reproduced live on this machine's two XPU devices. (1) src/heretic/model.py's get_residuals() (torch.stack at line 740 in the current tree, not 725 -- line numbers have drifted ~15 lines from whatever revision the finding was written against, but the code and surrounding structure match exactly) stacks the hidden_states tuple with no .to() normalization; the offload_outputs_to_cpu -> .cpu() call is indeed at line 764, after the crashing stack. (2) transformers/utils/output_capturing.py:98-115 confirms install_output_capturing_hook registers a plain module.register_forward_hook per decoder layer, and llama's modeling code confirms _can_record_outputs = {'hidden_states': LlamaDecoderLayer} with no manual all_hidden_states accumulation in forward -- capturing is entirely hook-based, one hook per decoder layer, capturing that layer's raw post-forward output. (3) accelerate/hooks.py:648/672 confirms io_same_device=(module_name == '') -- only the top-level module's AlignDevicesHook moves output back to the input device; per-layer hooks (as installed by dispatch_model for a genuinely split device_map) do not. add_hook_to_module (hooks.py:147-200) replaces module.forward directly, so this device (mis)placement is fully resolved before nn.Module.__call__ ever runs the transformers register_forward_hook-based capturing hook, meaning the captured tensor for a layer assigned to xpu:1 stays on xpu:1. (4) I built a toy 4-layer nn.Module, split it 2/2 across xpu:0 and xpu:1 with accelerate's real dispatch_model, attached a forward hook identical in spirit to output_capturing_hook, and ran it on this machine's actual XPU hardware: captured layer devices were exactly [xpu:0, xpu:0, xpu:1, xpu:1], and torch.stack on them raised 'RuntimeError: Expected all tensors to be on the same device, but got tensors is on xpu:1, different from other tensors on xpu:0 (when checking argument in method wrapper_XPU_cat)' -- the identical error text src/heretic/model.py's loading loop (lines ~163-172) already special-cases for device_map='auto' failures, corroborating this is a live, previously-encountered failure mode. (5) config.default.toml:28 confirms device_map = 'auto' is the shipped default, and README.md:122-123 documents that splitting across GPUs happens automatically via this default -- squarely the two-B70-GPU target scenario. (6) get_residuals_batched/get_residuals_mean are called unconditionally and early in main.py (lines 554-575), with no try/except, so any model whose bf16 weights exceed a single 30.3GB B70 card and gets split by device_map='auto' will crash on the very first residual computation. Root cause is backend-agnostic (would equally affect multi-GPU CUDA), but that does not put it outside scope: the branch's explicit goal is first-class support on this exact two-XPU machine, and this is a genuine, reachable, unhandled blocker for that hardware whenever a model requires both cards.

**Refuter correction.** Same fix as proposed, just with corrected line references for the current tree: in get_residuals() (src/heretic/model.py, function starting at line 718, torch.stack at line 740), normalize every per-layer hidden-state slice to a single device before stacking, e.g. `target_device = hidden_states[-1].device` (the final layer's output device, i.e. where generation's own output/logits already live) and change the comprehension to `[layer_hidden_states[:, -1, :].to(target_device) for layer_hidden_states in hidden_states]`. Since residuals are immediately upcast to float32 right after (line 749) and optionally moved to CPU when offload_outputs_to_cpu is set (line 764), an even simpler and more memory-friendly fix is to slice-and-move each layer's tensor to CPU inline (`.to('cpu', dtype=torch.float32)`) when offload_outputs_to_cpu is set, or to `.to(target_device)` otherwise, avoiding a large intermediate all-on-one-GPU stack for very large models. A short regression test asserting get_residuals()/the internal stack tolerates a hidden_states tuple whose entries live on different devices (fakeable via a tiny multi-device dummy model, as reproduced in this verification) would catch regressions for any future accelerator with per-layer device splits.

## Lens notes (areas checked and found clean)

Scope: audited every torch.cuda.*, torch.backends.*, PYTORCH_ALLOC_CONF/PYTORCH_CUDA_ALLOC_CONF handling, seeding (torch.manual_seed, transformers.set_seed, torch.cuda.manual_seed_all), empty_cache, device_map/.to(device) call, and torch.accelerator usage in src/heretic/*.py and src/heretic/scorers/*.py.

Confirmed CLEAN (not reported, with evidence):
- empty_cache() (system.py:26-47) and print_memory_usage() (utils.py:74-94): both already have correct elif is_xpu_available()/torch.xpu branches.
- Seeding: torch.manual_seed(...) at model.py:595 and transformers.set_seed(...) at main.py:270 both internally call torch.xpu.manual_seed_all(seed) when XPU is present -- verified by reading the installed torch/random.py (_manual_seed_impl explicitly does `import torch.xpu; if not torch.xpu._is_in_bad_fork(): torch.xpu.manual_seed_all(seed)`) and transformers/trainer_utils.py's set_seed (explicit `if is_torch_xpu_available(): torch.xpu.manual_seed_all(seed)`). The redundant torch.cuda.manual_seed_all(seed) call at model.py:598 is a documented no-op when CUDA is absent -- confirmed with `uv run --extra xpu python -c 'import torch; torch.cuda.manual_seed_all(42)'` (no exception, torch.cuda.is_available() is False). So the svd_lowrank reseed-before-randomized-op pattern is correctly covered for XPU via torch.manual_seed's device-wide seeding, exactly as the lens description anticipated -- not a bug.
- PYTORCH_ALLOC_CONF=expandable_segments:True (main.py:178-183): probed on real XPU hardware. torch.xpu.memory._snapshot()['allocator_settings'] shows {'PYTORCH_ALLOC_CONF': 'expandable_segments:True', 'expandable_segments': True} when the env var is set (both when set before and after `import torch`, matching main.py's actual import order where torch is imported at module scope but the env var is set later inside run()). Without the var it shows expandable_segments: False. So the option is genuinely parsed and honored by the XPU caching allocator's shared c10 AllocatorConfig, not silently dropped -- not a bug.
- No hardcoded "cuda"/".cuda()"/"cuda:0" device strings anywhere in src/heretic/.
- No torch.backends.cudnn/cuda-specific backend flags; the only torch.backends.* usage is mps.is_available() (already has an accelerator-agnostic fallback chain) and mkldnn.enabled (CPU backend info, printed unconditionally as debug info, not a device-selection branch).
- get_accelerator_info_dict's XPU branch missing VRAM/api_version and depending on xpu-smi (unavailable on this machine) for driver_version was explicitly called out in the task's "Environment facts" as already-known/tracked -- not re-reported per instructions.
- model.py's `.to(module.weight.device)` (line 531) and `.to(self.model.device)` (lines 657, 840) are fully generic/accelerator-agnostic and correctly support per-module devices under device_map='auto' sharding across the two XPUs.
- reproduce.py's check_environment()/verify() logic drives entirely off get_accelerator_info_dict()'s generic dict, not off torch.cuda directly -- no CUDA-specific branching there.
- No torch.accelerator.* API usage anywhere in src/heretic/ (codebase instead uses explicit is_xpu_available()/torch.cuda.is_available() branch chains everywhere else) -- a style choice, not a functional gap, since those branch chains do correctly include XPU except for the one reported finding.
- scorers/*.py: no device or cuda references at all; fully device-agnostic.
- config.default.toml and README.md: no cuda/xpu references in scope for this lens.

Environment used for probes: `uv run --extra xpu python ...` against the live 2x Arc Pro B70 XPU hardware, per the task's required invocation.

Lens: quantization, PEFT, evaluation, numerics. Extensive investigation found this locked stack (transformers 5.6.2 + bitsandbytes 0.50.1 + peft 0.19.1 + accelerate 1.13.0) to have genuinely solid, already-XPU-aware bnb 4-bit support: Bnb4BitHfQuantizer.update_device_map, validate_bnb_backend_availability, bnb.supported_torch_devices, bnb's xpu backend ops (dequantize_4bit/quantize_4bit/gemv_4bit, all bf16/fp16/fp32), Params4bit.to()/.xpu(), and peft's dispatch_bnb_4bit are all device-generic or already xpu-branched — verified several of these empirically with small probes on both xpu:0 and xpu:1 (quantize_4bit + dequantize_4bit round-trip with double-quant on xpu:1 succeeded with small reconstruction error). Areas explicitly checked and found clean (no findings): model.py's dequantize_4bit call and LoRA delta computation in abliterate(); get_merged_model's CPU-reload path for quantized models (device-agnostic, unaffected by XPU); RNG reseeding before torch.svd_lowrank (model.py:595-598) — initially suspected as an XPU gap since only torch.manual_seed + torch.cuda.manual_seed_all are called, but probing confirmed torch.manual_seed() in this torch 2.13 build already internally calls torch.xpu.manual_seed_all(), so XPU reproducibility of the FULL row-normalization SVD path is intact, not a bug; torch.svd_lowrank/torch.linalg.qr CPU fallback on XPU (already known/documented, produces correct results just with a perf-warning, not re-reported); torch.quantile and torch.linalg.vector_norm on xpu (both work correctly, probed); KL divergence scorer's F.kl_div on xpu (probed, works); HFLM's device inference when passed an already-loaded model object (uses model.device, not a string-path CUDA branch); accelerate's should_reduce_batch_size/find_executable_batch_size (already handles XPU OOM message pattern); accelerate's get_balanced_memory/infer_auto_device_map for multi-GPU device_map=\"auto\" splitting (already xpu-aware, uses torch.xpu.mem_get_info, probed working on both GPUs, 32530182144 bytes free confirmed on each). No blockers or crashes found in this lens; the one reported finding is a minor, well-scoped degraded-functionality issue confined to the optional interactive 'benchmark' command."

## Lead's own findings (from hardware runs)

- Gemma 4 multimodal models fail under `device_map="auto"` on any multi-accelerator system with `Expected all tensors to be on the same device` (transformers `modeling_gemma4.py:2282`); `device_map="sequential"` works → spec T6 (hint + test config).
- XPU output is deterministic run-to-run (identical safetensors hashes); `SHA256SUMS.xpu` reference hashes added for the tiny-model tests.
- bitsandbytes 0.49.x silently fell back to Triton kernels on this stack (native lib built for an older oneAPI); 0.50.x loads `libbitsandbytes_xpu2026.so` → dependency bumped.
- **Out-of-memory is not recoverable on XPU (blocker, fixed in wave 2).** When a kernel needs more memory than the card has, the `xe` driver resets the exec queue (`dmesg`: `exec queue reset detected`, `VM worker error: -12`) and the process's context is lost: the failing op raises `level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)` or `20 (UR_RESULT_ERROR_DEVICE_LOST)`, and every later operation, including `torch.xpu.empty_cache()`, fails with `DEVICE_LOST`. PyTorch's caching allocator never raises `torch.OutOfMemoryError` because Level Zero over-commits device memory. Observed with Qwen3.8-27B (bf16, split across both cards, 51 GB): batch 32 OK at 66 tokens/s, batch 64 → error 40, next generation → error 20. Heretic's batch-size benchmark relied on catching OOM, so wave 2 adds a predictive memory guard and an explanation when a device is lost. Measured on Qwen3.8-27B with `device_map="auto"`: weights 22.93 GB on card 0 and 27.17 GB on card 1 (of 30.3 GB each); peak activation memory above the weights grows linearly with batch size (card 1: 0.089 / 0.177 / 0.347 / 0.69 / 1.384 / 2.765 GB at batch 1 / 2 / 4 / 8 / 16 / 32), and at batch 32 the reserved pool on card 1 already exceeded physical memory (30.72 GB). A fractional headroom (10 %) therefore rejected even batch 2; the guard uses a fixed 1 GiB headroom against physical capacity (`XPU_MEMORY_HEADROOM`) and projects the next batch size as `baseline + 2 * (peak - baseline)`.
- **Gemma 4 on XPU (unresolved, torch bug).** After a single forward pass of `tiny-random/gemma-4e` on XPU, any boolean-mask indexing / `nonzero()` on a `(2, 36)` tensor fails with `RuntimeError: numel: integer multiplication overflow` (transformers `modeling_gemma4.py:2277`). Reproduced in an independent venv with torch 2.13.0+xpu; CPU is unaffected; not related to torch.compile (`disable_compile=True` does not help). The gemma-4e test therefore cannot produce XPU reference hashes on this stack; no `SHA256SUMS.xpu` was added for it.
- **Verification matrix on the two B70s (all end to end through trials and export):** tiny-model suite (mistral-3, minicpm5, qwen2.5, qwen3.5-moe) with identical hashes before/after the code changes; Qwen3.5-0.8B forced across both cards via `max_memory` (batch benchmark to 128); Qwen3.5-0.8B with `bnb_4bit`; Qwen3.8-27B bf16 across both cards with `max_batch_size = 32` (66 tokens/s, adapter exported).
