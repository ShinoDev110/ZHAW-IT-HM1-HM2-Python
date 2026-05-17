# AGENTS.md

## Repository purpose
This repository contains standalone Python scripts for exercises derived from HM1 and HM2 lecture material.
The scripts live under two course folders, `HM1/` and `HM2/`, and are educational support material for students.
They are meant to be readable, easy to adapt, and easy to combine into larger files if needed.
Correctness is not guaranteed, and the scripts must not be treated as authoritative exam solutions.

## General behavior for AI agents
- Preserve the existing style and structure of the repository.
- Do not rewrite scripts unnecessarily.
- Do not introduce complex abstractions, frameworks, or extra architecture.
- Prefer clarity for students over clever, compact, or highly generic code.
- Keep scripts self-contained and easy to inspect.
- Match the current tone and level of explanation in the repository.
- When possible, keep changes minimal and local.

## Python generation rules
Only generate or modify Python code when the user explicitly asks for Python code, for example by saying “write a script”, “in Python”, “code this”, “solve with Python”, or similar.
For all other math or engineering problems, solve them manually with step-by-step reasoning and clear mathematical notation.

## Mandatory script structure
This structure must never be broken.

Every generated or modified Python script must follow exactly this structure:

1. Topic header at the very top (with TOPIC, DESCRIPTION, USE WHEN, EXAMPLE labels)
2. Import statements directly below the topic header
3. Part 1 — Inputs
4. Part 2 — Method selection (always present; if there is only one method, fill with a `# Only one method here.` comment)
5. Part 3 — Implementation
6. Part 4 — Call

All four parts must be separated using clear comment-header dividers, for example:

```python
# ============================================================
# TOPIC: Partial differentiation — Wave equation 3D wireframe plot
# DESCRIPTION:
# One- or two-line summary of what the script does.
# USE WHEN:
# One- or two-line description of the kind of problem this solves.
# EXAMPLE:
# Concrete Aufgabe / problem instance that this script applies to.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — Inputs
# ============================================================

# ============================================================
# PART 2 — Method selection
# ============================================================

# ============================================================
# PART 3 — Implementation
# ============================================================

# ============================================================
# PART 4 — Call
# ============================================================
```

### Topic header
- Place a single comment block at the very top of the file or script block.
- It must contain four labeled sections: `TOPIC:`, `DESCRIPTION:`, `USE WHEN:`, `EXAMPLE:`.
- `TOPIC` states the math or engineering topic and the specific task. Example: `TOPIC: Partial differentiation — Wave equation 3D wireframe plot`.
- `DESCRIPTION` is a 1–2 line summary of what the script does.
- `USE WHEN` is a 1–2 line description of the kind of problem this script solves.
- `EXAMPLE` is one concrete Aufgabe / problem instance to which this script applies.
- This helps students quickly identify the script when multiple files or blocks are combined.

### Imports
- Place all import statements directly below the topic header.
- Do not scatter imports throughout the file.
- Keep imports obvious and minimal.

### Part 1 — Inputs
- Define all problem-specific variables at the top.
- Typical inputs include functions, numeric values, intervals, matrices, vectors, boundary conditions, constants, and plotting settings.
- These should be the only values a student needs to change to adapt the script to a similar problem.
- Every input must have a short comment explaining what it represents when the meaning is not obvious.

### Part 2 — Method selection
- This section is ALWAYS present, even when there is only one method.
- If a problem can be solved in more than one way, expose a selector variable such as `method = "x"` that controls which path the implementation takes, and list the available options in a comment above the selector.
- If there is genuinely only one method, fill the section with a `# Only one method here.` comment (and optionally a short note describing the chosen method). Do NOT skip the section.

### Part 3 — Implementation
- Use a single function named after what it does and its topic, for example `plot_wave_wireframe` or `solve_linear_system_gauss`.
- The function must read the inputs from Part 1.
- If a selector exists, the function must branch according to Part 2.
- The function must perform the computation and print, return, or plot the result at the end.
- The function must respect all parameters defined above and must not require students to edit values inside the function body.
- Avoid burying the exercise setup inside the implementation function.

### Part 4 — Call
- At the bottom, include a single line or minimal block that calls the Part 3 function using the variables from Parts 1 and 2.
- Keep the call separate so students can paste multiple script blocks into one file and call them independently.

## Style expectations
- Variable naming conventions are less important than preserving the mandatory four-part structure.
- Prefer beginner-friendly, explicit code.
- Avoid unnecessary classes unless the task clearly requires them.
- Use comments where they help students understand the math or the implementation.
- Favor direct mathematical expressions and readable control flow.
- Avoid overly compact one-liners when they reduce clarity.

## Script maintenance rules
- When modifying an existing script, preserve the topic header, imports section, input section, method selector section if present, implementation function, and final call section.
- Keep existing scripts recognizable and easy to compare with lecture material.
- If a script is missing the expected structure, refactor it into the required structure only when explicitly asked to clean up or standardize scripts.
- Do not move student-editable inputs into the middle of a function body.
- Do not remove the final call unless the user explicitly requests that behavior.

## Educational disclaimer
These scripts are for educational purposes only.
They may contain mistakes or simplifications.
They should not be treated as guaranteed correct exam solutions.

## Repository-specific observations
Based on the scripts currently in this repository:
- All scripts live under two course folders: `HM1/` (Höhere Mathematik 1) and `HM2/` (Höhere Mathematik 2). Both follow identical script-structure conventions.
- Inside each course folder, scripts are grouped by chapter into snake_case English subfolders, often with topic-level subfolders below that.
- `HM1/` chapter folders: `machine_numbers/`, `functions/` (`basic_operations/`, `root_finding/`, `visualization/`), `matrices/` (`basic_operations/`, `decompositions/`, `linear_systems/`, `iterative_methods/`, `error_estimation/`), `interpolation/`, `eigenvalue_problems/` (`basic_operations/`, `spectral_properties/`, `diagonalization/`, `iterative_methods/`), and `complex_dynamics/`.
- `HM2/` chapter folders: `nonlinear_systems/` (`newton_methods/`, `linearization/`, `visualization/`), `interpolation_and_least_squares/` (`polynomial_interpolation/`, `spline_interpolation/`, `linear_least_squares/`, `nonlinear_least_squares/`), `numerical_integration/` (`newton_cotes/`, `gauss_quadrature/`, `romberg_extrapolation/`, `error_and_comparison/`), and `ordinary_differential_equations/` (`direction_fields/`, `single_step_methods/`, `higher_order_systems/`, `applications/`).
- Filenames follow `verb_object_method.py` (e.g. `solve_linear_system_with_jacobi.py`, `interpolate_value_with_lagrange.py`, `find_root_with_bisection.py`).
- Scripts are self-contained, typically end with a direct function call, use clear topic headers with comment-separated sections, often labeled in English while some titles and outputs are in German.
- No classes or package structure are used; the code is organized as single-file exercises with small helper functions.
- See [README.md](README.md) for the complete per-file index of both course folders.

## Repository organization guidance
- New scripts belong under the appropriate course folder (`HM1/` or `HM2/`) inside an existing chapter subfolder if the topic already exists.
- Chapter folder names are snake_case English (e.g. `machine_numbers/`, `nonlinear_systems/`). Subtopic folders follow the same casing.
- If a new chapter is needed, add it as a new top-level subfolder of the appropriate course folder; never nest a new chapter inside an existing one.
- Filenames must follow `verb_object_method.py` and make the script's purpose recognizable at a glance.
- Update [README.md](README.md) when adding a new chapter or script so the index stays in sync.

## Practical guidance for future agents
- Treat each script as a teaching example first and a software module second.
- Keep the code easy to paste, combine, and adapt for similar exercise variants.
- If a change risks obscuring the math, prefer a simpler implementation.
- Preserve the current educational format even when improving correctness or readability.

