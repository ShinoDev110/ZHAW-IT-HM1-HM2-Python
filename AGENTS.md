# AGENTS.md

## Repository purpose
This repository contains standalone Python scripts for exercises derived from HM2 lecture material.
The scripts are educational support material for students.
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

1. Topic header at the very top
2. Import statements directly below the topic header
3. Part 1 — Inputs
4. Part 2 — Method selection, if applicable
5. Part 3 — Implementation
6. Part 4 — Call

All four parts must be separated using clear comment-header dividers, for example:

```python
# ============================================================
# TOPIC: Partial differentiation — Wave equation 3D wireframe plot
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
- State the math or engineering topic and the specific task clearly.
- Example: `TOPIC: Partial differentiation — Wave equation 3D wireframe plot`
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
- If a problem can be solved in more than one way, expose a selector variable such as `method = "x"` that controls which path the implementation takes.
- Add a comment above the selector listing the available options.
- If there is genuinely only one method, explicitly state that in a comment and skip the selector variable.

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
- The repository contains 12 standalone `.py` scripts in the root folder.
- The scripts focus on HM2-style math and engineering exercises, especially nonlinear systems, Newton methods, Jacobian matrices, linearization, and visualization.
- Newton-related scripts include standard Newton’s method, simplified Newton, damped Newton, comparing variants, listing all solutions from multiple start vectors, and reporting norms per iteration.
- Visualization scripts use `numpy`, `sympy`, and `matplotlib` to plot implicit curves, nonlinear system zero contours, 3D wireframes, surfaces, and contour plots.
- The scripts are self-contained and typically end with a direct function call.
- The current scripts use clear topic headers and comment-separated sections, often labeled in English while some titles and outputs are in German.
- No classes or package structure are used in the current scripts; the code is organized as single-file exercises with small helper functions.

## Practical guidance for future agents
- Treat each script as a teaching example first and a software module second.
- Keep the code easy to paste, combine, and adapt for similar exercise variants.
- If a change risks obscuring the math, prefer a simpler implementation.
- Preserve the current educational format even when improving correctness or readability.

