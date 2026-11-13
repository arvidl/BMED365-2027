Arvid Lundervold, 2026-02-11

# BMED365-2027: Lab 5 - Computational Modeling with LLMs (Local)

**Target Hardware:** MacBook Pro M4 Max (128GB Unified Memory)

**Model:** Qwen 2.5 72B (Instruct) via `mlx-lm`

**Environment:** uv (`mlx-bio`, see [`pyproject.toml`](./pyproject.toml)) + JupyterLab or Cursor IDE

## 1. Overview

This module sets up a high-performance local Large Language Model (LLM) environment for computational biology. We utilize the **M4 Max's unified memory** to run a quantized 72-billion parameter model locally. This allows for zero-latency inference, data privacy (patient/genomic data never leaves your laptop), and deep integration with Python workflows.

## 2. Prerequisites

* **macOS (Apple Silicon):** Sequoia (15.0) or later recommended for best Metal support.
* **Terminal:** You can use the built-in Terminal or the terminal inside Cursor.
* **Package Manager:** [`uv`](https://docs.astral.sh/uv/) (same as the main BMED365-2027 environment).

## 3. Environment Setup (The `mlx-bio` uv Environment)

All dependencies—including scientific/biological libraries (`numpy`, `scipy`, `biopython`, `scanpy`) and Apple's MLX framework (`mlx`, `mlx-lm[train]`)—are pinned in [`pyproject.toml`](./pyproject.toml) and [`uv.lock`](./uv.lock).

### Step 3.1: Create and Sync the Environment

Run the following commands in your terminal from the repository root:

```bash
cd Lab5-Comp-Mod/MLX-Bio-Qwen
uv sync
```

### Step 3.2: Register Kernel for Jupyter / Cursor (once)

This step ensures JupyterLab, VS Code, and Cursor can select the `mlx-bio` kernel:

```bash
uv run python -m ipykernel install --user --name mlx-bio --display-name "Python (BMED365 - MLX Bio)"
```

### Step 3.3: Launch JupyterLab (optional)

```bash
uv run jupyter lab
```

## 4. Cursor IDE Integration

1. Open this folder (`Lab5-Comp-Mod`) in **Cursor**.
2. Open your `.ipynb` notebook.
3. **Select Kernel:**
* Click the Kernel picker (top-right, usually says "Python 3").
* Select **Python (BMED365 - MLX Bio)**.



## 5. Usage: Running Qwen 2.5 72B

The following Python code loads the model into your RAM. On an M4 Max (128GB), this will consume approximately **45GB of RAM**, leaving ~80GB free for your biological datasets and context window.

### Python Loader Script (Put this in your first Notebook Cell)

```python
from mlx_lm import load, generate

# The 4-bit quantized version is optimized for Apple Silicon
# HuggingFace Repo: mlx-community/Qwen2.5-72B-Instruct-4bit
model_id = "mlx-community/Qwen2.5-72B-Instruct-4bit"

print(f"Loading {model_id} into Unified Memory...")
model, tokenizer = load(model_id)
print("✅ Model loaded. Ready for computational tasks.")

def ask_bio_agent(prompt, max_tokens=2000, temp=0.3):
    """
    Queries the local Qwen model.
    temp=0.3 is preferred for scientific accuracy (less creative/hallucinatory).
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt_formatted, 
        verbose=False, 
        max_tokens=max_tokens, 
        temp=temp
    )
    return response

```

---


## 6. The "Computational Medicine & Biology" System Prompt

When using the model for Lab 5 tasks—ranging from ODE modeling of pandemics to stochastic simulation of gene expression or MRI physics—prepend the instructions below to your session or wrap your prompt with this context.

### System Prompt Description

* **Role:** Expert Professor of Computational Medicine & Biophysics.
* **Tone:** Academic, mathematically rigorous, and code-centric.
* **Key Capabilities:** Covers Deterministic Systems (ODEs), Stochastic Processes (SDEs/HMMs), and Medical Physics (MRI/Diffusion).

### The Prompt Text

```text
You are an expert Professor of Computational Medicine and Biomedical Physics assisting the graduate course BMED365. Your goal is to translate complex biological and medical problems into precise, high-performance computational models.

**I. Domain Expertise & Methodology**
You must recognize and employ the correct mathematical frameworks for the following domains:

1.  **Dynamical Systems (Oncology & Epidemiology):**
    * **Tumor Growth:** Use Gompertzian, Logistic, or Von Bertalanffy models.
    * **Epidemiology:** Implement compartmental models (SIR, SEIR) with time-dependent parameters.
    * **Stability Analysis:** For multistable systems, ALWAYS provide **Phase Plane** analysis (nullclines, vector fields) alongside time-series plots.

2.  **Biophysics & Neuroscience:**
    * **Electrophysiology:** Use Hodgkin-Huxley (HH) or FitzHugh-Nagumo formalisms.
    * **Transport:** Apply Fick’s Laws of Diffusion or Nernst-Planck equations.

3.  **Medical Physics & Imaging:**
    * **MRI/NMR:** Use the **Bloch Equations** (T1/T2 relaxation, precession) using rotation matrices.
    * **Pharmacokinetics:** Use multi-compartment PK/PD models (ADME) with coupled ODEs.

4.  **Stochastic Processes:**
    * **Discrete:** Use Hidden Markov Models (HMM) or Gillespie algorithms for molecular noise.
    * **Continuous:** For systems like Ornstein-Uhlenbeck, use **Stochastic Differential Equations (SDEs)**.

**II. Operational Guidelines for Code & Math**

1.  **Mathematical Rigor:**
    * Before coding, ALWAYS write the governing equations using LaTeX.
    * Explicitly define state variables (e.g., $M_z$ for magnetization, $V_m$ for voltage) and units ($T$, $mV$, $mM$).

2.  **High-Performance Python (M4 Max Optimization):**
    * **Vectorization:** Never use loops for matrix operations (essential for MRI/Bloch simulations). Use `numpy` broadcasting.
    * **Solvers:**
        * **ODEs:** Use `scipy.integrate.odeint` or `solve_ivp`.
        * **SDEs:** Implement the **Euler-Maruyama** method explicitly (standard ODE solvers fail for noise).
    * **HMMs:** Use `hmmlearn` or explicit matrix multiplication for forward-backward algorithms.

3.  **Visualization:**
    * Generate `matplotlib` code for every simulation.
    * Ensure plots have clear labels, legends, and units.

**Example Interaction:**
User: "Model a noisy mean-reverting gene expression."
You:
1. Define SDE: $dX_t = \theta(\mu - X_t)dt + \sigma dW_t$ (Ornstein-Uhlenbeck).
2. Explain terms: $\theta$ (reversion speed), $\mu$ (mean), $\sigma$ (volatility).
3. Python code: Implement Euler-Maruyama loop using `numpy.random.normal` for $dW_t$.
4. Plot: Time series with mean line.

```

---



## 7. Troubleshooting

* **"Killed: 9" Error:** This usually means you ran out of RAM. Ensure you are not running Docker or other heavy background apps.
* **Slow Generation:** Verify `mlx-lm` is using the GPU. Run `import mlx.core as mx; print(mx.default_device())` in Python. It should output `Device(gpu, 0)`.

---

*Created for BMED365-2027 | Lab 5: Computational Modeling*
