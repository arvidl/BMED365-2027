# Quick Start: AI-Assisted Python Programming with Productive Friction

**Date:** Tuesday, January 12, 2027, 09:15–13:00  
**Location:** 9a110bp  
**Lecturer:** [Arvid Lundervold](https://www.uib.no/en/persons/Arvid.Lundervold) (Department of Biomedicine & MMIV, UiB)  
**Target audience:** Medical and biomedicine students (BSc/MSc/MD/PhD) — no prior programming experience required  
**Materials:** [Notebook (`quickstart-ai-python.ipynb`)](./notebooks/quickstart-ai-python.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab-QuickStart/notebooks/quickstart-ai-python.ipynb) · [Beamer Slides (`main.pdf`)](./Beamer/main.pdf) · [Google Slides](https://docs.google.com/presentation/d/e/2PACX-1vTglMMlBOEnMGR3d7RGKlvXBrKGKRQF38SPvaq0Ie1AwlfqoUoB4rBVYi6qBtLzPYNZVNxLXXiHVyJK/pub?start=false&loop=false&delayms=3000)

---

## Modern AI in Biomedical Education: Co-Existence, Productive Friction, and Verification

Why learn Python programming in 2027 when generative AI can write hundred-line scripts in seconds?

In **[BMED365 (*Computational Imaging, Modeling and AI in Biomedicine*)](../README.md)**, our answer comes directly from empirical learning science, clinical AI research, and [Ethan Mollick's](https://www.oneusefulthing.org/) reflections in ***[Co-Existence: The Next Phase of AI](https://co-existence.ai/)*** (Portfolio/Penguin, 2026; building on *[Co-Intelligence](https://www.penguinrandomhouse.com/books/741805/co-intelligence-by-ethan-mollick/)*, 2024): **capable AI makes rigorous domain and computational education *more* necessary, not less.**

### 1. What the Evidence Shows: AI Tutor vs. AI Crutch

Recent randomized trials and clinical studies reveal a sharp divide between using AI to **think *with*** versus using AI to **skip thinking**:

- **Structured AI tutoring accelerates learning — passive offloading harms it:** In a field experiment with nearly 1,000 students, [Bastani et al. (*PNAS*, 2025)](https://doi.org/10.1073/pnas.2422633122) showed that unrestricted ChatGPT access boosted immediate task scores (+48%) but **reduced subsequent unassisted exam performance by 17%** compared with never having AI at all — whereas a prompt-guarded Socratic AI tutor preserved learning while boosting practice performance (+127%). Similarly, in a randomized controlled trial at Harvard, [Kestin et al. (*Scientific Reports*, 2025)](https://doi.org/10.1038/s41598-025-97652-6) found that a pedagogically scaffolded AI tutor **more than doubled learning gains** in less time than in-class active learning — precisely because it gave graduated hints rather than finished answers.
- **Why unrestricted "help on demand" backfires (*productive struggle*):** [Poulidis, Bastani & Bastani (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5604932) demonstrated over 12 weeks of training that learners who could click an AI hint button anytime achieved **less than half the skill gain** (30% vs. 64%) of learners whose AI support was withheld inside their Zone of Proximal Development. Removing cognitive effort removes what [Bjork & Bjork (2011)](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/04/EBjork_RBjork_2011.pdf) call **desirable difficulties** — the active prediction, retrieval, and error-repair that wire durable mental models (see also [Shen & Tamkin, 2026](https://doi.org/10.48550/arXiv.2601.20245) on coding skill formation and [Lee et al., *CHI '25*](https://doi.org/10.1145/3706598.3713778) on cognitive effort and confidence).
- **Clinical stakes — *deskilling* and *never-skilling* in medicine:** In healthcare, unverified reliance on AI is not merely an exam problem; it is a patient-safety risk. A multicenter colonoscopy study ([Budzyń et al., *The Lancet Gastroenterology & Hepatology*, 2025](https://doi.org/10.1016/S2468-1253(25)00133-5)) found that clinicians' unassisted adenoma detection rate dropped significantly after routine AI exposure (*deskilling*). For students in training, [Ke et al. (*Nature Medicine*, 2026)](https://doi.org/10.1038/s41591-026-04438-y) warn of an even deeper hazard — **AI-induced *never-skilling***, where trainees lean on AI before forming baseline diagnostic and quantitative intuition in the first place. Meanwhile, [Griot et al. (*Nature Communications*, 2025)](https://doi.org/10.1038/s41467-024-55628-6) show that current LLMs lack reliable metacognition in medical reasoning: they sound equally fluent when they are right and when they are wrong.

### 2. Working with AI Along a Jagged Frontier (Insights from Mollick's *Co-Existence*)

In *[Co-Existence: The Next Phase of AI](https://co-existence.ai/)*, Ethan Mollick frames living and working alongside capable AI around four practical ideas that directly shape how we work in BMED365:

1. **Map the *Jagged Frontier* empirically:** AI capability is uneven ([Dell'Acqua, Mollick et al., 2023](https://doi.org/10.2139/ssrn.4573321)). The same multimodal model that writes clean PyTorch boilerplate or explains the Bloch MRI equation can confidently hallucinate the number of dark seeds in a single 7T MRI slice (`kiwi_064.png`) or leak patient data between training and test sets. You only discover the "valleys" by testing against reality.
2. **Choose your working mode deliberately (*Centaur*, *Cyborg*, or *Conjurer*):**
   - **Centaur mode** (*"I do this part; AI does that part"*) splits the work cleanly between human and machine (e.g., *you* define the physiological hypothesis, physical units, and train/test split; *AI* drafts the `matplotlib` figure formatting).
   - **Cyborg mode** (*"AI and I work through this together"*) interleaves human and AI line-by-line inside a tight loop (predicting output, asking Socratic *"why?"* questions, co-debugging an error traceback).
   - **Conjurer mode** (*"AI, carry out this assignment; I will evaluate what comes back"*) delegates an end-to-end task to an AI coding agent — powerful *only* when you already possess the domain expertise and automated sanity checks to audit what comes back. Reduced involvement during execution makes the human's specification and final evaluation more important, not less. Avoid becoming a *"reverse centaur"* (a human reduced to copy-pasting for a machine they cannot evaluate).

   These are not a ladder from inferior to superior. The appropriate mode depends on the task, the AI's capabilities, and the cost of checking its work - and one project can move among all three.
3. **Protect the durable human edge (*Deep Knowledge, Wide Knowledge, Taste, and Agency*):** As syntax becomes cheap, the bottleneck shifts from *generating code* to **judgment** — knowing what biological question is worth asking (*agency*), connecting physiology with imaging physics and mathematics (*wide & deep knowledge*), and recognizing when a clean-looking plot is scientifically wrong (*taste*).
4. **Use AI where *reality checks the work*:** Computational imaging and biophysical modeling are ideal arenas for AI co-existence because code is executable and constrained by **physical units (`mm³`, `ms`), conservation laws, anatomical priors, and held-out validation data**.

### 3. Our QuickStart Pedagogy: AI-Assisted Learning *with Productive Friction*

Throughout this Quick Start (and the rest of BMED365), we replace passive copy-pasting with four deliberate **friction habits**:

| Habit | What *Not* to Do (Zero Friction / Crutch) | What We Do Instead (Productive Friction) |
|:------|:------------------------------------------|:-----------------------------------------|
| **1. Predict → Run → Explain** | Press `Shift+Enter` blindly down the notebook | Predict what a cell will print or plot *before* running it; inspect mismatches |
| **2. Attempt First, Hint Second** | Paste the exercise prompt straight into Gemini/ChatGPT | Write a 2-minute skeleton or pseudocode first; open graduated `<details>` hints; ask AI for a *Socratic hint*, not the full solution ([Mollick & Mollick, 2023](https://doi.org/10.2139/ssrn.4475995)) |
| **3. Audit Plausible AI Bugs** | Trust code because it runs without a `SyntaxError` | Hunt for silent clinical & scientific bugs (unit mismatches, `>` vs. `>=` cutoffs, 2D vs. 3D voxel geometry, data leakage) |
| **4. Reality-Check Outputs** | Accept a fluent chatbot answer or `accuracy = 1.0` at face value | Check physical units, edge cases, visual overlays, and biological plausibility |

---

## Learning Objectives

After completing this Quick Start session, you should be able to:

| Topic | Learning Objective |
|:------|:-------------------|
| **Notebooks & Platforms** | Run and save Jupyter Notebooks in **Google Colab** (or locally with `uv` and the `bmed365-2027` kernel) |
| **Python Building Blocks** | Use variables, data types, lists, `if/else` conditions, `for` loops, functions, and 2D `NumPy` arrays |
| **AI with Productive Friction** | Partner with Gemini / ChatGPT / Claude / `chat.uib.no` in *Centaur* and *Cyborg* modes to explain, debug, and **critically verify** code |
| **Machine Learning & Networks (Labs 0–1)** | Train and evaluate a classification model (`scikit-learn`) and build/analyze a Patient Similarity Network (`NetworkX`) |
| **Computational Imaging & Modeling (Labs 4–5)** | Inspect a real 7T MRI slice (`kiwi_064.png`) as a numerical matrix, compute 3D voxel volumes, see why naive thresholding/LLM counting fails, and explore a 1-line biophysical growth curve |

---

## Preparation

### Before the session

1. **Option A — Zero-install in the browser (recommended for Quick Start): Google Colab**
   - Create or sign in to a [Google account](https://accounts.google.com).
   - Open [colab.research.google.com](https://colab.research.google.com), click **New notebook**, type `print("Hello, BMED365!")`, and press `Shift+Enter`.
   - When opening the course notebook from GitHub, click **File → Save a copy in Drive** so your edits are saved.

2. **Option B — Run locally on macOS (Apple Silicon) or Linux (`uv`)**
   - If you already completed the local setup from Lecture 1 (see the [main course README](../README.md#option-1-run-locally-macos-apple-silicon-or-linux)), run `uv run jupyter lab` from `BMED365-2027` (or open the folder in VS Code / Cursor) and select the **`bmed365-2027`** kernel. Every cell in this notebook works identically in Colab and locally.

3. **AI Assistant Access (choose at least one)**
   - **Built into Colab:** Gemini (`✨` icon in Google Colab)
   - **UiB privacy-approved chat:** [chat.uib.no](https://chat.uib.no) (log in with your UiB Feide account — never paste real patient data into consumer AI tools!)
   - **Optional external tools:** [ChatGPT](https://chatgpt.com), [Claude](https://claude.ai), or [Gemini](https://gemini.google.com)

### Bring with you
- Laptop (physical keyboard strongly recommended over tablet) + charger
- Curiosity — and willingness to make predictions and mistakes before asking AI!

---

## Agenda (Tuesday, January 12, 2027 · 09:15–13:00)

| Time | Module | Focus & Friction Element | Course Link |
|:-----|:-------|:-------------------------|:------------|
| **09:15–09:45** | **1. Welcome, Setup & How We Learn with AI** | Jupyter/Colab vs. local `uv`, first lines of code, *Predict-Before-You-Run* | [Main README](../README.md) |
| **09:45–10:30** | **2. Python Building Blocks for Biomedicine** | Variables, clinical formulas (BMI, MAP), types, lists, `if/else`, loops, & functions | [Skills F01–F14](../Course-description-topics/Beamer/F-Skills/main.pdf) |
| **10:30–10:45** | **Break** | ☕ | |
| **10:45–11:25** | **3. AI as Thinking Partner (With Friction!)** | *Centaur/Cyborg* prompting, Socratic tutor prompt, debugging tracebacks, & **auditing a confident AI bug** | [Lab 3 (GenAI & LLMs)](../Lab3-GenAI-LLM/README.md) |
| **11:25–12:05** | **4. Machine Learning Taste (Lab 0 Bridge)** | Tabular data (`pandas`), scatter plots, `DecisionTreeClassifier`, & testing generalization | [Lab 0 (ML)](../Lab0-ML/README.md) · [Lab 2 (DL)](../Lab2-DL/README.md) |
| **12:05–12:30** | **5. Patient Similarity Networks (Lab 1 Bridge)** | Nodes, weighted edges, degree centrality, & perturbing a clinical graph (`NetworkX`) | [Lab 1 (PSN)](../Lab1-NetworkSci-PSN/README.md) |
| **12:30–12:50** | **6. Pixels, Voxels & Models: The Kiwi 7T MRI Bridge** | Images as 2D `NumPy` arrays, naive threshold trap, 3D voxel `mm³` math, & $E_{\max}$ organ growth slider | [Lab 4 Kiwi](../Lab4-Comp-Imag/Kiwi-challenge/README.md) · [Lab 5 Kiwi](../Lab5-Comp-Mod/Kiwi-challenge/README.md) |
| **12:50–13:00** | **7. Summary & The Road Ahead** | Self-test checklist, Socratic study prompts, and how Block 1 & Block 2 connect | [Team](../Team-project/README.md) & [Individual](../Individual-project/README.md) Projects |

---

## Course Notebook & Slides

| Resource | Description | Open |
|:---------|:------------|:----:|
| **[`quickstart-ai-python.ipynb`](./notebooks/quickstart-ai-python.ipynb)** | Interactive hands-on notebook with *Predict-First* prompts, AI-bug audits, graduated `<details>` hints, ML, PSN, and the Kiwi 7T MRI mini-lab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab-QuickStart/notebooks/quickstart-ai-python.ipynb) |
| **[`Beamer/main.pdf`](./Beamer/main.pdf)** | Companion Beamer lecture slides (*source:* [`Beamer/main.tex`](./Beamer/main.tex)) | [PDF](./Beamer/main.pdf) |

---

## Ready-to-Use "Socratic AI Tutor" Prompt

When you get stuck while studying on your own, paste this system prompt at the start of a chat in [chat.uib.no](https://chat.uib.no), Claude, ChatGPT, or Gemini so the AI acts as a **coach that preserves productive struggle** rather than an answer dispenser:

```text
Act as a Socratic Python tutor for a biomedicine student in BMED365.
Rules:
1. Do NOT give me the complete working code right away.
2. First ask me what I think the code or error message means, or ask me to predict the output.
3. Give me one small hint at a time, using biomedical examples (blood pressure, MRI pixels/voxels, patient tables).
4. Once my code runs, ask me one "reality-check" question about units, edge cases, or biological plausibility.
```

---

## Where This Connects Across BMED365 (Spring 2027)

- **[Lab 0: Machine Learning](../Lab0-ML/README.md)** — Tabular classification, cross-validation, and AutoML (`scikit-learn` & `PyCaret`)
- **[Lab 1: Network Science & PSN](../Lab1-NetworkSci-PSN/README.md)** — Graph theory and Patient Similarity Networks in precision medicine
- **[Lab 2: Deep Learning](../Lab2-DL/README.md)** — Neural networks, CNNs for ECG & medical images, Grad-CAM (XAI), and TabPFN
- **[Lab 3: Generative AI & LLMs](../Lab3-GenAI-LLM/README.md)** — Transformers, prompt & context engineering, XAI, medical AI ethics (EU AI Act), trustworthy & neurosymbolic AI
- **[Lab 4: Computational Imaging](../Lab4-Comp-Imag/README.md)** — Brain mpMRI glioma segmentation, IMC, and the **[Kiwi 3D Seed Segmentation Challenge](../Lab4-Comp-Imag/Kiwi-challenge/README.md)**
- **[Lab 5: Computational Modeling](../Lab5-Comp-Mod/README.md)** — Physiological ODEs/PDEs (action potentials, tumor growth, blood flow, kidney filtration, wound healing) and the **[Kiwi Morphogenesis & 7T MRI Digital Twin](../Lab5-Comp-Mod/Kiwi-challenge/README.md)**
- **[Team Project](../Team-project/README.md)** & **[Individual Project](../Individual-project/README.md)** — Interdisciplinary glioblastoma research plan (Block 1) and personal speed-poster investigation (Block 2)

---

## References & Further Reading

### AI in Education, Skill Formation & Productive Friction
1. **Mollick, E.** (2026). *[Co-Existence: The Next Phase of AI](https://co-existence.ai/)*. Portfolio / Penguin Random House. *(Chapters 4–6 & 8: Centaur/Cyborg/Conjurer modes, the human edge, and why AI makes education more necessary.)*
2. **Bastani, H., Bastani, O., Sungu, A., Ge, H., Kabakcı, Ö., & Mariman, R.** (2025). [Generative AI without guardrails can harm learning: Evidence from high school mathematics](https://doi.org/10.1073/pnas.2422633122). *Proceedings of the National Academy of Sciences (PNAS)*, 122(26), e2422633122.
3. **Kestin, G., Miller, K., Klales, A., Milbourne, T., & Ponti, G.** (2025). [AI tutoring outperforms in-class active learning: an RCT introducing a novel research-based design in an authentic educational setting](https://doi.org/10.1038/s41598-025-97652-6). *Scientific Reports*, 15, 17458.
4. **Ke, Y., Jin, L., Ong, J. C. L., et al.** (2026). [AI-induced never-skilling in medical education](https://doi.org/10.1038/s41591-026-04438-y). *Nature Medicine*, 32(6), 1997–2006.
5. **Budzyń, K., Romańczyk, M., ..., Kalager, M., Bretthauer, M., & Mori, Y.** (2025). [Endoscopist deskilling risk after exposure to artificial intelligence in colonoscopy: a multicentre, observational study](https://doi.org/10.1016/S2468-1253(25)00133-5). *The Lancet Gastroenterology & Hepatology*, 10(10), 896–903.
6. **Griot, M., Hemptinne, C., Vanderdonckt, J. M., & Yüksel, D.** (2025). [Large Language Models lack essential metacognition for reliable medical reasoning](https://doi.org/10.1038/s41467-024-55628-6). *Nature Communications*, 16, 642.
7. **Poulidis, S., Bastani, H., & Bastani, O.** (2025). [Self-Regulated AI Use Hinders Long-Term Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5604932). *SSRN Working Paper*.
8. **Shen, J. H., & Tamkin, A.** (2026). [How AI Impacts Skill Formation](https://doi.org/10.48550/arXiv.2601.20245). *arXiv:2601.20245*.
9. **Lee, H.-P., Sarkar, A., Tankelevitch, L., et al.** (2025). [The Impact of Generative AI on Critical Thinking: Self-Reported Reductions in Cognitive Effort and Confidence Effects from a Survey of Knowledge Workers](https://doi.org/10.1145/3706598.3713778). *Proceedings of ACM CHI '25*.
10. **Dell'Acqua, F., McFowland III, E., Mollick, E. R., et al.** (2023). [Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality](https://doi.org/10.2139/ssrn.4573321). *Harvard Business School Working Paper 24-013*.
11. **Mollick, E. R., & Mollick, L.** (2023). [Assigning AI: Seven Approaches for Students, with Prompts](https://doi.org/10.2139/ssrn.4475995). *Wharton Generative AI Labs* ([genai.wharton.upenn.edu](https://genai.wharton.upenn.edu/)).
12. **Bjork, E. L., & Bjork, R. A.** (2011). [Making Things Hard on Yourself, But in a Good Way: Creating Desirable Difficulties to Enhance Learning](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/04/EBjork_RBjork_2011.pdf). In *Psychology and the Real World* (pp. 56–64).

### Python & Jupyter Self-Study
- [Kaggle Learn: Intro to Python](https://www.kaggle.com/learn/python) — Free, short interactive exercises
- [Jupyter Notebook 101](https://www.kaggle.com/code/jhoward/jupyter-notebook-101) — Jeremy Howard's practical introduction
- [Wharton Generative AI Labs Prompt Library](https://genai.wharton.upenn.edu/prompt-library/) — Open educational prompts for tutoring and simulation

---

## Frequently Asked Questions

<details>
<summary><b>Do I need prior programming or math experience?</b></summary>

No! This Quick Start begins from absolute zero (`print("Hello!")`). Every mathematical or clinical formula we use is written first in plain English and anchored in everyday biomedical units (`kg`, `m`, `mmHg`, `°C`, `mm³`).
</details>

<details>
<summary><b>Why shouldn't I just let AI write all my lab code?</b></summary>

Because code that *runs* is not the same as science that is *right*. As studies in [*PNAS* (2025)](https://doi.org/10.1073/pnas.2422633122) and [*Nature Medicine* (2026)](https://doi.org/10.1038/s41591-026-04438-y) show, skipping the struggle early prevents you from building the baseline mental model needed to catch silent AI mistakes later — such as unit mismatches, data leakage, or counting MRI artifacts as lesions.
</details>

<details>
<summary><b>Should I use Google Colab or install the local uv environment?</b></summary>

For this 4-hour Quick Start, **Google Colab** is the fastest way to begin (zero installation). If you use macOS (Apple Silicon) or Linux and want to run notebooks locally with `uv` (`bmed365-2027` kernel), both work seamlessly — and you can mix them freely during the semester.
</details>

<details>
<summary><b>Can I paste medical data into ChatGPT, Gemini, or Claude?</b></summary>

Public open-access course datasets (such as Iris, simulated patient lists, or the UiB 7T kiwifruit MRI) are completely safe to use. **Never** paste identifiable patient data or unpublished clinical records into external AI services; for UiB work, use institutional tools such as [chat.uib.no](https://chat.uib.no) in accordance with UiB privacy guidelines.
</details>

---

## Contact

For academic questions about the session or course, contact [Arvid Lundervold](https://www.uib.no/en/persons/Arvid.Lundervold).

---

*Updated: Spring 2027 edition*
