# DeepFusionGuard — Setup & Workflow (Group RG16)

How the work is split:

| Where | What runs there |
|---|---|
| **Laptop** (VS Code) | Writing code, git, quick CPU tests (`pytest`), tiny samples |
| **Colab free (T4 GPU)** | Preprocessing, feature extraction, training |
| **Google Drive** | Everything that must survive a session: data archives, cached features, checkpoints, logs, model cache |
| **GitHub** | The code. The only way code travels from laptop to Colab |

Golden rules:
1. Code is edited **only on the laptop**, then `git push`. Colab only `git pull`s.
2. Anything not saved to Drive is **lost** when the Colab session ends (max ~12 h, ~90 min idle).
3. Never read thousands of small files directly from `/content/drive`. Stage archives to local disk first (`stage_in` / `stage_out` in the notebook).

---

## Part A — Laptop, one time (each teammate)

Ubuntu and Windows 10/11 are both supported. Commands are given for each; use the one for your OS.
Locally everything runs on the **CPU**, on both systems. Do not install CUDA, ROCm or `torch-directml`.

**A1. Install tools**

Ubuntu:
```bash
sudo apt update && sudo apt install -y git ffmpeg
sudo snap install code --classic          # VS Code
```
Windows (PowerShell):
```powershell
winget install --id Git.Git -e
winget install --id Microsoft.VisualStudioCode -e
```
FFmpeg, the Visual C++ runtime (needed by PyTorch) and uv are installed by the setup script in A3.
Close and reopen PowerShell after installing Git so `git` is on PATH.

**A2. Get repo access.** The repo owner (kritikapandey1) adds the other two:
GitHub → repo → Settings → Collaborators → Add people. Accept the email invite.

**A3. Clone and set up the environment**

Ubuntu:
```bash
git clone https://github.com/kritikapandey1/Major_project.git
cd Major_project
bash scripts/setup_env.sh                 # installs uv, Python 3.12, CPU torch; prints a report
```
Windows (PowerShell):
```powershell
git config --global core.autocrlf false   # once, BEFORE cloning; .gitattributes then handles line endings
git clone https://github.com/kritikapandey1/Major_project.git
cd Major_project
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```
(Or double-click `scripts\setup_env.bat`.) The `-ExecutionPolicy Bypass` applies to this one run only;
it does not change your system's PowerShell security setting.

The report must end with `All checks passed.` If not, copy the full output to the team chat.

**A4. Git identity** (once per laptop)
```bash
git config --global user.name  "Your Name"
git config --global user.email "your_github_email@example.com"
```
To push, GitHub needs a Personal Access Token as the password
(GitHub → Settings → Developer settings → Personal access tokens), or use `gh auth login`.

**A5. VS Code**
1. `code .` inside the repo folder.
2. Extensions (Ctrl+Shift+X): **Python**, **Jupyter** (both Microsoft), **Google Colab** (Google).
3. Ctrl+Shift+P → *Python: Select Interpreter* → choose `.venv/bin/python` (Ubuntu)
   or `.venv\Scripts\python.exe` (Windows).

---

## Part B — Google Drive, one time (team)

All three must use the **same** Drive folder, or your cached features and checkpoints get split across accounts.

**B1. Owner runs the notebook first** (Part C). It creates `My Drive/DeepFusionGuard/` with:
```
archives/raw/        dataset archives (.tar/.zip), as downloaded
archives/processed/  preprocessed clips, packed as .tar
features/            cached encoder outputs (.npy)
checkpoints/         model weights, saved every epoch
logs/                CSV training logs
hf_cache/            Hugging Face models (downloaded once, reused)
```

**B2. Owner shares it.** In Drive: right-click `DeepFusionGuard` → Share → add both teammates as **Editor**.

**B3. Teammates add a shortcut, before running the notebook themselves.**
Drive → *Shared with me* → right-click `DeepFusionGuard` → *Organize* → *Add shortcut* → *My Drive*.
The path `/content/drive/MyDrive/DeepFusionGuard` then points to the shared folder for everyone.
If you run the notebook *before* adding the shortcut, it creates your own empty folder instead:
delete that folder, add the shortcut, run again.

**B4. Storage.** Files count against the owner's quota (part of JUIT's pooled storage).
Drive also limits uploads to 750 GB per user per day, so plan large dataset uploads across days.

---

## Part C — Every Colab session

**C1. Open the notebook** (browser route, recommended):
colab.research.google.com → *File → Open notebook* → *GitHub* tab → search `kritikapandey1/Major_project`
→ `notebooks/colab_setup.ipynb`.

VS Code route (optional): open `notebooks/colab_setup.ipynb` → *Select Kernel* → *Colab* → *New Colab Server* → GPU.
If the Drive-mount cell fails there, use the browser route.

**C2. Pick the GPU:** *Runtime → Change runtime type → T4 GPU → Save.*

**C3. Run everything:** *Runtime → Run all.* Approve the Drive permission popup.
The first install in a fresh session downloads the CUDA build of torch (~2 GB on Google's network) and takes a few minutes.

**C4. Check the report** printed by the environment-report cell: `Platform: Colab`, a `GPU: Tesla T4` line, and `All checks passed.`

**C5. Do the work** in the cells added for each step (preprocessing, features, training).
Outputs go to Drive directly (small files) or via `stage_out` (folders of many files).

**C6. Finish cleanly:** make sure results are on Drive, then *Runtime → Disconnect and delete runtime*.
This stops quota being used while you are away.

Do not save notebook edits back to GitHub from Colab. If you want to keep outputs, use *File → Save a copy in Drive*.

---

## Part D — Daily development loop

```
laptop: git pull --rebase            # get teammates' changes first
laptop: edit code in VS Code
laptop: .venv/bin/python -m pytest   # quick CPU tests (Windows: .venv\Scripts\python -m pytest)
laptop: git add … && git commit -m "Step N: …" && git push
colab : re-run "3. Clone or update the repo" cell (does git pull)
colab : run the step's cells
```
Write the commit ID in the weekly log: `git rev-parse --short HEAD`.

Keep commits small and push often, so three people editing the same repo rarely conflict.
If `git pull --rebase` reports a conflict, stop and resolve it together. Don't force-push.

---

## Part E — When Colab won't give you a GPU

Colab's free GPU quota is dynamic, and some days it refuses ("Cannot connect to GPU backend"). Options:
1. Do CPU-friendly work: coding, tests, small-sample preprocessing on the laptop.
2. Try again later; quota usually returns within a day.
3. **Kaggle** (free, weekly GPU hours): Notebook → Settings → Accelerator → **GPU T4 x2** (P100 also works:
   the install cell picks torch's CUDA 12.6 build for it). Kaggle uses Kaggle Datasets instead of Drive;
   a Kaggle version of the setup notebook will be added when we first need it.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Report says `torch 2.x found; project pins 2.11.x` | Re-run the install cell; check it printed `Install finished.` |
| `Transport endpoint is not connected` on Drive paths | Drive mount dropped. Run `drive.mount("/content/drive", force_remount=True)` |
| Odd numpy/torch errors when importing *inside* the notebook | *Runtime → Restart session*, then run again from the top (project scripts run as `!python …` are unaffected) |
| `no kernel image is available` | Wrong CUDA build for this GPU. Re-run the install cell and check the "torch backend" line it prints |
| Windows: `OSError: [WinError 126]` / DLL load failed when importing torch | Install the VC++ runtime: `winget install --id Microsoft.VCRedist.2015+.x64 -e`, then open a new terminal |
| Windows: `ffmpeg` / `uv` "not recognized" right after install | Open a new PowerShell window (PATH is only refreshed in new windows), then re-run the setup script |
| Windows: "running scripts is disabled on this system" | Run the script via `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1` or the `.bat` file |
| Linux: `$'\r': command not found` in a `.sh` script | The file got Windows line endings. Fix it with `sed -i 's/\r$//' scripts/setup_env.sh`, and make sure the repo has `.gitattributes` |
| Session ended mid-training | Normal. Training resumes from the last checkpoint in `checkpoints/` (added in Step 5) |
