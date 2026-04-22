# The Student Cluster Guide

[`Digital Humans FS 26`](https://vlg.inf.ethz.ch/teaching/Digital-Humans-FS-26.html) | `Cluster Tutorial` | `23.4.2026`

ETH Zurich's Department of Computer Science (D-INFK) provides the student cluster for GPU jobs in this course. The official cluster documentation is maintained by ISG and should be treated as the source of truth:

- [Student Cluster overview](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster)
- [Running jobs](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs)
- [CUDA and PyTorch](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda)
- [Copying data](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCopyingData)
- [Troubleshooting](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterTroubleshooting)

This tutorial gives one practical workflow for the course: log in, keep work alive with `tmux`, set up Python, run GPU jobs with Slurm, store data in the right place, and develop code without depending too much on fragile remote IDE support.

## Tutorial outline

1. **[Logging into the cluster](#1-logging-into-the-cluster)**
2. **[Tmux and useful configuration files](#2-tmux-and-useful-configuration-files)**
3. **[Python environment setup](#3-python-environment-setup)**
4. **[Accessing GPUs with `srun`](#4-accessing-gpus-with-srun)**
5. **[Accessing GPUs with `sbatch`](#5-accessing-gpus-with-sbatch)**
6. **[Monitoring and canceling jobs](#6-monitoring-and-canceling-jobs)**
7. **[Storage options](#7-storage-options)**
8. **[Useful command-line tools](#8-useful-command-line-tools)**
9. **[Remote file access and port forwarding](#9-remote-file-access-and-port-forwarding)**
10. **[Development workflow](#10-development-workflow)**
11. **[AI coding agents](#11-ai-coding-agents)**
12. **[Two-way sync using Mutagen](#12-two-way-sync-using-mutagen)**

Throughout the tutorial, replace `login_name` with your ETH username, i.e., the part before `@ethz.ch` in your ETH email address. For this course, the Slurm account names are `digital_human` for short interactive/Jupyter use and `digital_human_jobs` for longer Slurm jobs. If Slurm reports that an account is invalid, run `courses` after login and use the accounts printed there. ISG provisioned 100 hours of interactive/Jupyter time and 150 hours of Slurm job time per student.

## 1. Logging into the cluster

Log in with SSH:

```bash
ssh login_name@student-cluster.inf.ethz.ch
```

`student-cluster.inf.ethz.ch` points to one of the login nodes. You can also connect to a specific login node:

```bash
ssh login_name@student-cluster1.inf.ethz.ch
ssh login_name@student-cluster2.inf.ethz.ch
```

Do not run computationally intensive tasks on login nodes. Use login nodes to edit files, manage environments, submit jobs, monitor jobs, and run lightweight commands. Run training, inference, large preprocessing, and notebooks through Slurm/Jupyter instead.

Access from outside ETH usually requires VPN. Follow the [ETH/ISG VPN instructions](https://www.isg.inf.ethz.ch/Main/ServicesNetworkVPN) to set it up before connecting from outside the ETH network.

### 1.1 SSH keys

SSH keys avoid typing your ETH password on every login:

```bash
# On your laptop
ssh-keygen -t ed25519 -C "laptop-to-student-cluster"
ssh-copy-id login_name@student-cluster.inf.ethz.ch
```

If `ssh-copy-id` is not available on your laptop, copy the public key manually:

```bash
cat ~/.ssh/id_ed25519.pub
```

Then append it to `~/.ssh/authorized_keys` on the cluster.

### 1.2 Use a fixed login node for persistent tmux

The generic SSH address `student-cluster.inf.ethz.ch` may send different connections to different login nodes. `tmux` sessions live on the login node where they were started. If you start `tmux` on `student-cluster1` but later connect to `student-cluster2`, you will not see that session.

For persistent work, choose one fixed login node and always reconnect to it:

```bash
ssh -t login_name@student-cluster1.inf.ethz.ch 'tmux new -A -s digihum'
```

This command creates or attaches to a tmux session named `digihum`. If your laptop disconnects, reconnect with the same command and the tmux session should still be there unless the login node rebooted.

## 2. Tmux and useful configuration files

`tmux` keeps shells alive after your SSH connection drops. It is one of the most important tools for cluster work.

Basic tmux commands:

```bash
tmux new -s digihum       # create session named digihum
tmux attach -t digihum    # attach existing session
tmux detach               # leave session running in background
tmux ls                   # list sessions
```

If you copy the provided [`.tmux.conf`](./.tmux.conf) to `~/.tmux.conf`, the prefix is `Ctrl-a` instead of the default `Ctrl-b`.

Useful shortcuts with the provided config:

```txt
Ctrl-a |        split pane horizontally
Ctrl-a -        split pane vertically
Alt-arrow       move between panes
Ctrl-a z        zoom/unzoom the current pane (i.e., full-screen)
Ctrl-a c        create a new tmux window (like a terminal tab)
Ctrl-a n        go to the next tmux window
Ctrl-a p        go to the previous tmux window
Ctrl-a d        detach from session
Ctrl-a P        save pane history to a file
```

The provided config also enables mouse support, so you can click panes, resize panes, and switch tmux windows with the mouse if your terminal supports it.

The repository also provides an example [`.bashrc`](./.bashrc). The most useful parts are persistent shell history and a colored prompt. Do not replace the cluster's default `~/.bashrc` blindly, because site-specific defaults can change. A safer setup is to keep this tutorial config as a separate file and source it from your existing `~/.bashrc`:

```bash
cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
cp ~/.tmux.conf ~/.tmux.conf.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

cp /path/to/this/repo/.bashrc ~/.bashrc.digihum
cp /path/to/this/repo/.tmux.conf ~/.tmux.conf

cat >> ~/.bashrc <<'EOF'

# Digital Humans tutorial shell helpers
if [ -f ~/.bashrc.digihum ]; then
    . ~/.bashrc.digihum
fi
EOF
```

If you only want part of the setup, open the file and copy the pieces that are useful for your workflow.

## 3. Python environment setup

Use either `conda` or Python `venv`. `conda` is convenient for course projects because it can also install non-Python tools such as `nodejs`, `ncdu`, or compilers.

### 3.1 Install Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Accept the license, install into your home directory, and allow the installer to initialize your shell. Then log out and back in:

```bash
exit
ssh -t login_name@student-cluster1.inf.ethz.ch 'tmux new -A -s digihum'
```

If Conda asks you to accept the Anaconda Terms of Service, run the commands it prints, for example:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Install `PyYAML` once in the base environment so the cluster helper commands keep working if Conda `base` is active:

```bash
conda activate base
python -m pip install --upgrade pip PyYAML
```

### 3.2 Create the course environment

The student cluster has multiple CUDA versions through environment modules. A practical setup is to load a specific CUDA module and install matching [PyTorch wheels](https://pytorch.org/get-started/locally/). For example, an environment with the CUDA 13.0 module and PyTorch wheels built for CUDA 13.0 can be installed as:

```bash
conda create -n digihum python=3.11 -y
conda activate digihum

module add cuda/13.0
pip install --upgrade pip
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir gpustat tqdm matplotlib ipython PyYAML
```

Run `conda activate digihum` and `module add cuda/13.0` in each new shell or job before running GPU code. The order is not important for this setup, but the examples below keep this order consistently.

Check that PyTorch imports on the login node:

```bash
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available on this node?', torch.cuda.is_available())"
```

On a login node, `torch.cuda.is_available()` should normally be `False`, because login nodes do not have GPUs. It should become `True` inside a GPU job.

If you need a different CUDA version, inspect available modules:

```bash
module avail
```

Then load the matching module and use the matching PyTorch index URL, for example `cu129` for CUDA 12.9 or `cu130` for CUDA 13.0.

### 3.3 Jupyter notebooks

This tutorial focuses on terminal and Slurm workflows. If you use Jupyter, start it through [student-jupyter.inf.ethz.ch](https://student-jupyter.inf.ethz.ch) and follow the [ISG Jupyter instructions](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterJupyter). Closing the browser tab does not stop the server; stop it from the JupyterHub home page when you are done so it does not keep consuming your time budget.

## 4. Accessing GPUs with `srun`

Slurm manages GPU access. For quick interactive work, use `srun`.

Start a short interactive GPU shell:

```bash
srun --account digital_human --time=00:10:00 --gpus=1 --pty bash --login
```

Your prompt should move from a login node to a GPU node such as `studgpu-node...`. Inside the job:

```bash
hostname
nvidia-smi
conda activate digihum
module add cuda/13.0
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Run the MNIST example:

```bash
cd ~/code/eth-digihum-cluster-tutorial
python train_mnist.py --epochs 1 --batch-size 256 --data-dir /work/scratch/$USER/digihum-tutorial-data
```

The short time limit keeps examples cheap. Increase it for real experiments.

More examples:

```bash
# Request a specific GPU model.
srun --account digital_human --time=00:10:00 --gpus=5060ti:1 --pty bash --login
srun --account digital_human --time=00:10:00 --gpus=2080ti:1 --pty bash --login
srun --account digital_human --time=00:10:00 --gpus=1080ti:1 --pty bash --login

# Request CPU and memory explicitly.
srun --account digital_human --time=00:10:00 --gpus=1 --cpus-per-gpu=2 --mem=24G --pty bash --login
```

For this course, each user can run at most one GPU job at a time, and each job can request one GPU. Current GPU node layouts are:

| GPU type    | GPUs/node | CPUs/node | RAM/node | VRAM/GPU                 | Nodes | Total GPUs |
| ----------- | --------: | --------: | -------: | ------------------------ | ----: | ---------: |
| `5060ti`    |         8 |        28 | ~248 GiB | 16 GB                    |     4 |         32 |
| `2080ti`    |         8 |        36 | ~369 GiB | 11 GB                    |     4 |         32 |
| `1080ti`    |         8 |        20 | ~248 GiB | 11 GB                    |    24 |        192 |
| `gb10`      |         1 |        20 | ~117 GiB | 128 GB, shared with CPUs |     6 |          6 |
| **Total**   |           |           |          |                          |    38 |        262 |

GB10 is useful for projects that need large CUDA memory, but there are only 6 total GB10 GPUs and the maximum runtime is 24 hours, while the other GPU types can run for up to 48 hours.

A good default request is `--cpus-per-gpu=2 --mem=24G`. It works across all four GPU types and avoids reserving more shared-node resources than most starter jobs need. Increase CPU or RAM if data loading, preprocessing, or memory usage becomes the bottleneck; larger requests can make scheduling harder.

Important: if you run `srun` in a normal SSH shell and then close the SSH window, the interactive job is likely to die. Start `srun` inside `tmux` on a fixed login node if you want it to survive laptop sleep or network changes.

## 5. Accessing GPUs with `sbatch`

Use `sbatch` for jobs that should run without an interactive terminal. You submit one script or a batch of scripts, Slurm runs them when resources are available, and you inspect the logs and results after the jobs finish. This is the normal way to run longer experiments or many experiments in sequence.

Submit the provided script:

```bash
sbatch sbatch_example_script.sh
```

Check the queue:

```bash
squeue --user $USER
```

When `sbatch` accepts the job, it prints a line like `Submitted batch job 123456`; that number is the job id. You can also find it with `squeue`.

The provided script writes output to `~/slurm_output__sbatch_example_script.sh-JOBID.out`. Replace `JOBID` with the number printed by `sbatch`:

```bash
cat ~/slurm_output__sbatch_example_script.sh-JOBID.out
```

The provided [sbatch script](./sbatch_example_script.sh) loads CUDA, activates the `digihum` conda environment, checks PyTorch CUDA availability, and runs one epoch of `train_mnist.py`. Use it as the skeleton for your own batch scripts.

## 6. Monitoring and canceling jobs

Basic queue commands:

```bash
squeue
squeue --user $USER
squeue --account digital_human
squeue --account digital_human_jobs
```

Detailed queue view, refreshed every two seconds:

```bash
watch -n 2 'squeue -o "%16i %40j %7a %5g %10v %8u %2t %18R %8Q %5c %4C %8m %5D %13b %10l %10L %20S" --user $USER'
```

Cancel one job:

```bash
scancel JOBID
```

Cancel all your jobs:

```bash
scancel -u $USER
```

If a job is pending, look at the reason column. Common reasons are lack of available resources, asking for too many CPUs/GPUs/RAM, missing or wrong account, or a node that is still powering up.

Useful cluster info:

```bash
courses    # course tags, time limits, and resources
space      # writable storage locations and free space
```

## 7. Storage options

The cluster login scripts print `courses` and `space` when you log in. Run `space` again when you want the current paths and quotas:

```bash
space
```

Use the storage locations as follows:

- Home: `/home/$USER`, 20 GB. Use it for shell configuration, small environments, and code. Do not store datasets or large checkpoints here.
- Personal scratch: `/work/scratch/$USER`, 100 GB per user. Use it for datasets, checkpoints, logs, and temporary project data. It has no backup and old data is cleaned automatically.
- Team storage: `/work/courses/digital_human/teamX`, 100 GB per team. Use it for shared datasets, checkpoints, logs, generated outputs, and other files that several team members need to read or write.
- Course data: `/cluster/courses/digital_human`, if course-provided shared data is used.
- Job-local temporary storage: `$TMPDIR`, under `/tmp` on the compute node. It is fast and deleted when the job ends.

The best tool for inspecting disk usage inside a directory is `ncdu`:

```bash
conda install -c conda-forge ncdu -y
ncdu /work/scratch/$USER
```

If your home fills up, start with caches:

```bash
python -m pip cache info
python -m pip cache purge
conda clean -a
du -sh ~/* ~/.cache ~/.local 2>/dev/null
```

## 8. Useful command-line tools

Commands worth knowing:

- `tmux` or `screen`: keep sessions alive after SSH disconnects.
- `htop`: inspect CPU and RAM usage.
- `gpustat -i 1 -P -u`: compact live GPU view with process and user names.
- `nvidia-smi`: inspect GPU usage.
- `watch -n 1 command`: rerun a command every second, useful for live monitoring.
- `nvtop`: interactive GPU monitor if installed.
- `ncdu`: interactive disk usage analyzer.
- `du -sh path`: quick disk usage summary.
- `rsync`: robust file copying and syncing.
- `scp` and `sftp`: simple SSH-based copy tools.
- `which python`, `which pip`, `python -m pip`: verify which environment you are using.
- `module avail`, `module list`, `module add cuda/13.0`: manage system modules.
- `vim`, `nano`, or `micro`: terminal editors.

If a small tool is missing, try installing it in conda:

```bash
conda install -c conda-forge ncdu htop gpustat nodejs -y
```

## 9. Remote file access and port forwarding

For one-off copies, use `scp` or `rsync`. For a real-time synced setup where files can be edited both locally and remotely, for example when coding locally while running agents on the cluster, use Mutagen; it is described in [section 12](#12-two-way-sync-using-mutagen).

Simple copying:

```bash
# local -> cluster
scp local_file.txt login_name@student-cluster.inf.ethz.ch:/home/login_name/

# cluster -> local
scp login_name@student-cluster.inf.ethz.ch:/home/login_name/remote_file.txt .

# directory sync
rsync -avh --progress ./my_project/ login_name@student-cluster.inf.ethz.ch:/home/login_name/code/my_project/
```

TensorBoard port forwarding:

```bash
# On your laptop
ssh -L 6006:localhost:6006 login_name@student-cluster.inf.ethz.ch

# On the cluster
tensorboard --logdir /path/to/logs --port 6006
```

Then open `http://localhost:6006/` locally.

For experiment logs that should be accessible from anywhere without running a TensorBoard server, consider using Weights & Biases (`wandb`) instead. Keep the local `wandb/` directory out of Git and Mutagen sync.

Use `uploadserver` to browse, download, and upload generated files through a forwarded local port:

```bash
# On the cluster
python -m pip install uploadserver
python -m uploadserver --bind 127.0.0.1 --directory /path/to/project/root 8000

# On your laptop
ssh -L 8000:localhost:8000 login_name@student-cluster.inf.ethz.ch
```

Then open `http://localhost:8000/` locally.

## 10. Development workflow

Use local [PyCharm Professional](https://www.jetbrains.com/pycharm/) as the main development environment. PyCharm is the best IDE for software developers coding in Python, with the best debugger, global search, and indexing capabilities. It is included in the free [JetBrains Student Pack](https://www.jetbrains.com/academy/student-pack/). PyCharm can [configure SSH interpreters](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html), which means the IDE stays local while its run, debug, and indexing features use the cluster environment. Keeping the IDE local avoids many of the instabilities, network delays, and bugs that in my experience come with running the full IDE directly on the cluster, for example through [PyCharm Gateway](https://www.jetbrains.com/help/pycharm/jetbrains-gateway.html) or [VS Code Server](https://code.visualstudio.com/docs/remote/ssh).

To SSH into a GPU node, do not use plain `ssh studgpu-nodeXX`; use ISG's supported [`cluster-tunnel`](https://isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs) workflow instead. Use it when you want PyCharm, a terminal, or another SSH client to connect directly to the allocated GPU node. In that case, the tunnel job replaces the interactive `srun` command: it submits a Slurm job, allocates the GPU node, starts an SSH server inside that job, and exposes it through the login node. Do not start a separate `srun` job for the same work. The provided [`pycharm_tunnel_sbatch.sh`](./pycharm_tunnel_sbatch.sh) keeps the normal `cluster-tunnel` SSH configuration but enables the SFTP subsystem that PyCharm needs, so use it as the default tunnel job.

```bash
# On the login node: print the SSH config once, then copy it to the laptop.
cluster-tunnel config

# On the login node: start the PyCharm-compatible GPU tunnel.
sbatch pycharm_tunnel_sbatch.sh

# Check or stop the tunnel job.
cluster-tunnel status
cluster-tunnel stop

# Optional: from the login node, SSH through the tunnel.
ssh -o ProxyCommand='cluster-tunnel connect' cluster-tunnel

# On the laptop, after adding the SSH config printed by `cluster-tunnel config`:
ssh cluster-tunnel
# or for tmux:
ssh -t cluster-tunnel 'tmux new -A -s gpu-tunnel'
```

In PyCharm, use `cluster-tunnel` as the SSH server. When adding the SSH interpreter, the last step shows the "Sync folders" settings. Do not keep PyCharm's default remote path such as `/tmp/pycharm_project_*`. Instead, map the local project root to a stable path on the cluster, for example `/home/login_name/code/project_name`. For the interpreter, select an existing `Conda` environment, set the Conda path to `/home/login_name/miniconda3/bin/conda` if it is not detected automatically, and select your environment, for example `/home/login_name/miniconda3/envs/digihum`.

**Alternatives.**
If you do not want to use PyCharm or local IDEs, the following are also possible. [VS Code Remote SSH](https://code.visualstudio.com/docs/remote/ssh) runs VS Code locally while connecting to a server-side VS Code component over SSH. [code-server](https://coder.com/docs/code-server) runs a VS Code-like editor on the remote and serves it through the browser. [PyCharm Gateway](https://www.jetbrains.com/help/pycharm/jetbrains-gateway.html) is JetBrains' fully remote IDE workflow. These fully remote IDE setups can work, but in my experience they are often slower, buggier, and less stable than running IDEs locally. For simpler file movement, manual `rsync` is also fine. For debugging, possible alternatives include VS Code remote debugging, PyCharm debug servers, or the simplest option: put `breakpoint()` in the code and debug in the terminal. The right workflow is the one that is fast enough, stable enough, and lets you understand what your code is doing.

To debug the code executed on the GPU node, the overall simplest way available across all environments is the built-in [`breakpoint()`](https://docs.python.org/3/library/pdb.html) function (a shortcut for `import pdb; pdb.set_trace()`). This works on any machine where you have an interactive shell as follows:

1. Insert `breakpoint()` in your code where you wish to pause execution, possibly in an if statement if you are looking for a condition to be met.
2. Run your script as `python your_script.py`.
3. When the `breakpoint()` line is reached, execution pauses and you get an interactive `pdb` debugger prompt.
4. Use `pdb` commands to inspect variables or move around the call stack. Common `pdb` commands include:
   - `h` for help
   - `s` to step to the next line of code
   - `p EXPRESSION` to evaluate and print an expression, e.g., to inspect the value of a variable
   - `c` to continue the code execution
   - `l` to list the code surrounding the breakpoint
   - `ll` to list the complete code of the currently evaluated function or frame
   - `q` to quit the debugger and stop the program

For an example, refer to [`debugging_example.py`](./debugging_example.py) or run `python debugging_example.py`.

## 11. AI coding agents

AI coding agents are allowed in the course with the following policy: "Students may use AI tools of their choice. However, all results must be verifiable, accountable, and reproducible, and students bear full responsibility for meeting these standards." A good cluster workflow is to run agents through their CLI directly on the cluster in a `tmux` session. This lets the agent keep working if the laptop disconnects, run cluster commands and jobs directly, and edit the synced project. Agents can otherwise also run locally on your laptop and interact with the cluster over SSH.

Here is a quick starter for using [Codex CLI](https://developers.openai.com/codex/cli) on the cluster:

```bash
# Run agents in a tmux session on the cluster so they survive SSH disconnects.
ssh -t login_name@student-cluster1.inf.ethz.ch 'tmux new -A -s codex'

# Install Codex into the Conda base environment.
conda activate base
conda install -c conda-forge nodejs=20 -y
npm install -g @openai/codex
npm update -g @openai/codex
codex --version
codex
```

When writing code, use version-control tools to track changes. You do not have to follow standard Git practices such as committing frequently or branching if you do not want to, or if you are not collaborating on the same codebase. Git is still useful for inspecting diffs and understanding what changed. For tracking and reviewing changes, I recommend VS Code because it has an excellent out-of-the-box view of staged and unstaged changes. "Unstaged" changes are everything currently modified. "Staged" changes are the reviewed changes you are happy with. Use staging to separate changes you have reviewed from changes you still need to inspect. When working with coding agents, let their edits appear as unstaged changes, review the important changes carefully and skim the mechanical ones, stage the parts you are happy with, and revert or ask the agent to fix the parts you do not want to keep.

## 12. Two-way sync using Mutagen

If agents edit code on the cluster while you edit locally, you need fast two-way synchronization between the code on the laptop and the code on the cluster. [Mutagen](https://mutagen.io/) is the recommended tool for that. If you only copy code in one direction, `rsync` or PyCharm deployment is enough and you do not need Mutagen.

```bash
# Install Mutagen on the laptop, e.g. with Homebrew on macOS.
brew install mutagen-io/mutagen/mutagen
mutagen version

# Create the cluster project directory.
ssh login_name@student-cluster1.inf.ethz.ch 'mkdir -p ~/code/my_project'

# Create a sync from the laptop project to a fixed login node.
# Only sync code. Do not sync outputs, data, Git metadata, etc.
# This creation command only needs to be run once.
mutagen sync create --name digihum-my-project --ignore-vcs \
  -i 'data' \
  -i 'datasets' \
  -i 'outputs' \
  -i 'logs' \
  -i 'checkpoints' \
  -i '.cache' \
  -i '.pytest_cache' \
  -i '.mypy_cache' \
  -i '.ruff_cache' \
  -i '**/__pycache__' \
  -i '**/.ipynb_checkpoints' \
  -i '*.pyc' \
  -i '*.pyo' \
  -i '*.swp' \
  -i '*.swo' \
  -i '.DS_Store' \
  /Users/YOUR_USERNAME/code/my_project \
  login_name@student-cluster1.inf.ethz.ch:/home/login_name/code/my_project

# Useful Mutagen commands after creation.
mutagen sync list digihum-my-project
mutagen sync monitor digihum-my-project
mutagen sync pause digihum-my-project
mutagen sync resume digihum-my-project
mutagen sync terminate digihum-my-project
```

Use a fixed login node for Mutagen, such as `student-cluster1`, so the SSH endpoint is stable. Keep datasets, checkpoints, logs, and generated outputs outside the synced source tree. `--ignore-vcs` ignores `.git`; use GitHub/GitLab as the source of truth instead of syncing `.git`. If both sides edit the same file while disconnected, Mutagen may report a conflict. Resolve it manually, for example by overwriting one side with `scp` or patching the file in a text editor, then resume.
