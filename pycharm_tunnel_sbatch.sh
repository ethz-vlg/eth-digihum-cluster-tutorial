#!/bin/bash
#SBATCH --job-name=cluster-tunnel
#SBATCH --account=digital_human_jobs
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/slurm_output__%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G

set -euo pipefail

TUNNEL_HOME="$HOME/.config/cluster-tunnel"
TUNNEL_CACHE="$HOME/.cache/cluster-tunnel"

mkdir -p "$TUNNEL_HOME" "$TUNNEL_CACHE"
chmod 700 "$TUNNEL_HOME" "$TUNNEL_CACHE"

if [ ! -f "$TUNNEL_HOME/ssh_host_ed25519_key" ]; then
    cluster-tunnel config >/dev/null
fi

PORT="$(
    python3 - <<'PY'
import socket

s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"

scontrol update "JobId=$SLURM_JOB_ID" "Comment=$PORT"

SETENV="$(
    env | awk -F= '
        /^(CUDA|SLURM|NVIDIA)/ { print }
        /^PATH=/ { print }
        /^LD_LIBRARY_PATH=/ { print }
    ' | while IFS='=' read -r key value; do
        case "$value" in
            *[!A-Za-z0-9_.,:\/@%+=-]*)
                ;;
            *)
                printf '%s=%s ' "$key" "$value"
                ;;
        esac
    done | sed 's/[[:space:]]*$//'
)"

echo "Starting SFTP-enabled cluster tunnel on $(hostname -s):$PORT"
echo "Use the normal 'cluster-tunnel config' SSH block on your laptop."

ARGS=(
    -D -e
    -p "$PORT"
    -f /dev/null
    -o "PidFile=$TUNNEL_CACHE/sshd.pid"
    -o "PrintLastLog=no"
    -o "PasswordAuthentication=no"
    -o "KbdInteractiveAuthentication=no"
    -o "Subsystem=sftp /usr/lib/openssh/sftp-server"
    -h "$TUNNEL_HOME/ssh_host_ed25519_key"
)

if [ -n "$SETENV" ]; then
    ARGS+=(-o "SetEnv=$SETENV")
fi

exec /usr/sbin/sshd "${ARGS[@]}"
