# Example ~/.bashrc for the D-INFK student cluster.
# Copy only the parts you want into your own ~/.bashrc.

# Source global definitions if present.
if [ -f /etc/bash.bashrc ]; then
    . /etc/bash.bashrc
elif [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# Do not run interactive setup for non-interactive shells.
[[ $- != *i* ]] && return

# Better bash history: append instead of overwrite, keep timestamps, and flush
# after each command so history survives disconnected SSH sessions.
shopt -s histappend
export HISTFILESIZE=
export HISTSIZE=
export HISTTIMEFORMAT="[%F %T] "
export HISTCONTROL=
export HISTFILE=~/.bash_eternal_history
PROMPT_COMMAND="history -a; ${PROMPT_COMMAND:-}"

# Check terminal size after each command.
shopt -s checkwinsize

# Bash completion if available.
[ -r /usr/share/bash-completion/bash_completion ] && . /usr/share/bash-completion/bash_completion

# Color prompt.
force_color_prompt=yes
if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
        color_prompt=yes
    else
        color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# Set xterm title to user@host:dir.
case "$TERM" in
    xterm*|rxvt*)
        PS1="\[\e]0;\u@\h: \w\a\]$PS1"
        ;;
esac

# Safer defaults and convenience aliases.
alias cp='cp -i'
alias df='df -h'
alias free='free -m'
alias more='less'
alias m='less'
alias ll='ls -hail'
alias lss='ls -hailF'
alias watch='watch --color -n 0.7 '
alias wgpu='nvidia-smi -l 2'

# Git shortcuts.
alias gus='git status'
alias glol='git log --graph --decorate --oneline'
alias gline='git log --oneline'
alias gput='git add . && git stash'
alias gpop='git stash pop'
alias grsa='git restore --staged .'

# Slurm shortcuts.
alias sq='squeue -o "%8i %35j %12a %8u %4t %30R %9Q %5c %8m %20b %11M %11l %11L %20S"'
alias si='sinfo -o "%20P %8a %8l %8D %20G %N"'
alias si2='sinfo -o "%20P %5D %14F %8z %10m %10d %11l %16f %N"'

# Show Slurm associations. Pass filters such as user=$USER if desired.
function show_assoc() {
    sacctmgr -p list associations "$@" format=Account,User,Partition,Qos,DefaultQOS tree | column -ts'|'
}

# Show Slurm QoS definitions. Pass filters such as name=normal if desired.
function show_qos() {
    sacctmgr -p list qos "$@" format=Name,Priority,GraceTime,GrpTRES,GrpJobs,GrpSubmit,MaxTRES,MaxTRESPerUser,MaxJobsPU | column -ts'|'
}

# Archive extractor: ex file.tar.gz
function ex() {
    if [ ! -f "$1" ]; then
        echo "'$1' is not a valid file"
        return 1
    fi

    case "$1" in
        *.tar.bz2) tar xjf "$1" ;;
        *.tar.gz)  tar xzf "$1" ;;
        *.bz2)     bunzip2 "$1" ;;
        *.rar)     unrar x "$1" ;;
        *.gz)      gunzip "$1" ;;
        *.tar)     tar xf "$1" ;;
        *.tbz2)    tar xjf "$1" ;;
        *.tgz)     tar xzf "$1" ;;
        *.zip)     unzip "$1" ;;
        *.Z)       uncompress "$1" ;;
        *.7z)      7z x "$1" ;;
        *)         echo "'$1' cannot be extracted via ex()" ;;
    esac
}

# Conda helper. The exact path depends on where you installed Miniconda.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Project-specific setup examples. Uncomment and adapt if useful.
# cd "$HOME/code/my_project"
# conda activate digihum
# module add cuda/13.0
# export HYDRA_FULL_ERROR=1
