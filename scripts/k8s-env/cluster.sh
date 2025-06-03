#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

venv_spawn() {
  dotdir="$(mktemp -d)"
  cat > "${dotdir}/.zshrc" <<EOF
case "\$(basename "\$SHELL")" in
  zsh)
    export ZDOTDIR="\$OLD_ZDOTDIR"
    if [ -f "\$ZDOTDIR/.zshenv" ]; then
      . "\$ZDOTDIR/.zshenv"
    fi
    if [ -f "\$ZDOTDIR/.zshrc" ]; then
      . "\$ZDOTDIR/.zshrc"
    fi
    ;;
  bash)
    if [ -f ~/.bashrc ]; then
      . ~/.bashrc
    fi
    if [ -f /etc/bash.bashrc ]; then
      . /etc/bash.bashrc
    fi
    ;;
esac
. $1
printf >&2 '\\nYou have entered the kube env.\\n'
printf >&2 'Use CTRL-D or "exit" to quit.\\n'
EOF
  ln -s "${dotdir}/.zshrc" "${dotdir}/.bashrc"
  case $(basename "${SHELL}") in
    zsh)
      export OLD_ZDOTDIR="${ZDOTDIR:-${HOME}}"
      export ZDOTDIR="${dotdir}"
      exec zsh -i
      ;;
    bash)
      exec bash --init-file "${dotdir}/.bashrc" -i
      ;;
    *)
      printf >&2 'Unrecognized shell %s\n' "${SHELL}"
      ;;
  esac
}

TMP_FILE="/tmp/.env"

$ROOT_DIR/pick.py  $TMP_FILE

if [ ! -f $TMP_FILE ];then
   exit 0
fi

venv_spawn  $TMP_FILE

exit 0