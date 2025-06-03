#!/usr/bin/env bash

set -e
set -o pipefail

HUB_MIRROR_DIR="$(dirname "${BASH_SOURCE[0]}")"
HUB_MIRROR_URL="${HUB_MIRROR_URL}"
HUB_CREDS_AUTHFILE="${HUB_MIRROR_DIR}/.authfile"
HUB_CREDS=""

[ -n "$HUB_MIRROR_URL" ] || {
  printf >&2 'HUB_MIRROR_URL is not set\n'
  exit 1
}

[ -f "${HUB_MIRROR_DIR}/VERSIONS" ] || {
  printf >&2 'VERSIONS file does not exist in HUB_MIRROR_DIR: %s\n' "$HUB_MIRROR_DIR"
  exit 1
}

check_deps() {
  if [ -z "$(command -v skopeo)" ]; then
    case "$OSTYPE" in
      darwin*)
        printf >&2 'Missing skopeo:\n\n  brew install skopeo\n'
        exit 1
        ;;
      *)
        printf >&2 'Missing skopeo:\n\n  https://github.com/containers/skopeo#obtaining-skopeo\n'
        exit 1
        ;;
    esac
  fi
}

check_creds() {
  printf 'Registry: %s\n' "${HUB_MIRROR_URL}"

  if [ -f "${HUB_CREDS_AUTHFILE}" ]; then
    HUB_CREDS="$(cat "${HUB_CREDS_AUTHFILE}")"
  fi
  if [ -n "${HUB_CREDS}" ]; then
    printf 'Username: (Using %s)\n' "${HUB_CREDS_AUTHFILE}"
    printf 'Password: (Using %s)\n' "${HUB_CREDS_AUTHFILE}"
    return
  fi

  username=""
  password=""

  while [ -z "$username" ]; do
    printf 'Username: '
    read -r username
  done

  while [ -z "$password" ]; do
    printf 'Password: '
    stty -echo
    read -r password
    stty echo
    printf '\n'
  done

  HUB_CREDS="$username:$password"
  (
    umask 0177
    printf '%s\n' "${HUB_CREDS}" > "${HUB_CREDS_AUTHFILE}"
  )
}

sync_image() {
  skopeo \
    --override-os linux \
    --override-arch amd64 \
    copy \
    --src-no-creds \
    --dest-creds "$HUB_CREDS" \
    "docker://$1" \
    "docker://$2"
}

printf '=> Gathering system facts\n'
check_deps
check_creds

while read -r source_image target_prefix target_image
do
  target_repo="${HUB_MIRROR_URL}/${target_prefix}/${target_image}"
  printf '=> Copying %s to %s\n' "${source_image}" "${target_repo}"
  sync_image "${source_image}" "${target_repo}"
done < "${HUB_MIRROR_DIR}/VERSIONS"
