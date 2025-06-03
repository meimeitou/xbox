#!/usr/bin/env bash

set -e
set -o pipefail

skopeo copy --src-no-creds --override-os linux --override-arch amd64  docker://$1  docker://$2
