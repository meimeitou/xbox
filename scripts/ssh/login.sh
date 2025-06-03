#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOSTS_FILE_NAME="${HOSTS_FILE_NAME:-hosts.csv}"
HOSTS_FILE="${ROOT_DIR}/${HOSTS_FILE_NAME}"
EXPECT_FILE="${ROOT_DIR}/login-expect.sh"
HOST_PASSWORD_FILE="${HOST_PASSWORD_FILE:-$HOME/.host_password.txt}"

[ -n "$HOSTS_FILE_NAME" ] || {
  echo "HOSTS_FILE_NAME is not set"
  exit 1
}

if [ ! -f "$HOSTS_FILE" ]; then
  echo "Hosts file not found: $HOSTS_FILE"
  exit 1
fi

# 检查密码文件
if [ ! -f "$HOST_PASSWORD_FILE" ]; then
  read -s -p "请输入主机密码: " HOST_PASSWORD
  echo
  echo "$HOST_PASSWORD" > "$HOST_PASSWORD_FILE"
  chmod 600 "$HOST_PASSWORD_FILE"
else
  HOST_PASSWORD=$(<"$HOST_PASSWORD_FILE")
fi

echo "可登录主机列表："
mapfile -t hosts < "$HOSTS_FILE"

for i in "${!hosts[@]}"; do
    IFS=',' read -r user host <<< "${hosts[$i]}"
    printf "\033[31m%d. \033[0m%s@%s\n" "$i" "$user" "$host"
done

echo "-----------------------------"
echo ""
read -p "Enter Host Number: " HOST_NUM

re='^[0-9]+$'
if ! [[ $HOST_NUM =~ $re ]] || [ "$HOST_NUM" -ge "${#hosts[@]}" ]; then
   echo "请输入有效数字" >&2; exit 1
fi

IFS=',' read -r user host <<< "${hosts[$HOST_NUM]}"
echo "ssh 登录串: ssh ${user}@${host}"

# printf "ssh- %s@%s\n" "${val[@]}" "${HOST_NUM}"

$EXPECT_FILE ${user} "${HOST_PASSWORD}" ${host}
