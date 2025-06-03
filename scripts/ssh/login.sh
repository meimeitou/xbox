#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOSTS_FILE_NAME="${HOSTS_FILE_NAME:-hosts.csv}"
HOSTS_FILE="${HOME}/${HOSTS_FILE_NAME}"
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
    IFS=',' read -r user host desc <<< "${hosts[$i]}"
    printf "\033[31m%d. \033[0m%s@%s  [%s]\n" "$i" "$user" "$host" "$desc"
done

echo "-----------------------------"
echo ""
read -p "Enter Host Number: " HOST_NUM

re='^[0-9]+$'
if ! [[ $HOST_NUM =~ $re ]] || [ "$HOST_NUM" -ge "${#hosts[@]}" ]; then
   echo "请输入有效数字" >&2; exit 1
fi

if [ "$HOST_NUM" -eq 0 ]; then
    read -p "请输入用户名: " user
    read -p "请输入主机地址: " host
    read -p "请输入描述: " desc
    echo "${user},${host},${desc}" >> "$HOSTS_FILE"
    echo "ssh 登录串: ssh ${user}@${host}"
    $EXPECT_FILE "${user}" "${HOST_PASSWORD}" "${host}"
    exit 0
fi

IFS=',' read -r user host desc <<< "${hosts[$HOST_NUM]}"
echo "ssh 登录串: ssh ${user}@${host}"

$EXPECT_FILE "${user}" "${HOST_PASSWORD}" "${host}"
