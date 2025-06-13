#!/usr/bin/env expect
set timeout 1

set HOST [lindex $argv 0]
set PASSWORD [lindex $argv 1]

# 判断 PASSWORD 是否为空，如果为空则要求输入
if { "$PASSWORD" == "" } {
    set pwfile [open "$env(HOME)/.host_password.txt" r]
    set PASSWORD [string trim [read $pwfile]]
    close $pwfile
}

# ssh -J alice@bastion.example.com $USER@$HOST # 使用堡垒机登录
spawn ssh root@$HOST
expect {
    "fingerprint"  {send "yes\r";exp_continue}
    "password:" {send "${PASSWORD}\r";}
}
expect "~]$"
send "sudo su\r"
expect {
    "密码" {send "${PASSWORD}\r";}
    "password" {send "${PASSWORD}\r";}
}
interact