#!/usr/bin/env expect
set timeout 1

set USER [lindex $argv 0]
set PASSWORD [lindex $argv 1]
set HOST [lindex $argv 2]

# 判断 PASSWORD 是否为空，如果为空则要求输入
if { "$PASSWORD" == "" } {
    stty -echo
    send_user "请输入主机密码: "
    expect_user -re "(.*)\n"
    set PASSWORD $expect_out(1,string)
    stty echo
    send_user "\n"
}

spawn ssh $USER@$HOST
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