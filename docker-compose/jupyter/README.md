# jupyter

```shell
# generate password
python -c "from jupyter_server.auth import passwd; print(passwd())"

# tls certificates
GEN_CERT=no

# change password in docker-compose.yml and run
docker compose -f docker-compose.yml up -d
```
