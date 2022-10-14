# ssh ProxyCommand

## 문제 상황
- 사내 서버는 모두 Linux
- 모든 연결은 ssh를 사용
- 사내에 서버가 여러개 있는데, 그 중 하나만 외부에서 접근이 가능했고 나머지 서버는 사내 망 밖에서는 접근할 수 없었다.
- 따라서 외부 접근이 가능한 하나의 서버에 들어가서, 작업을 해야하는 서버로 접근해야 하는 상황

## 문제 해결 방법
- Reference를 보면 여러 방법이 있는 것 같은데, 사용한 방법은 ssh ProxyCommand를 활용한 방법이다.
- `.ssh`의 config 파일의 내용에 ProxyCommand를 한 줄 추가해준다.

## 해결 방법
### Windows(개인 컴퓨터) > 연결가능서버(server1) > 연결해야하는서버(server2)
```python
Host server1
User username
HostName username@server1_IP
Port 8888
IdentityFile ~/.ssh/id_rsa

Host server2
User username
HostName username@server2_IP
Port 9999
IdentityFile ~/.ssh/id_rsa
ProxyCommand C:\Windows\System32\OpenSSH/ssh.exe server1 -W %h:%p
```

- config에 추가한 코드
```
ProxyCommand C:\Windows\System32\OpenSSH/ssh.exe {Host} -W %h:%p
```
```
ProxyCommand C:\Windows\System32\OpenSSH/ssh.exe {username@server1_IP:Port} -W %h:%p
```

### MacOS(개인 컴퓨터) > 연결가능서버(server1) > 연결해야하는서버(server2)
```python
Host server1
User username
HostName username@server1_IP
Port 8888
IdentityFile ~/.ssh/id_rsa

Host server2
HostName username@server2_IP
Port 9999
User username
IdentityFile ~/.ssh/id_rsa
ProxyCommand ssh server -W %h:%p

```

- config에 추가한 코드
```
ProxyCommand ssh {Host} -W %h:%p
```
```
ProxyCommand ssh {username@server1_IP:Port} -W %h:%p
```


## Reference
[SSH ProxyCommand example: Going through one host to reach another server](https://www.cyberciti.biz/faq/linux-unix-ssh-proxycommand-passing-through-one-host-gateway-server/ )

[ProxyCommand를 이용한 SSH 중계 접속](https://w.cublr.com/application/openssh/proxycommand)