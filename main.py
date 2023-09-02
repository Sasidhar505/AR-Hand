import socket

host = '127.0.0.1' 
port = 5050       
data = "0,0,0"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.sendall(str(data).encode())
s.close()