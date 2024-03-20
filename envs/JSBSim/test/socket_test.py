import socket
import struct

def main():
    host = '127.0.0.1'
    port = 54000

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        for _ in range(5):
            # Receive a number
            data = s.recv(1024)
            message = data.decode()
            print("Received:", message)

            # Add 1 and send it back
            message = "1, 2, 3, 4, 5"
            s.sendall(message.encode())
            print("Sent:", message)

if __name__ == '__main__':
    main()
