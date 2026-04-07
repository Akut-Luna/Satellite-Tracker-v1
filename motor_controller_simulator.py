import socket
import threading
from flask import Flask
from waitress import serve

app = Flask(__name__)
state = {"az": 0.0, "el": 0.0, "ph": 10, "pv": 10}

def handle_client(conn, addr):
    print(f"[TCP] Connected by {addr}")
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            
            print(f"[RECV] {len(data)} bytes: {data.hex(' ')}")
            
            if len(data) < 13:
                print("[WARN] Packet too short, ignoring.")
                continue

            cmd = data[11]
            
            if cmd == 0x2F: # SET
                try:
                    h_val = int(data[1:5].decode('ascii'))
                    v_val = int(data[6:10].decode('ascii'))
                    state["az"] = (h_val / state["ph"]) - 360
                    state["el"] = (v_val / state["pv"]) - 360
                    print(f"[CMD] SET -> Az: {state['az']}°, El: {state['el']}°")
                except Exception as e:
                    print(f"[ERR] Failed to decode SET: {e}")
            
            elif cmd in [0x0F, 0x1F]: # STOP or STATUS
                if cmd == 0x0F:
                    label = "STOP" 
                else:
                    label = "STATUS"

                h = int((state["az"] + 360) * state["ph"])
                v = int((state["el"] + 360) * state["pv"])
                
                resp = bytearray([0x57])
                resp.extend([h // 1000, (h // 100) % 10, (h // 10) % 10, h % 10])
                resp.append(state["ph"])
                resp.extend([v // 1000, (v // 100) % 10, (v // 10) % 10, v % 10])
                resp.append(state["pv"])
                resp.append(0x20)
                
                conn.sendall(resp)
                print(f"[CMD] {label} -> Sent Response: {resp.hex(' ')}")

def start_tcp_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 50007))
    server.listen()
    print("[INIT] TCP Motor Simulator listening on port 50007...")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    threading.Thread(target=start_tcp_server, daemon=True).start()
    @app.route('/')
    def index(): return f"Current State - Az: {state['az']}, El: {state['el']}"

    serve(
        app, 
        host='0.0.0.0', 
        port=8080,
        threads=6
    )
    app.run(port=8080)