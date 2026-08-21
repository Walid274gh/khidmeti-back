import subprocess, sys, socket
print("=== test DNS depuis kernel ===", flush=True)
for host in ["pypi.org", "huggingface.co", "www.kaggle.com"]:
    try:
        ip = socket.gethostbyname(host)
        print(f"DNS OK {host} -> {ip}", flush=True)
    except Exception as e:
        print(f"DNS FAIL {host}: {e}", flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"],
                   capture_output=True, text=True)
print("pip jiwer rc=", r.returncode, r.stdout[-200:], r.stderr[-200:], flush=True)