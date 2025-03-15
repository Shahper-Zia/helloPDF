import os
from chainlit.cli import run_chainlit

def start():
    os.environ["CHAINLIT_HOST"] = "0.0.0.0"  # Listen on all interfaces
    os.environ["CHAINLIT_PORT"] = "8000"  # Listen on port 8000
    run_chainlit("app2.py")

if __name__ == "__main__":
    start()
