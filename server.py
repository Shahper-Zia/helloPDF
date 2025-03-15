from chainlit.cli import run_chainlit

def start():
    run_chainlit("app2.py", headless=True)

if __name__ == "__main__":
    start()
