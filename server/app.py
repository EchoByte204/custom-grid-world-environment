import sys
import os

# Append project root to sys.path so 'app.py' at the root can be resolved
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app
import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
