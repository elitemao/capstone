import os

with open("/Volumes/personal1T/code/GOOGLE_API_KEY.txt", "r", encoding="utf-8") as f:
    GOOGLE_API_KEY = f.read().strip()

GOOGLE_API_KEY

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ADK Web / server environment should set GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Please set GOOGLE_API_KEY in your environment.")
