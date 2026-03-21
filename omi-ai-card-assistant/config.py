import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# System configuration

# --- CAMERA SETUP ---
# To use your phone camera, you have two main options:
#
# Option 1: Using "IP Webcam" app (WiFi)
# - Download "IP Webcam" on your phone.
# - Start the server in the app.
# - It will show an IP address (e.g., http://192.168.1.100:8080)
# - Change CAMERA_SOURCE in .env to that URL + "/video"
# Example: CAMERA_SOURCE = "http://192.168.1.100:8080/video"
#
# Option 2: Using "DroidCam" or "Iriun Webcam" (WiFi or USB)
# - Install the app on your phone AND the client on your PC.
# - Connect them. Your PC will now have a "Virtual Webcam".
# - In this case, keep it as a number in .env! Try 0, 1, or 2 until it finds your phone.
# Example: CAMERA_SOURCE = 1

# Load CAMERA_SOURCE from env, default to 0 if not set
_camera_source = os.getenv("CAMERA_SOURCE", "0")

# If it's a digit (like "0", "1"), convert to int for OpenCV
if _camera_source.isdigit():
    CAMERA_SOURCE = int(_camera_source)
else:
    CAMERA_SOURCE = _camera_source
