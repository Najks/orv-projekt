from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'C:\Users\nikda\project-rai-backend\uploads'