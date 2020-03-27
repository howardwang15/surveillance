from flask import Flask
import os

app = Flask(__name__)


@app.route('/api/videos')
def fetch_videos():
    print('hello')