from flask import Flask, request, send_file, render_template
import os
import glob
import mysql.connector

app = Flask(__name__)


@app.route('/')
def test():
    return render_template('gallery.html')

@app.route('/api/videos')
def fetch_videos():
    cnx = mysql.connector.connect(user='root', password='howardwang2000', host='mysql', database='surveillance')
    cursor = cnx.cursor()
    timestamp = request.args.get('time')
    # fetch_all = "SELECT * FROM videos"
    fetch_videos = "SELECT * FROM videos WHERE start_time < '{}'".format(timestamp)
    cursor.execute(fetch_videos)
    rows = cursor.fetchall()
    images = glob.glob('../files/*.png')
    return render_template('gallery.html', image_names=images)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

