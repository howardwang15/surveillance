from flask import Flask, request, send_file, render_template
import os
import glob
import mysql.connector
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def test():
    return render_template('gallery.html')

@app.route('/api/videos')
def fetch_videos():
    cnx = mysql.connector.connect(user='root', password='howardwang2000', host='mysql', database='surveillance')
    cursor = cnx.cursor()
    timestamp = request.args.get('datetime')

    person = request.args.get('person')
    animal = request.args.get('animal')
    if person and animal:
        fetch_videos = "SELECT \
                            start_time, video_name, first_frame \
                        FROM videos \
                        WHERE start_time < '{}'".format(timestamp)
    elif person:
        fetch_videos = "SELECT \
                            start_time, video_name, first_frame \
                        FROM videos INNER JOIN detections \
                        ON videos.id=detections.video_id \
                        WHERE type='person' \
                        GROUP BY start_time, video_name, first_frame"
    elif animal:
        fetch_videos = "SELECT \
                            start_time, video_name, first_frame \
                        FROM videos INNER JOIN detections \
                        ON videos.id=detections.video_id \
                        WHERE type!='person' and type!='car' \
                        GROUP BY start_time, video_name, first_frame"
    else:
        fetch_videos = "SELECT \
                            start_time, video_name, first_frame \
                        FROM videos \
                        WHERE start_time < '{}'".format(timestamp)

    cursor.execute(fetch_videos)
    rows = cursor.fetchall()

    times = [start_time for start_time, _, _ in rows]
    images = [os.path.join('/static', 'files', first_frame) for _, _, first_frame in rows]
    videos = [os.path.join('/static', 'files', video_name) for _, video_name, _ in rows]
    print(videos)
    data = [(time, image, video) for time, image, video in zip(times, images, videos)]

    return render_template('gallery.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

