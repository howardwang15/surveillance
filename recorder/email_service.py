import os
import smtplib
import datetime
import dateutil.parser
import time
import base64
from email.message import EmailMessage
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

class EmailService():
    def __init__(self, source, dest, password):
        self.source = source
        self.dest = dest
        self.password = password
        self.smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        self.smtp_server.ehlo()
    
    def send_email(self, image_path, video_path, timestamp):
        timestamp = dateutil.parser.parse(timestamp)
        timestamp = timestamp.strftime('%B %d %Y, %I:%M:%S %p')
        
        msg = MIMEMultipart()
        msg['Subject'] = 'Motion detected at {}'.format(timestamp)
        msg['From'] = self.source
        msg['To'] = self.dest


        alternative = MIMEMultipart('alternative')
        alternative.attach(MIMEText('<img src="cid:image1" width=50%>', 'html'))
        msg.attach(alternative)

        with open(os.path.join(image_path), 'rb') as f:
            image_data = f.read()
        
        image_path = image_path.split('/')[-1]
        image = MIMEImage(image_data, name=image_path)
        image.add_header('Content-ID', '<image1>')
        msg.attach(image)


        with open(os.path.join(video_path), 'rb') as f:
            video_data = f.read()

        video_path = video_path.split('/')[-1]
        video = MIMEApplication(video_data, name=video_path)
        msg.attach(video)

        # login to SMTP server and send email message   
        self.smtp_server.login(self.source, self.password)
        self.smtp_server.send_message(msg)
