from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
import smtplib
import os

threshold_identified = float(os.getenv("THRESHOLD_IDENTIFIED", 0.80))
from_email = os.getenv("FROM_EMAIL", "seuemail@email.com")
to_email = os.getenv("TO_EMAIL", "destinatario@email.com").split(",")
pwd_email = os.getenv("PASSWORD_EMAIL", "xxxxx")

def send_email(file_save, location, name_file_final):

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = COMMASPACE.join(to_email)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = f"ATENÇÃO - Pessoa identificada no horário de supervisão - {location}"

    # Adiciona a mensagem ao objeto MIMEMultipart
    message = f"Foi identificado uma pessoa suspeita no horário de vigilância ativa na área: {location}, com um percentual de certeza de {(threshold_identified * 100):.0f}%.\nVerifique com cuidado."
    msg.attach(MIMEText(message))

    part = MIMEBase('video', "mp4")#application, octet-stream
    part.set_payload(open(file_save, "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename=name_file_final)
    msg.attach(part)

    smtp = smtplib.SMTP('smtp-mail.outlook.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login(from_email, pwd_email)
    smtp.sendmail(from_email, to_email, msg.as_string())
    smtp.quit()