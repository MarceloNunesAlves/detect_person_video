from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from google.oauth2 import service_account
import os

path_credentials = os.getenv("PATH_GOOGLE_CREDENTIALS", os.path.expanduser('~') + "/credentials_conta_servico.json")
folder_id = os.getenv("ID_FOLDER_DRIVER_URL")

def upload_to_drive(file_path, file_name):
    credentials = service_account.Credentials.from_service_account_file(
        path_credentials, scopes=['https://www.googleapis.com/auth/drive.file'])

    service = build('drive', 'v3', credentials=credentials)

    media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=False) #application/octet-stream'

    file_metadata = {'name': file_name, 'parents': [folder_id]}

    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(F'Arquivo {file_name} salvo com sucesso no drive com o id: {file.get("id")}')
