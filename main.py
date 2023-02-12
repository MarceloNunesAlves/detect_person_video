from utils.verify_interval_datetime import check_time
import exception
from process_log import config_log
from flask import Flask, request
import send_email
import threading
import queue
import detector
import os

logger = config_log(__name__)

app = Flask(__name__)
q = queue.Queue()
detect = detector.detect()

@app.route("/", methods=['POST'])
def push_message():
    requisicao = request.get_json(force=True)
    try:
        path        = str(requisicao['path_video'])
        location    = str(requisicao['location'])
        try:
            hour_start = int(requisicao['hour_start_email'])
        except:
            hour_start = None
        try:
            hour_end = int(requisicao['hour_end_email'])
        except:
            hour_end = None

        q.put({"path": path,
               "hour_start_email": hour_start,
               "hour_end_email": hour_end,
               "location": location})
    except:
        raise exception.InvalidUsage('Não foi possivel enviar para a fila!', status_code=400)

    return "OK"

def delete_file(file_save):
    os.remove(file_save)

def worker():
    while True:
        item = q.get()

        logger.debug(f"Inicio da verificação {item['path']}")

        # Detecção de pessoas no video
        file_save, name_file_final = detect.video_analysis(item['path'], item['location'])

        if file_save is not None:
            if item['hour_start_email'] is not None and item['hour_end_email'] is not None and check_time(item['hour_start_email'], item['hour_end_email']):
                send_email.send_email(file_save, item['location'], name_file_final)

            # Remove o arquivo após o envio
            delete_file(file_save)

        logger.debug(f"Final do processo {item['path']}")
        q.task_done()


# Turn-on the worker thread.
threading.Thread(target=worker, daemon=True).start()

# Block until all tasks are done.
if __name__ == '__main__':
    logger.debug('Inicio do endpoint')
    app.run(host="0.0.0.0", port=5001, load_dotenv=False)
