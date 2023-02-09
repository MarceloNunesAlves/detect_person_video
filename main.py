import exception
from process_log import config_log
from flask import Flask, request
import threading
import queue
import detector

logger = config_log(__name__)

app = Flask(__name__)
q = queue.Queue()
detect = detector.detect()

@app.route("/", methods=['POST'])
def push_message():
    requisicao = request.get_json(force=True)
    try:
        path = str(requisicao['path_video'])
        q.put(path)
    except:
        raise exception.InvalidUsage('Não foi possivel enviar para a fila!', status_code=400)
    return "OK"

def worker():
    while True:
        item = q.get()
        print(f'Working on {item}')
        detect.video_analysis(item)
        print(f'Finished {item}')
        q.task_done()


# Turn-on the worker thread.
threading.Thread(target=worker, daemon=True).start()

# Block until all tasks are done.
if __name__ == '__main__':
    logger.debug('Inicio do endpoint')
    app.run(host="0.0.0.0", port=5001)
