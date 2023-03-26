from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, check_img_size, cv2, increment_path, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

from utils.plots import plot_one_box
from upload_driver import upload_to_drive
from process_log import config_log
from datetime import datetime
from pathlib import Path

import numpy as np
import time
import os
import sys
import torch

count_min_identified = int(os.getenv("COUNT_MIN_IDENTIFIED", 900)) # Representa um minuto de uma pessoa identificada no video
labels_identified = os.getenv("LABELS_IDENTIFIED", 'person,persons').split(",")
threshold_identified = float(os.getenv("THRESHOLD_IDENTIFIED", 0.80))
path_destination = os.getenv("PATH_DESTINATION", "attach_file")
remove_source = bool(os.getenv("REMOVE_SOURCE_FILE_IN_THE_END", "False"))

logger = config_log(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class detect():

    def __init__(self):

        weights = ROOT / 'yolov5n.pt'  # model path or triton URL
        data = ROOT / 'data/coco128.yaml'  # dataset.yaml path

        self.vid_stride = 1  # video frame-rate stride
        # Load model
        self.device = select_device('')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=False)
        self.imgsz = check_img_size((640, 640), s=self.model.stride)  # check image size
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.model.names]

    def video_analysis(self, source, location):

        name_file_final = f"VIDEO_{location}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"

        # Directories
        save_dir = Path(path_destination)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)

        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt, vid_stride=self.vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        #self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        #seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        count_labels_detection = 0
        t0 = time.time()

        frames = []

        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            t1 = time.time()
            pred = self.model(im, augment=False)
            t2 = time.time()

            # NMS
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
            t3 = time.time()

            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Verify label
                        if self.model.names[int(cls)] not in labels_identified or conf < threshold_identified:
                            continue

                        count_labels_detection += 1

                        label = f'{self.model.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)


                # Print time (inference + NMS)
                logger.debug(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Save results (video with detections)
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                frames.append(im0)

            # Print time (inference-only)


        if count_labels_detection <= count_min_identified:
            save_path = None
        else:
            # Converter o array de frames para um array numpy
            frames = np.array(frames)

            # Escrever o vídeo com todos os frames de uma só vez
            for frame in frames:
                vid_writer.write(frame)

            # Fecha arquivo gerado
            vid_writer.release()

            frames = None

            upload_to_drive(save_path, name_file_final)

        if remove_source:
            os.remove(source)

        logger.info(f'Quantidade de objetos encontrados: {count_labels_detection:.0f} - {name_file_final}')
        logger.debug(f'Feito. ({time.time() - t0:.3f}s)')

        return save_path, name_file_final

