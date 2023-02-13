from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from upload_driver import upload_to_drive

from process_log import config_log
from pathlib import Path
from numpy import random
from datetime import datetime

import time
import os
import cv2
import torch

count_min_identified = int(os.getenv("COUNT_MIN_IDENTIFIED", 900)) # Representa um minuto de uma pessoa identificada no video
labels_identified = os.getenv("LABELS_IDENTIFIED", 'person,persons').split(",")
threshold_identified = float(os.getenv("THRESHOLD_IDENTIFIED", 0.80))
path_destination = os.getenv("PATH_DESTINATION", "attach_file")

logger = config_log(__name__)

class detect():

    def __init__(self):
        self.imgsz = 640
        # Initialize
        self.device = select_device('')

        # Load model
        self.model = attempt_load('yolov7.pt', map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride

        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def video_analysis(self, source, location):

        name_file_final = f"VIDEO_{location}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"

        # Directories
        save_dir = Path(path_destination)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        count_labels_detection = 0
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / name_file_final)  # Caminho do arquivo de destino
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Verify label
                        if self.names[int(cls)] not in labels_identified or conf < threshold_identified:
                            continue

                        count_labels_detection += 1

                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)


                # Print time (inference + NMS)
                logger.debug(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Save results (video with detections)
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        # Fecha arquivo gerado
        vid_writer.release()

        if count_labels_detection <= count_min_identified:
            os.remove(save_path)
            save_path = None
        else:
            upload_to_drive(save_path, name_file_final)

        logger.debug(f'Quantidade de objetos encontrados: {count_labels_detection:.0f}')
        logger.debug(f'Feito. ({time.time() - t0:.3f}s)')

        return save_path, name_file_final