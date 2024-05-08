"""Seguimiento de objetos con filtro de Kalman."""

import cv2
import numpy as np
import torch
from sort import Sort

# modelo de yolo 5
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def object_tracking():
    """Rastreo de Objetos."""
    cap = cv2.VideoCapture(r"test.mp4")
    mov_tracker = Sort()
    frame_id = 0  # contador de frames
    skip_frames = 6  # número de frames a saltar

    while True:
        ret, frame = cap.read()  # leer cada frame del video
        if not ret:
            break

        # procesar cuadro actual para detección o no
        if frame_id % skip_frames == 0:
            results = yolo(frame)  # aplicar modelo yolo al frame
            results = results.xyxy[0].numpy()
            yolo_people = results[
                results[:, 5] == 0
            ]  # obtener solo detecciones de personas de yolo

        # procesar cada detección
        dets = []
        for *xyxy, conf, cls in yolo_people:
            x1, y1, x2, y2 = map(int, xyxy)  # extraer las coordenadas
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets)

        # actualizar SORT con nuevas detecciones
        trackers = mov_tracker.update(dets)

        # dibujar rectangulos delimitadores y etiquetas
        for d in trackers:
            x1, y1, x2, y2, track_id = map(int, d[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                str(track_id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                thickness=2,
            )

        cv2.imshow("Video Tracker", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        # incrementar el contador de frames
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    object_tracking()
