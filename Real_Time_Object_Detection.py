"""
Object Detection using SSD MobileNet v3 and OpenCV
Author: Subhrajit Baral
Description:
    - Detect objects from images, videos, and webcam
    - Uses pre-trained SSD MobileNet v3 (COCO dataset)
"""

import cv2
import matplotlib.pyplot as plt


# -----------------------------
# Configuration Paths
# -----------------------------
CONFIG_FILE = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
MODEL_FILE = "frozen_inference_graph.pb"
LABELS_FILE = "Labels.txt"

IMAGE_PATH = "download31.jpg"
VIDEO_PATH = "videop.mp4"


# -----------------------------
# Load Class Labels
# -----------------------------
def load_labels(file_path):
    with open(file_path, "rt") as f:
        labels = f.read().strip().split("\n")
    return labels


# -----------------------------
# Load Detection Model
# -----------------------------
def load_model(model_path, config_path):
    model = cv2.dnn_DetectionModel(model_path, config_path)

    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    return model


# -----------------------------
# Object Detection on Image
# -----------------------------
def detect_on_image(model, labels, image_path):
    image = cv2.imread(image_path)

    class_ids, confidences, boxes = model.detect(
        image, confThreshold=0.5
    )

    font = cv2.FONT_HERSHEY_PLAIN

    if len(class_ids) != 0:
        for class_id, conf, box in zip(
            class_ids.flatten(), confidences.flatten(), boxes
        ):
            cv2.rectangle(image, box, (255, 0, 0), 2)
            cv2.putText(
                image,
                labels[class_id - 1],
                (box[0] + 10, box[1] + 40),
                font,
                2,
                (0, 255, 0),
                2,
            )

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Object Detection - Image")
    plt.axis("off")
    plt.show()


# -----------------------------
# Object Detection on Video/Webcam
# -----------------------------
def detect_on_video(model, labels, video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open video or webcam")

    font = cv2.FONT_HERSHEY_PLAIN

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        class_ids, confidences, boxes = model.detect(
            frame, confThreshold=0.55
        )

        if len(class_ids) != 0:
            for class_id, conf, box in zip(
                class_ids.flatten(), confidences.flatten(), boxes
            ):
                if class_id <= 80:
                    cv2.rectangle(frame, box, (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        labels[class_id - 1],
                        (box[0] + 10, box[1] + 40),
                        font,
                        2,
                        (0, 255, 0),
                        2,
                    )

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# Main Function
# -----------------------------
def main():
    labels = load_labels(LABELS_FILE)
    model = load_model(MODEL_FILE, CONFIG_FILE)

    print(f"Loaded {len(labels)} class labels")

    detect_on_image(model, labels, IMAGE_PATH)
    detect_on_video(model, labels, VIDEO_PATH)


if __name__ == "__main__":
    main()
