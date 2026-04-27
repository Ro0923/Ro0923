import cv2
import os
import numpy as np
import hashlib

IMG_SIZE = 224
MAX_FRAMES = 16

seen_hashes = set()

def video_hash(video_path):
    cap = cv2.VideoCapture(video_path)
    hashes = []

    count = 0
    while count < 5:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hashes.append(hashlib.md5(gray.tobytes()).hexdigest())
        count += 1

    cap.release()
    return "".join(hashes)


def extract_frames_and_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    frames, flows = [], []

    ret, prev = cap.read()
    if not ret:
        return None, None

    prev = cv2.resize(prev, (IMG_SIZE, IMG_SIZE))
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        frames.append(frame)
        flows.append(flow)
        prev_gray = gray

    cap.release()

    if len(frames) < 5:
        return None, None

    return np.array(frames), np.array(flows)


def process_split(split):
    for label in ["Fight", "NonFight"]:
        in_dir = os.path.join("RWF-2000", split, label)
        out_dir = os.path.join("Processed_Clean", split, label)
        os.makedirs(out_dir, exist_ok=True)

        for video in os.listdir(in_dir):
            if not video.endswith(".avi"):
                continue

            video_path = os.path.join(in_dir, video)

            v_hash = video_hash(video_path)
            if v_hash in seen_hashes:
                print(f"SKIPPED duplicate: {video_path}")
                continue

            frames, flows = extract_frames_and_flow(video_path)
            if frames is None:
                continue

            seen_hashes.add(v_hash)

            name = os.path.splitext(video)[0]
            np.save(os.path.join(out_dir, name + "_frames.npy"), frames)
            np.save(os.path.join(out_dir, name + "_optflow.npy"), flows)

            print(f"Processed: {video_path}")


if __name__ == "__main__":
    process_split("train")
    process_split("val")
