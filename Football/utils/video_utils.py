import cv2
import os
import pickle

def read_video(video_path,read_from = None):
    if read_from is not None and os.path.exists(read_from):
        with open(read_from,"rb") as f:
            file = pickle.load(f)
        return file
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 1
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        #print(f"Frame {i} read")
        i += 1
    # if read_from is not None:
    #     with open(read_from,"wb") as f:
    #         pickle.dump(frames,f)
    print("Successfully done")
    return frames
def load_files(path):
    frames = []
    i = 1
    for filename in sorted(os.listdir(path)):
        image = os.path.join(r"C:\Users\owner\Desktop\Projects\Football\frames",filename)
        frames.append(cv2.imread(image))
        print(f"Frame {i} of {len(os.listdir(path))} added to frames")
        i+=1
    print("Done")
    return frames
def save_video(video_frames,video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path,fourcc,24,(video_frames[0].shape[1],video_frames[0].shape[0]))
    i = 1
    video_frames_length = len(video_frames)
    for frame in video_frames:
        print(f"video 1/1 (frame {i}/{video_frames_length}) {video_path}")
        out.write(frame)
        i += 1
    print("Done")
    out.release()
 