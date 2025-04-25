from utils import read_video,save_video,load_files
from tracker import Tracker
import cv2


def main():
    video_frames = read_video("129_new.mp4")

    #  # Initialize Tracker
    tracker = Tracker('best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path= r'stubs\track_stubs.pkl')
    
    for track_ids, player in tracks["players"][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]

        cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        cv2.imwrite("output/cropped_image.jpg",cropped_image)
        print("Done")
        break

        
    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    save_video(output_video_frames,"output/output_video.avi")


if __name__ == '__main__':
    main()