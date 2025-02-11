import os
import subprocess

SOURCE_FOLDER = "datasetforcorrection/"
DEST_FOLDER = "datasetforcorrection/"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_all_frames(source_folder, dest_folder):
    """
    For each .mp4 (or .mov, .mkv, .webm) in `source_folder`, extract *every frame*.
    Each frame is saved as an image: <video_name>_0001.jpg, etc.
    """
    ensure_dir(dest_folder)

    for filename in os.listdir(source_folder):
        # Include any video extensions you might have
        if filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
            input_path = os.path.join(source_folder, filename)
            
            # We'll use the base name (no extension) as part of the output file prefix
            base_name = os.path.splitext(filename)[0]
            
            # e.g. dataset/viral_frames/<base_name>_0001.jpg
            output_pattern = os.path.join(dest_folder, f"{base_name}_%04d.jpg")
            
            # Command to extract every frame
            #   -i <input_path> => input video
            #   <output_pattern> => where to store frames
            # We do NOT use `-vf fps=...` because we want *every* frame
            cmd = [
                "ffmpeg",
                "-i", input_path,
                # optional: "-hide_banner",  # to suppress extra ffmpeg info
                output_pattern
            ]
            
            print(f"Extracting ALL frames from: {filename}")
            subprocess.run(cmd, check=True)
            print(f"Done extracting: {filename}\n")

if __name__ == "__main__":
    extract_all_frames(SOURCE_FOLDER, DEST_FOLDER)
    print("All done! Check your datasetforcorrection folder.")

