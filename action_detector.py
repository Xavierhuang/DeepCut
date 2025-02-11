import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import Counter

class ActionDetector:
    def __init__(self):
        weights = R3D_18_Weights.KINETICS400_V1
        self.model = r3d_18(weights=weights)
        self.model.eval()
        
        # Get class mapping from weights
        self.classes = weights.meta["categories"]
        
        # Movie-specific action mappings
        self.movie_actions = {
            "dramatic_scene": [
                "talking", "crying", "shouting", "arguing", "presenting", 
                "public speaking", "acting", "reading", "singing"
            ],
            "action_scene": [
                "running", "jumping", "falling", "fighting", "chasing",
                "sword fighting", "punching", "kicking", "wrestling"
            ],
            "emotional_moment": [
                "hugging", "kissing", "laughing", "crying", "embracing",
                "shaking hands", "waving", "pointing", "nodding"
            ],
            "suspense": [
                "walking", "sneaking", "crawling", "tiptoeing", "looking",
                "searching", "waiting", "listening", "watching"
            ],
            "transformation": [
                "stretching", "shaking", "twitching", "falling", "turning",
                "spinning", "bending", "writhing", "convulsing"
            ],
            "ritual_scene": [
                "kneeling", "praying", "bowing", "meditating", "dancing",
                "raising arms", "standing", "sitting", "lying"
            ]
        }
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
    def load_video(self, video_path, target_frames=16):
        """Load video and sample frames more intelligently"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample evenly across the video
        frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112))
                frames.append(frame)
        cap.release()
        
        # Handle if we couldn't get enough frames
        while len(frames) < target_frames:
            frames.append(frames[-1] if frames else np.zeros((112, 112, 3)))
            
        return np.array(frames) / 255.0
                
    def map_to_movie_action(self, action_name, confidence):
        """Map Kinetics action to movie scene type"""
        action_lower = action_name.lower()
        best_match = ("other", confidence)
        max_score = 0
        
        for scene_type, related_actions in self.movie_actions.items():
            # Calculate how well this action matches this scene type
            matches = sum(1 for act in related_actions if act in action_lower)
            if matches > 0:
                # Consider both matches and confidence
                scene_score = confidence * (1.0 + 0.2 * matches)
                if scene_score > max_score:
                    best_match = (scene_type, scene_score)
                    max_score = scene_score
        
        return best_match
    
    def detect_action(self, video_path):
        """Detect action with temporal consistency"""
        with torch.no_grad():
            all_frames = self.load_video(video_path, target_frames=32)
            
            window_size = 16
            stride = 8
            scene_scores = {scene_type: [] for scene_type in self.movie_actions.keys()}
            original_actions_list = []
            
            for i in range(0, len(all_frames) - window_size + 1, stride):
                window = all_frames[i:i+window_size]
                frames = torch.FloatTensor(window).permute(3, 0, 1, 2).unsqueeze(0)
                
                if torch.cuda.is_available():
                    frames = frames.cuda()
                    
                outputs = self.model(frames)
                probs = torch.softmax(outputs, dim=1)
                
                # Get top predictions for this window
                top_probs, top_idx = torch.topk(probs[0], k=5)
                
                # Track scene scores over time
                window_scores = {scene_type: 0.0 for scene_type in self.movie_actions.keys()}
                
                for idx, prob in zip(top_idx, top_probs):
                    kinetics_action = self.classes[idx.item()]
                    scene_type, score = self.map_to_movie_action(kinetics_action, prob.item())
                    if scene_type != "other":  # Only track non-other scenes
                        window_scores[scene_type] = max(window_scores[scene_type], score)
                    original_actions_list.append((kinetics_action, prob.item()))
                
                # Add window scores to temporal tracking
                for scene_type, score in window_scores.items():
                    scene_scores[scene_type].append(score)
            
            # Calculate final scene scores with temporal consistency
            final_scores = {}
            for scene_type, scores in scene_scores.items():
                if scores:
                    # Consider both average and peak scores
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    final_scores[scene_type] = (avg_score + max_score) / 2
            
            # Get most frequent original actions
            original_counter = Counter(act for act, _ in original_actions_list)
            top_original = sorted([(act, count) for act, count in original_counter.items()], 
                                key=lambda x: x[1], reverse=True)[:5]
            
            # Sort and filter final scores
            final_actions = [(scene, score) for scene, score in final_scores.items() 
                            if score > 0.1]  # Filter low confidence scenes
            final_actions.sort(key=lambda x: x[1], reverse=True)
            
            if not final_actions:  # If no confident scene types found
                final_actions = [("unclassified", 0.0)]
            
            return final_actions[:3], top_original

def main():
    print("Initializing action detector...")
    detector = ActionDetector()
    
    clips_dir = "clips"
    if not os.path.exists(clips_dir):
        print(f"Error: Clips directory not found at {clips_dir}")
        return
        
    clips = sorted([f for f in os.listdir(clips_dir) if f.endswith('.mp4')])
    print(f"\nFound {len(clips)} clips to analyze")
    
    with open("action_results.txt", "w") as f:
        for clip in clips:
            clip_path = os.path.join(clips_dir, clip)
            print(f"\n{'='*50}")
            print(f"Processing {clip}...")
            try:
                movie_actions, original_actions = detector.detect_action(clip_path)
                
                result = f"\nClip: {clip}\n"
                result += "Scene types detected:\n"
                print(result)
                f.write(result)
                
                if movie_actions[0][0] == "unclassified":
                    line = "No confident scene type detected"
                    print(line)
                    f.write(line + "\n")
                else:
                    for scene_type, conf in movie_actions:
                        line = f"- {scene_type:15} (confidence: {conf:.2f})"
                        print(line)
                        f.write(line + "\n")
                
                result = "\nRaw actions detected:\n"
                print(result)
                f.write(result)
                
                for action, count in original_actions:
                    line = f"- {action:30} (detected {count} times)"
                    print(line)
                    f.write(line + "\n")
                    
            except Exception as e:
                error = f"Error processing {clip}: {str(e)}"
                print(error)
                f.write(error + "\n")

if __name__ == "__main__":
    main() 