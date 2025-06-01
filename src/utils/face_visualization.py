import cv2
import numpy as np
from pathlib import Path
import os
import imageio
import time
import uuid
import tempfile

def visualize_face_analysis(image_path, backend, output_dir=None):
    """
    Create visualizations of face analysis steps
    
    Args:
        image_path: Path to the input image
        backend: InsightFace backend instance
        output_dir: Directory to save visualization images (optional)
        
    Returns:
        List of (image, caption) tuples for visualization
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        return []
        
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = img_rgb.copy()
    
    # Get face analysis from InsightFace
    try:
        faces = backend.model.get(img)
        if not faces:
            return [(img_rgb, "No faces detected")]
    except Exception as e:
        return [(img_rgb, f"Error: {str(e)}")]
    
    # Store visualization images and captions
    viz_results = []
    viz_results.append((orig_img, "Original Image"))
    
    # 1. Bounding Box
    box_img = orig_img.copy()
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(box_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        
        # Add center point
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        cv2.circle(box_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add dimensions text
        width = box[2] - box[0]
        height = box[3] - box[1]
        cv2.putText(box_img, f"w: {width}", (box[0], box[3] + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(box_img, f"h: {height}", (box[0], box[3] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    viz_results.append((box_img, "Box Detection (4 scalars)"))
    
    # 2. 5-Point Landmarks
    landmarks_img = orig_img.copy()
    for face in faces:
        landmarks = face.kps.astype(int)
        for point in landmarks:
            cv2.circle(landmarks_img, tuple(point), 2, (255, 0, 0), -1)
    
    viz_results.append((landmarks_img, "5 Landmarks (10 scalars)"))
    
    # 3. Pose (if available)
    if hasattr(face, 'pose'):
        pose_img = orig_img.copy()
        for face in faces:
            box = face.bbox.astype(int)
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            
            # Draw pose arrows from angle
            yaw, pitch, roll = face.pose
            
            # Convert to radians
            yaw = yaw * np.pi / 180
            pitch = pitch * np.pi / 180
            
            # Draw arrows for yaw (horizontal) and pitch (vertical)
            arrow_length = 50
            dx = int(np.sin(yaw) * arrow_length)
            dy = int(-np.sin(pitch) * arrow_length)
            
            cv2.arrowedLine(pose_img, (center_x, center_y), 
                          (center_x + dx, center_y + dy), 
                          (255, 0, 0), 2)  # Yaw (horizontal/blue)
            cv2.arrowedLine(pose_img, (center_x, center_y), 
                          (center_x, center_y + dy), 
                          (0, 0, 255), 2)  # Pitch (vertical/red)
            
            # Add text for the angles
            cv2.putText(pose_img, f"Yaw: {yaw:.1f}", (box[0], box[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(pose_img, f"Pitch: {pitch:.1f}", (box[0], box[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(pose_img, f"Roll: {roll:.1f}", (box[0], box[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        viz_results.append((pose_img, "Pose Estimation (7 scalars)"))
    
    # 4. Face Mask (if available - most models don't have this)
    if hasattr(face, 'mask') and face.mask is not None:
        mask_img = orig_img.copy()
        # This depends on the specific format provided by InsightFace
        # Basic implementation assuming mask is a binary matrix
        for face in faces:
            if face.mask is not None:
                mask = cv2.resize(face.mask, 
                               (int(face.bbox[2] - face.bbox[0]), 
                                int(face.bbox[3] - face.bbox[1])))
                x, y = int(face.bbox[0]), int(face.bbox[1])
                
                # Create a colored overlay
                overlay = mask_img.copy()
                blue_mask = np.zeros_like(overlay)
                blue_mask[y:y+mask.shape[0], x:x+mask.shape[1], 0] = mask * 128
                
                # Blend the overlay
                cv2.addWeighted(blue_mask, 0.5, mask_img, 1, 0, mask_img)
        
        viz_results.append((mask_img, "Face Mask"))
    
    # 5. Generate combined visualization (box + landmarks + pose)
    combined_img = orig_img.copy()
    for face in faces:
        # Draw box
        box = face.bbox.astype(int)
        cv2.rectangle(combined_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0, 128), 1)
        
        # Draw 5-pt landmarks
        landmarks = face.kps.astype(int)
        for point in landmarks:
            cv2.circle(combined_img, tuple(point), 2, (0, 255, 255), -1)
            
        # If pose available, draw simplified pose indicator
        if hasattr(face, 'pose'):
            yaw, pitch, roll = face.pose
            yaw = yaw * np.pi / 180
            pitch = pitch * np.pi / 180
            
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            arrow_length = 30
            
            dx = int(np.sin(yaw) * arrow_length)
            dy = int(-np.sin(pitch) * arrow_length)
            
            cv2.arrowedLine(combined_img, (center_x, center_y), 
                          (center_x + dx, center_y + dy), 
                          (0, 255, 0), 2)
        
    viz_results.append((combined_img, "Combined Analysis"))
    
    # Save images if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save individual images
        for i, (img, caption) in enumerate(viz_results):
            safe_caption = caption.replace(" ", "_").replace("(", "").replace(")", "")
            cv2.imwrite(str(output_dir / f"{i:02d}_{safe_caption}.jpg"), 
                       cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return viz_results

def create_analysis_animation(viz_results, output_path=None):
    """Create a GIF animation from visualization results"""
    if not viz_results:
        return None
        
    # Create a copy of frames to modify
    frames = []
    for img, _ in viz_results:
        img_copy = img.copy()
        # Add a timestamp directly on the image
        timestamp = str(int(time.time()))
        cv2.putText(img_copy, timestamp, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 
                   cv2.LINE_AA)
        frames.append(img_copy)
    
    if output_path:
        # Create a completely random path each time
        import uuid
        random_id = str(uuid.uuid4())
        random_dir = os.path.join(tempfile.gettempdir(), f"insightface_viz_{random_id}")
        os.makedirs(random_dir, exist_ok=True)
        
        # Use random filename in random directory
        new_output_path = os.path.join(random_dir, f"analysis_{random_id}.gif")
        
        # Save with slow duration
        imageio.mimsave(
            new_output_path, 
            frames, 
            format='GIF',
            duration=250.0,
            loop=0
        )
        return new_output_path
    return output_path

def combine_animations(animation_paths, output_path):
    """Combines multiple GIF animations side by side into one"""
    if not animation_paths or len(animation_paths) == 0:
        return None
        
    # If just one animation, return it directly
    if len(animation_paths) == 1:
        return animation_paths[0]
        
    # Load all animations
    animations = []
    for path in animation_paths:
        try:
            frames = imageio.mimread(path)
            animations.append(frames)
        except Exception as e:
            print(f"Error loading animation {path}: {e}")
    
    if not animations:
        return None
    
    # Get the frame count (use the shortest animation)
    min_frames = min(len(anim) for anim in animations)
    
    # Get dimensions
    heights = [anim[0].shape[0] for anim in animations]
    widths = [anim[0].shape[1] for anim in animations]
    max_height = max(heights)
    total_width = sum(widths)
    
    # Create combined frames
    combined_frames = []
    for i in range(min_frames):
        # Create blank canvas
        combined = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
        
        # Add each animation frame
        x_offset = 0
        for j, anim in enumerate(animations):
            h, w = anim[i].shape[:2]
            # Center vertically
            y_offset = (max_height - h) // 2
            combined[y_offset:y_offset+h, x_offset:x_offset+w] = anim[i]
            x_offset += w
            
        combined_frames.append(combined)
    
    # Save combined animation
    imageio.mimsave(output_path, combined_frames, format='GIF', duration=10.0, loop=0)
    return output_path