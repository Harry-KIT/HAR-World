import os
import argparse
import torch
import pickle
from pathlib import Path
from tqdm import tqdm  # Progress bar
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized
import numpy as np

def save_keypoints_as_pkl(keypoints, video_name, save_dir):
    pkl_file = os.path.join(save_dir, f"{video_name}.pkl")
    with open(pkl_file, 'wb') as f:
        pickle.dump(keypoints, f)

def detect(opt):
    source, weights, imgsz, kpt_label = opt.source, opt.weights, opt.img_size, opt.kpt_label

    save_dir = Path(opt.project)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    keypoints_data = []
    for path, img, im0s, vid_cap in tqdm(dataset, desc="Processing video frames"):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
        t2 = time_synchronized()

        frame_keypoints = []
        for i, det in enumerate(pred):
            p, im0 = path, im0s.copy()
            p = Path(p)
            video_name = p.stem

            if len(det):
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                # Process each detected person and extract keypoints
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    kpts = del_tensor_ele(det[det_index, 6:], 3, 15).reshape(-1, 3)
                    # kpts = det[det_index, 6:].reshape(-1, 3)

                    frame_keypoints.append(kpts)

        # Add keypoints for this frame
        keypoints_data.append(frame_keypoints)

    # Normalize keypoints
    normalized_keypoints_data = []
    for frame_keypoints in keypoints_data:
        normalized_frame_keypoints = []
        for kpts in frame_keypoints:
            if isinstance(kpts, torch.Tensor):
                kpts = kpts.cpu()  # Move to CPU
            xy = kpts[:, :2]  # Extract x, y coordinates
            scores = kpts[:, 2]  # Extract scores
            xy = normalize_points_with_size(xy, im0s.shape[1], im0s.shape[0])
            xy = scale_pose(xy)
            kpts[:, :2] = torch.tensor(xy)  # Convert back to PyTorch tensor and update x, y coordinates
            normalized_frame_keypoints.append(kpts)
        normalized_keypoints_data.append(normalized_frame_keypoints)

    # Save keypoints in a .pkl file
    save_keypoints_as_pkl(normalized_keypoints_data, video_name, save_dir)
    print(f"Keypoints for video {video_name} saved to {save_dir}.")


def process_dataset(video_root, save_root):
    action_classes = [d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))]

    for action_class in tqdm(action_classes, desc="Processing action classes"):
        action_class_path = os.path.join(video_root, action_class)

        save_class_path = os.path.join(save_root, action_class)
        os.makedirs(save_class_path, exist_ok=True)

        video_files = [f for f in os.listdir(action_class_path) if f.endswith('.mp4')]
        # video_files = [f for f in os.listdir(action_class_path) if f.endswith('.avi')]


        for video_file in tqdm(video_files, desc=f"Processing videos in {action_class}"):
            video_path = os.path.join(action_class_path, video_file)
            print(f"Processing video: {video_path}")

            opt = argparse.Namespace(
                weights='weights/yolov7-w6-pose.pt',
                source=video_path,
                img_size=960,
                conf_thres=0.25,
                iou_thres=0.45,
                device='',
                nosave=True,
                project=save_class_path,  # Save directly in class folder
                name=video_file.split('.')[0],
                exist_ok=True,
                kpt_label=True,
                classes=None,  # Add default value for classes
                agnostic_nms=False  # Add default value for agnostic_nms
            )

            with torch.no_grad():
                detect(opt)

# kpts = del_tensor_ele(kpts, 3, 15)
def del_tensor_ele(arr, index_a, index_b):
    arr1 = arr[0:index_a]
    arr2 = arr[index_b:]
    return torch.cat((arr1, arr2), dim=0)

def normalize_points_with_size(xy, width, height, flip=False):
    """Normalize scale points in image with size of image to (0-1).
    xy : (frames, parts, xy) or (parts, xy)
    """
    if isinstance(xy, torch.Tensor):
        xy = xy.cpu().numpy()  # Move to CPU and convert to NumPy array
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy[:, :, 0] /= width
    xy[:, :, 1] /= height
    if flip:
        xy[:, :, 0] = 1 - xy[:, :, 0]
    return xy


def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if isinstance(xy, torch.Tensor):
        xy = xy.cpu().numpy()  # Move to CPU and convert to NumPy array
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


if __name__ == '__main__':
    
    video_root = '../video_folder/'  # Path to your dataset
    save_root = '../pickle_folder/'  # Path where .pkl files will be saved
    process_dataset(video_root, save_root)
