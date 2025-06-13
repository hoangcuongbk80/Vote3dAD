# python newly_gt.py --dataset_dir /home/cuong/Datasets/AD3D/Newly --class_name frasvaf --index 0

import os
import argparse
import numpy as np
import open3d as o3d


def load_point_cloud_and_mask(pcd_path, gt_dir):
    """
    Loads a .ply file and corresponding anomaly ground truth from .txt (if any).
    """
    filename = os.path.basename(pcd_path)
    name, _ = os.path.splitext(filename)

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    txt_path = os.path.join(gt_dir, name + '.txt')

    if os.path.isfile(txt_path):
        # Anomalous sample
        anomaly_indices = np.loadtxt(txt_path, dtype=int)
        mask = np.zeros(points.shape[0], dtype=np.uint8)
        mask[anomaly_indices] = 1
    else:
        # Normal (prototype) sample
        mask = np.zeros(points.shape[0], dtype=np.uint8)

    return points, mask


def visualize(points, mask, point_size=2.0):
    """
    Visualizes the point cloud with red color on anomalous points.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((points.shape[0], 3), dtype=float)
    colors[mask == 0] = [0.5, 0.5, 0.5]  # gray
    colors[mask == 1] = [1.0, 0.0, 0.0]  # red
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Newly Dataset Viewer", width=800, height=600)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    #opt.background_color = np.array([1.0, 1.0, 1.0])  # white background

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize a Newly dataset test sample with anomalies highlighted."
    )
    parser.add_argument(
        '--dataset_dir', type=str, required=True,
        help="Root of Newly dataset, e.g. /home/cuong/Datasets/AD3D/Newly"
    )
    parser.add_argument(
        '--class_name', type=str, required=True,
        help="Object category, e.g. 'airplane'"
    )
    parser.add_argument(
        '--index', type=int, default=0,
        help="Which scan in the test folder to visualize (0-based)"
    )
    parser.add_argument(
        '--point_size', type=float, default=2.0,
        help="Point size in the visualization"
    )
    args = parser.parse_args()

    test_dir = os.path.join(args.dataset_dir, args.class_name, 'test')
    gt_dir   = os.path.join(args.dataset_dir, args.class_name, 'gt')

    pcd_files = sorted(f for f in os.listdir(test_dir) if f.endswith('.ply'))
    if not (0 <= args.index < len(pcd_files)):
        raise IndexError(f"Index out of range: there are {len(pcd_files)} scans.")

    pcd_path = os.path.join(test_dir, pcd_files[args.index])
    print(f"Visualizing: {pcd_path}")

    points, mask = load_point_cloud_and_mask(pcd_path, gt_dir)
    visualize(points, mask, point_size=args.point_size)
