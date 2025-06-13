# python newly_multi_gt.py --dataset_dir /home/cuong/Datasets/AD3D/Newly --class_name frasvaf --index 0

import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def load_point_cloud_and_group_mask(pcd_path, gt_dir):
    """
    Loads .ply point cloud and corresponding ground truth anomaly group indices.
    Returns: points, group_labels (np.array with int group index per point, -1 if normal)
    """
    filename = os.path.basename(pcd_path)
    name, _ = os.path.splitext(filename)

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    txt_path = os.path.join(gt_dir, name + '.txt')

    group_labels = np.full(len(points), -1, dtype=int)  # -1 means normal

    if os.path.isfile(txt_path):
        with open(txt_path, 'r') as f:
            for group_id, line in enumerate(f):
                if line.strip() == '':
                    continue
                idxs = list(map(int, line.strip().split()))
                group_labels[idxs] = group_id

    return points, group_labels


def generate_color_palette(n):
    """
    Returns n distinct RGB colors using matplotlib's colormap.
    """
    cmap = plt.get_cmap('tab20', n)
    return [cmap(i)[:3] for i in range(n)]


def visualize(points, group_labels, point_size=3.0):
    """
    Visualizes point cloud with group-wise colored anomalies.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    n_groups = group_labels.max() + 1
    colors = np.full((points.shape[0], 3), 0.5)  # gray for normal
    if n_groups > 100:
        palette = generate_color_palette(n_groups)
        for gid in range(n_groups):
            #colors[group_labels == gid] = palette[gid]
            colors[group_labels == gid] = np.array([1.0, 0.0, 0.0]) #

    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Newly Dataset Viewer", width=800, height=600)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
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
        help="Object category, e.g. 'frasvaf'"
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

    points, group_labels = load_point_cloud_and_group_mask(pcd_path, gt_dir)
    visualize(points, group_labels, point_size=args.point_size)
