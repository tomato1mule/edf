from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.lines as mlines
import numpy as np

import torch
from torch_cluster import radius_graph, radius
from pytorch3d import transforms



default_transform = {'X':np.zeros(3), 
                     'R':np.eye(3)}
rotated_transform = {'X':np.array([0., 0., 0.]), 
                     'R':np.array([[0, -1, 0],
                                   [1, 0, 0],
                                   [0, 0, 1.]]
                                 )}
 


def plot_color_and_depth(data, axs = None):
    if axs is None:
        fig, axs = plt.subplots(2,len(data), figsize=(5*len(data),5*2))
    for i in range(len(data)):
        axs[0,i].imshow(data[i]['color'])
    for i in range(len(data)):
        axs[1,i].imshow(data[i]['depth'])

    return axs

def append_alpha(color):
    assert len(color.shape) == 2
    if color.shape[-1] == 3:
        alpha = np.ones((len(color),1),dtype=float)
        color = np.concatenate((color,alpha), axis=-1)
    elif color.shape[-1] == 4:
        pass
    else:
        raise ValueError
    return color

def cat_pointclouds(data):
    coords = []
    colors = []
    for coord, color in data:
        coords.append(coord)
        colors.append(append_alpha(color))
    coords = np.concatenate(coords, axis=0)
    colors = np.concatenate(colors, axis=0)

    return coords, colors

def draw_poincloud_line(begin, end, color, N=100):
    xmin, ymin, zmin = begin[0], begin[1], begin[2]
    xmax ,ymax, zmax = end[0], end[1], end[2]
    N = 100
    X,Y,Z = np.linspace(xmin,xmax,N), np.linspace(ymin,ymax,N), np.linspace(zmin,zmax,N)
    coord_line = np.stack([X,Y,Z]).T # (100,3)
    color_line = np.repeat(color.reshape(1,-1), N, axis = 0)

    return coord_line, color_line

def draw_poincloud_arrow(begin, end, color, arrowhead_size = 1., N=50, view_normal = None):
    c1, d1 = draw_poincloud_line(begin, end, color, N=N)
    dir = end-begin
    scale = np.linalg.norm(dir) + 1e-5
    if view_normal is None:
        normal = np.array([1., 1., -1.]) /np.sqrt(3)
    else:
        normal = view_normal
    perp1 = np.cross(dir / scale, normal)
    perp2 = np.cross(dir / scale, -normal)
    arr1 = (2*perp1 - dir) * 0.1 * arrowhead_size
    arr2 = (2*perp2 - dir) * 0.1 * arrowhead_size
    c2, d2 = draw_poincloud_line(end, end + arr1, color, N=int(N * np.linalg.norm(arr1) / (scale+0.1)) + 1 )
    c3, d3 = draw_poincloud_line(end, end + arr2, color, N=int(N * np.linalg.norm(arr2) / (scale+0.1)) + 1 )

    coord_line = np.concatenate([c1,c2,c3], axis=0)
    color_line = np.concatenate([d1,d2,d3], axis=0)
    
    return coord_line, color_line

def draw_poincloud_arrow_new(begin, end, color, arrowhead_size = 1., density=10, view_normal = None):
    dir = end-begin
    scale = np.linalg.norm(dir) + 1e-5
    c1, d1 = draw_poincloud_line(begin, end, color, N=int(scale*density)+1)

    if view_normal is None:
        normal = np.array([1., 1., -1.]) /np.sqrt(3)
    else:
        normal = view_normal
    perp1 = np.cross(dir / scale, normal)
    perp2 = np.cross(dir / scale, -normal)
    arr1 = (2*perp1 - dir)
    arr2 = (2*perp2 - dir)
    arr1 = arr1 / np.linalg.norm(arr1) * arrowhead_size
    arr2 = arr2 / np.linalg.norm(arr2) * arrowhead_size

    c2, d2 = draw_poincloud_line(end, end + arr1, color, N=int(density*arrowhead_size) + 1 )
    c3, d3 = draw_poincloud_line(end, end + arr2, color, N=int(density*arrowhead_size) + 1 )

    coord_line = np.concatenate([c1,c2,c3], axis=0)
    color_line = np.concatenate([d1,d2,d3], axis=0)
    
    return coord_line, color_line


def draw_frame(ax, frame, origin, scale=1., alpha = 1., pointcloud = None):

    X = frame[:,0]*scale + origin
    Y = frame[:,1]*scale + origin
    Z = frame[:,2]*scale + origin

    X = draw_poincloud_line(origin, X, np.array([1.,0.,0.,alpha]))
    Y = draw_poincloud_line(origin, Y, np.array([0.,1.,0.,alpha]))
    Z = draw_poincloud_line(origin, Z, np.array([0.,0.,1.,alpha]))

    coord_frame, color_frame = cat_pointclouds([X,Y,Z])
    if pointcloud is not None:
        coord_frame, color_frame = cat_pointclouds([(coord_frame, color_frame), pointcloud])

    return coord_frame, color_frame


def scatter_plot_ax(ax, coord, color, ranges, frame_infos = [], transform = 'rotate'): # frame_infos =[frame_info, ...],   frame_info = (frame, origin, alpha)
    transform_type = transform
    if transform == 'rotate':
        transform = rotated_transform
    elif transform == 'default':
        transform = default_transform
    else:
        raise KeyError

    ranges = transform['R'] @ ranges + transform['X'].reshape(3,1)
    xlim, ylim, zlim = ranges[0], ranges[1], ranges[2]
    if xlim[0] > xlim[1]:
        xlim = [xlim[1], xlim[0]]
    if ylim[0] > ylim[1]:
        ylim = [ylim[1], ylim[0]]
    if zlim[0] > zlim[1]:
        zlim = [zlim[1], zlim[0]]


    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    #print(xlim, ylim, zlim)
    #ax.set_box_aspect(aspect = (1,1,1))

    if transform_type == 'rotate':
        ax.set_xlabel('-Y')
        ax.set_ylabel('X')
    elif transform_type == 'default':
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        raise KeyError
    ax.set_zlabel('Z')

    color = append_alpha(color)

    frames = []
    for frame_info in frame_infos:
        frame = frame_info['frame']
        origin = frame_info['origin']
        alpha = frame_info['alpha'] # 0~1
        try:
            scale = frame_info['scale']
        except KeyError:
            scale = (xlim[1]-xlim[0]) * 0.3
        try:
            pointcloud = frame_info['pointcloud']
        except KeyError:
            pointcloud = None
        coord_frame, color_frame = draw_frame(ax, frame, origin, scale, alpha, pointcloud)
        frames.append((coord_frame, color_frame))

    coord, color = cat_pointclouds([(coord, color), *frames])
    coord = coord @ (transform['R'].T) + transform['X'].reshape(1,3)
    ax.scatter(coord[:,0],coord[:,1],coord[:,2], c=color)
    

def scatter_plot(coord, color, ranges, frame_infos = [],
                 figsize = (8, 8), transform = 'rotate'):
    fig = plt.figure(figsize=figsize)                                      
    ax = fig.add_subplot(projection='3d')

    scatter_plot_ax(ax=ax, coord=coord, color=color, ranges=ranges, frame_infos=frame_infos, transform=transform)
    plt.show();


def visualize_samples_key(samples, coord_key='coord', color_key='color', range_key='range', grasp_key=None, fig_W=8):
    sample_indices = np.arange(len(samples))
    sample_len = len(sample_indices)
    columns = 4
    rows = (sample_len-1) // columns + 1

    fig, axes = plt.subplots(rows, columns, figsize=(fig_W*columns,fig_W*rows), subplot_kw={'projection':'3d'})

    for i, sample_idx in enumerate(sample_indices):
        r,c = i//columns, i%columns
        sample = samples[sample_idx]
        coord, color = sample[coord_key], sample[color_key]
        ranges = sample[range_key]
        if grasp_key is not None:
            X_sg, R_sg = sample[grasp_key]
            frame_info = [{'frame':R_sg, 'origin':X_sg, 'alpha':0.5}]
        else:
            frame_info = []
        
        if rows == 1:
            scatter_plot_ax(axes[c], coord, color, ranges, frame_info)
        else:
            scatter_plot_ax(axes[r,c], coord, color, ranges, frame_info)

def visualize_samples(samples, fig_W=8):
    visualize_samples_key(samples, coord_key='coord', color_key='color', range_key='range', grasp_key='grasp', fig_W=fig_W)
    visualize_samples_key(samples, coord_key='coord_pick', color_key='color_pick', range_key='range_pick', grasp_key=None, fig_W=fig_W)
    visualize_samples_key(samples, coord_key='coord_place', color_key='color_place', range_key='range_place', grasp_key='place', fig_W=fig_W)


def add_gaussian_blob(pointcloud, pos, std, N=100):
    coord = pointcloud[0].copy()
    color = pointcloud[1].copy()
    blob_pos = np.random.randn(N,3) * std + pos.reshape(1,3)
    blob_color = np.repeat(np.array([[0, 1., 0]]), N, axis=0)
    coord = np.concatenate((coord, blob_pos), axis=0)
    color = np.concatenate((color, blob_color), axis=0)

    return coord, color

def visualize_cluster(coord, color, ranges, grasp, max_radius, figsize=8):
    X_sdg, R_sdg = grasp

    num_nodes = coord.shape[0]
    edge_src, edge_dst = radius_graph(torch.tensor(coord), max_radius, max_num_neighbors=num_nodes - 1)
    closest_node = np.argmin(np.linalg.norm(coord - X_sdg, axis=-1))
    red_node = np.argmax(color[:,0])
    green_node = np.argmax(color[:,1])

    src_nodes_to_visualize = [closest_node, red_node, green_node, 123]
    fig, axes = plt.subplots(1,len(src_nodes_to_visualize), figsize=(figsize * len(src_nodes_to_visualize),figsize), subplot_kw={'projection':'3d'})

    for i, src_node in enumerate(src_nodes_to_visualize):
        neighbor = edge_dst[(edge_src == src_node).nonzero().squeeze(1)]

        color_neighbor = color.copy()

        color_neighbor[neighbor] = np.repeat(np.array([[0,0,1.]]), len(neighbor), axis = 0)

        scatter_plot_ax(axes[i], *add_gaussian_blob((coord,color_neighbor), coord[src_node], 0.001, 100), ranges)
    print(f"Number of Edges: {len(edge_src)}")

def visualize_sample_cluster(sample, max_radius = 0.025, max_radius_pick = None, max_radius_place = None, figsize=8):
    visualize_cluster(sample['coord'], sample['color'], sample['range'], sample['grasp'], max_radius=max_radius, figsize=figsize)
    if max_radius_pick is not None:
        grasp = (np.zeros(3), np.eye(3))
        visualize_cluster(sample['coord_pick'], sample['color_pick'], sample['range_pick'], grasp, max_radius=max_radius_pick, figsize=figsize)
    if max_radius_place is not None:
        visualize_cluster(sample['coord_place'], sample['color_place'], sample['range_place'], sample['place'], max_radius=max_radius_place, figsize=figsize)