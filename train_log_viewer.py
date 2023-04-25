import os
import argparse
import gzip, pickle

from dash import Dash, html, dcc

import torch

from edf.pc_utils import draw_geometry, create_o3d_points, get_plotly_fig
from edf.data import PointCloud, SE3, TargetPoseDemo, DemoSequence, DemoSeqDataset, gzip_load
from edf.preprocess import Rescale, NormalizeColor, Downsample, PointJitter, ColorJitter
from edf.agent import PickAgent, PlaceAgent

app = Dash(__name__)

def main_func(log_dir, show_processed_pcd = True):
    with gzip.open(log_dir, 'rb') as f:
        train_logs = pickle.load(f)
    scene_raw: PointCloud = train_logs['scene_raw']
    grasp_raw: PointCloud = train_logs['grasp_raw']
    scene_proc: PointCloud = train_logs['scene_proc']
    grasp_proc: PointCloud = train_logs['grasp_proc']
    query_points = train_logs['edf_outputs']['query_points']
    query_attention = train_logs['edf_outputs']['query_attention']
    target_pose = SE3(train_logs['target_T'])
    best_pose = SE3(train_logs['best_neg_T'])
    sampled_poses= SE3(train_logs['sampled_Ts'])


    grasp_pl = grasp_raw.plotly(point_size=1.0, name="grasp")
    query_opacity = query_attention ** 1
    query_pl = PointCloud.points_to_plotly(pcd=query_points, point_size=15.0, opacity=query_opacity / query_opacity.max())#, custom_data={'attention': query_attention.cpu()})
    fig_grasp = get_plotly_fig("Grasp")
    fig_grasp = fig_grasp.add_traces([grasp_pl, query_pl])



    target_pcd = PointCloud.merge(scene_raw, grasp_raw.transformed(target_pose)[0])
    target_pl = target_pcd.plotly(point_size=1.0)
    fig_target = get_plotly_fig("Target Placement")
    fig_target = fig_target.add_traces([target_pl])


    if show_processed_pcd:
        revert_color = NormalizeColor(color_mean=torch.tensor([-1., -1., -1.]), color_std=torch.tensor([2., 2., 2.]))
        scene_proc = revert_color(scene_proc)
        grasp_proc = revert_color(grasp_proc)
        best_sample_pcd = PointCloud.merge(scene_proc, grasp_proc.transformed(best_pose)[0])
        best_sample_pl = best_sample_pcd.plotly(point_size=5.0)
    else:
        best_sample_pcd = PointCloud.merge(scene_raw, grasp_raw.transformed(best_pose)[0])
        best_sample_pl = best_sample_pcd.plotly(point_size=1.0)
    sample_pl = PointCloud.points_to_plotly(pcd=sampled_poses.points, point_size=7.0, colors=[0.2, 0.5, 0.8])
    fig_sample = get_plotly_fig("Sampled Placement")
    fig_sample = fig_sample.add_traces([best_sample_pl, sample_pl])

    app.layout = html.Div(children=[
                                    html.Div(children=[
                                                    dcc.Graph(id='target',
                                                            figure=fig_target)
                                    ], style={'padding': 10, 'flex': 1}),

                                    html.Div(children=[
                                                    dcc.Graph(id='grasp',
                                                            figure=fig_grasp)
                                    ], style={'padding': 10, 'flex': 1}),

                                    html.Div(children=[
                                                    dcc.Graph(id='sample',
                                                            figure=fig_sample)
                                    ], style={'padding': 10, 'flex': 1})
    ], style={'display': 'flex', 'flex-direction': 'row'})



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization webserver for EDF place training')
    parser.add_argument('--logdir', type=str,
                        help='')
    args = parser.parse_args()
    
    log_dir = args.logdir
    main_func(log_dir=log_dir)

    app.run_server(debug=True, host='127.0.0.1')