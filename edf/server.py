from typing import Union, List, Optional, Dict, Iterable
import threading
import time
from datetime import datetime

import dash
from dash import Dash, html, dcc, Input, Output
import dash_vtk
import dash_daq
from flask import request

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from edf.data import SE3, PointCloud
from edf.env_interface import RESET, FEASIBLE, INFEASIBLE


DEFAULT_VTK_INTERACTION_SETTING=[
  {
    'button': 1,
    'action': 'Rotate',
  }, {
    'button': 2,
    'action': 'Pan',
  }, {
    'button': 3,
    'action': 'Zoom',
    'scrollEnabled': True,
  }, {
    'button': 1,
    'action': 'Pan',
    'shift': True,
  }, {
    'button': 1,
    'action': 'Zoom',
    'alt': True,
  }, {
    'button': 1,
    'action': 'Pan', #'ZoomToMouse',
    'control': True,
  }, {
    'button': 1,
    'action': 'Roll',
    'alt': True,
    'shift': True,
  }
]

# https://vtk.org/doc/nightly/html/classvtkProp3D.html#a1c44f66f6ce311d9f38b9e1223f9cee5
# Orientation is specified as X,Y and Z rotations in that order, but they are performed as RotateZ, RotateX, and finally RotateY.
def _get_pcd_repr(id: str, points: Union[torch.Tensor, np.ndarray], colors: Union[torch.Tensor, np.ndarray], point_size: float):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu()
    return dash_vtk.GeometryRepresentation(id=id, 
                                           property={"pointSize": point_size},
                                           children=[dash_vtk.PolyData(id = id + '-poly', 
                                                                       points=points.ravel(), 
                                                                       connectivity='points', 
                                                                       children=[dash_vtk.PointData([dash_vtk.DataArray(id = id + '-array',
                                                                                                                        registration='setScalars',
                                                                                                                        type='Uint8Array',
                                                                                                                        numberOfComponents=3,
                                                                                                                        values=colors.ravel() * 255)])],)],)

def _get_range_repr(id: str, ranges: Union[torch.Tensor, np.ndarray, List[List[float]]], color: Iterable = [0.2, 0.3, 0.4], linewidth: float = 4.):
    if isinstance(ranges, torch.Tensor):
        ranges = ranges.detach().cpu()
    if isinstance(ranges, List):
        assert len(ranges) == 3
    else:
        assert ranges.shape[-1] == 2 and ranges.shape[-2] == 3
    
    x, y, z = ranges[0][0], ranges[1][0], ranges[2][0]
    X, Y, Z = ranges[0][1], ranges[1][1], ranges[2][1]

    return dash_vtk.GeometryRepresentation(id=id, 
                                           property={"color": color, "lineWidth": linewidth, 'lighting': False},
                                           children=[dash_vtk.PolyData(id = id + '-poly', 
                                                                       points=[X,Y,Z,
                                                                               X,Y,z,
                                                                               X,y,Z,
                                                                               X,y,z,
                                                                               x,Y,Z,
                                                                               x,Y,z,
                                                                               x,y,Z,
                                                                               x,y,z], 
                                                                       lines=[2,0,1,
                                                                              2,0,2,
                                                                              2,0,4,
                                                                              2,1,3,
                                                                              2,1,5,
                                                                              2,2,3,
                                                                              2,2,6,
                                                                              2,3,7,
                                                                              2,4,6,
                                                                              2,4,5,
                                                                              2,5,7,
                                                                              2,6,7],
                                                                       )],)


class DashEdfDemoServer():
    def __init__(self, scene_ranges: Union[np.ndarray, torch.Tensor, List[List[float]]],
                 name: str = "dash_edf_demo_server", 
                 point_size: float = 3., 
                 n_trans_step: int = 100, 
                 n_rot_step: int = 100, 
                 slider_size_px: int = 500,
                 vtk_interaction_setting: Optional[List[Dict]] = None,
                 host_id: str = '127.0.0.1',
                 server_debug: bool = False,            # Dash debug option does not support threading.
                 pcd_update_check_interval: float = 1. # in seconds
                 ):
        
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        self._request_response_flag: bool = False
        self._user_response_ready: bool = False
        self._user_response: Optional[Union[SE3, str]] = None
        self._scene_pcd_updated: bool = False # False
        self._grasp_pcd_updated: bool = False # False
        self.pcd_update_check_interval_ms: int = int(pcd_update_check_interval * 1000) + 1
        self.robot_state_msg_update_interval_ms: int = 500
        self.robot_state_msg: str = "Offline"
        self._robot_state_msg_update_flag: bool = False

        self.scene_ranges: Union[np.ndarray, torch.Tensor, List[List[float]]] = scene_ranges
        self.name: str = name
        self.point_size: float = point_size
        self.n_trans_step: int = n_trans_step
        self.n_rot_step: int = n_rot_step
        self.slider_size_px: int = slider_size_px
        self.vtk_interaction_setting: List[Dict] = DEFAULT_VTK_INTERACTION_SETTING if vtk_interaction_setting is None else vtk_interaction_setting
        self.host_id: str = host_id
        self.server_debug: bool = server_debug
        self.scene_pcd_id = 'scene-pcd'
        self.grasp_pcd_id = 'grasp-pcd'

        #### CENTER PANEL ####
        self._target_pose: SE3 = SE3.empty()
        self.scene_pcd: PointCloud = PointCloud.empty()
        self.grasp_pcd: PointCloud = PointCloud.empty()
        self.scene_repr = self._get_scene_repr(id = self.scene_pcd_id)
        self.grasp_repr = self._get_grasp_repr(id = self.grasp_pcd_id)
        self.ranges_repr = _get_range_repr(id = 'ranges-bbox', ranges=self.scene_ranges)
        self.vtk1 =  dash_vtk.View(children=[self.scene_repr, self.grasp_repr, self.ranges_repr], id="vtk-view1", background=[1., 1., 1.], interactorSettings=self.vtk_interaction_setting)

        self.app = Dash(self.name)
        self.server = self.app.server




        rows = []
        first_row = html.Div(children=[html.Div(children=[html.Div(["Show Scene: ", dash_daq.BooleanSwitch(id='scene_visible_toggle', on=True),]),
                                                        html.Div(["Show Gripper: ", dash_daq.BooleanSwitch(id='grasp_visible_toggle', on=True),]),
                                                        html.Div(["Robot State: ",
                                                                  html.Div(id='robot-state-msg', children=[self.robot_state_msg]),
                                                                  dcc.Interval(id='robot-state-msg-update-clock', interval=self.robot_state_msg_update_interval_ms),
                                                                  ])
                                                        ], style={"height": "1000px", "width": "500px"}, id="left-panel"),

                                    html.Div(children=[self.vtk1,
                                                       dcc.Interval(id='scene-pcd-update-interval', interval=self.pcd_update_check_interval_ms),
                                                       dcc.Interval(id='grasp-pcd-update-interval', interval=self.pcd_update_check_interval_ms),
                                                       ], style={"height": "1000px", "width": "1000px"}, id="vtk-view1-panel"),

                                    html.Div([html.H3("Target Pose"),
                                                #html.Div(["Input: ", dcc.Input(id='my-input', value=0., type='number')]),
                                                html.Div(["x: ", 
                                                        dash_daq.Slider(id='x-slider', min=self.scene_ranges[0,0], max=self.scene_ranges[0,1], value=self.scene_ranges[0,0], step=(self.scene_ranges[0,1]-self.scene_ranges[0,0])/self.n_trans_step, size=self.slider_size_px),
                                                        html.Div(id='current-x')
                                                        ]),
                                                html.Div(["y: ", 
                                                        dash_daq.Slider(id='y-slider', min=self.scene_ranges[1,0], max=self.scene_ranges[1,1], value=self.scene_ranges[1,0], step=(self.scene_ranges[1,1]-self.scene_ranges[1,0])/self.n_trans_step, size=self.slider_size_px),
                                                        html.Div(id='current-y')
                                                        ]),
                                                html.Div(["z: ", 
                                                        dash_daq.Slider(id='z-slider', min=self.scene_ranges[2,0], max=self.scene_ranges[2,1], value=self.scene_ranges[2,0], step=(self.scene_ranges[2,1]-self.scene_ranges[2,0])/self.n_trans_step, size=self.slider_size_px),
                                                        html.Div(id='current-z')
                                                        ]),
                                                html.Div(["Rx: ", 
                                                        dash_daq.Slider(id='Rx-slider', min=-180, max=180, value=-180, step=2*180/self.n_rot_step, size=self.slider_size_px),
                                                        html.Div(id='current-Rx')
                                                        ]),
                                                html.Div(["Ry: ", 
                                                        dash_daq.Slider(id='Ry-slider', min=-180, max=180, value=-180, step=2*180/self.n_rot_step, size=self.slider_size_px),
                                                        html.Div(id='current-Ry')
                                                        ]),
                                                html.Div(["Rz: ", 
                                                        dash_daq.Slider(id='Rz-slider', min=-180, max=180, value=-180, step=2*180/self.n_rot_step, size=self.slider_size_px),
                                                        html.Div(id='current-Rz')
                                                        ]),
                                                html.Br(),
                                                html.Div(["Target SE(3) Pose (qw, qx, qy, qz, x, y, z): ",
                                                          html.Div(id='target-pose'),
                                                          ],
                                                         id='output-info'),
                                                html.Br(),
                                                html.Div([html.Button('Submit', id='submit-count', n_clicks=0),
                                                          html.Div(id='submitted')
                                                          ],
                                                         id='submit-info'),
                                                html.Br(),
                                                html.Div([html.Button('Reset', id='reset-count', n_clicks=0),
                                                          html.Div(id='reset')
                                                          ],
                                                         id='reset-info'),

                                              ], id="right-panel"),                            
                                    ],
                            style={'display': 'flex', 'flex-direction': 'row', "height": "1200px", })
        rows.append(first_row)
        # second_row = html.Div(children=["Input: ", dcc.Input(id='my-input', value='initial value', type='text')])
        # rows.append(second_row)
        self.app.layout = html.Div(children=rows, style={'display': 'flex', 'flex-direction': 'column'}, id='main_panel')


        self.app.callback(
            Output(component_id=self.scene_pcd_id, component_property="actor"),
            Input(component_id='scene_visible_toggle', component_property='on')
        )(self._scene_interaction_cb)

        self.app.callback(
            Output(component_id='target-pose', component_property='children'),
            Output(component_id=self.grasp_pcd_id, component_property="actor"),
            Input(component_id='grasp_visible_toggle', component_property='on'),
            Input(component_id='x-slider', component_property='value'),
            Input(component_id='y-slider', component_property='value'),
            Input(component_id='z-slider', component_property='value'),
            Input(component_id='Rx-slider', component_property='value'),
            Input(component_id='Ry-slider', component_property='value'),
            Input(component_id='Rz-slider', component_property='value'),
        )(self._grasp_interaction_cb)

        self.app.callback(
            Output(component_id='current-x', component_property='children'),
            Output(component_id='current-y', component_property='children'),
            Output(component_id='current-z', component_property='children'),
            Output(component_id='current-Rx', component_property='children'),
            Output(component_id='current-Ry', component_property='children'),
            Output(component_id='current-Rz', component_property='children'),
            Input(component_id='x-slider', component_property='value'),
            Input(component_id='y-slider', component_property='value'),
            Input(component_id='z-slider', component_property='value'),
            Input(component_id='Rx-slider', component_property='value'),
            Input(component_id='Ry-slider', component_property='value'),
            Input(component_id='Rz-slider', component_property='value'),
        )(self._slider_update_cb)

        self.app.callback(
            Output(component_id=f"{self.scene_pcd_id}-poly", component_property="points"),
            Output(component_id=f"{self.scene_pcd_id}-array", component_property="values"),
            Input(component_id='scene-pcd-update-interval', component_property='n_intervals')
        )(self._scene_update_cb)

        self.app.callback(
            Output(component_id=f"{self.grasp_pcd_id}-poly", component_property="points"),
            Output(component_id=f"{self.grasp_pcd_id}-array", component_property="values"),
            Input(component_id='grasp-pcd-update-interval', component_property='n_intervals')
        )(self._grasp_update_cb)

        self.app.callback(
            Output(component_id='submitted', component_property="children"),
            Input(component_id='submit-count', component_property="n_clicks")
        )(self._submit_button_cb)

        self.app.callback(
            Output(component_id='reset', component_property="children"),
            Input(component_id='reset-count', component_property="n_clicks")
        )(self._reset_button_cb)

        self.app.callback(
            Output(component_id='robot-state-msg', component_property="children"),
            Input(component_id='robot-state-msg-update-clock', component_property='n_intervals')
        )(self._robot_state_msg_cb)

        self.threads=[]
        if not self.server_debug:
            self.threads.append(threading.Thread(name='dash_app_server', target=self._run_server))

    def run(self):
        for thread in self.threads:
            thread.start()
        if self.server_debug:
            self._run_server()

    def close(self):
        for thread in self.threads:
            thread.terminate()

        # func = request.environ.get('werkzeug.server.shutdown')
        # if func is None:
        #     raise RuntimeError('Not running with the Werkzeug Server')
        # func()

    def _run_server(self):
        self.app.run_server(debug=self.server_debug, host=self.host_id)

    def update_scene_pcd(self, pcd: PointCloud):
        self.scene_pcd: PointCloud = pcd
        self._scene_pcd_updated = True

    def update_grasp_pcd(self, pcd: PointCloud):
        self.grasp_pcd: PointCloud = pcd
        self._grasp_pcd_updated = True

    def _get_scene_repr(self, id: str):
        return _get_pcd_repr(id = id, points = self.scene_pcd.points, colors=self.scene_pcd.colors, point_size=self.point_size)
    
    def _get_grasp_repr(self, id: str):
        return _get_pcd_repr(id = id, points = self.grasp_pcd.points, colors=self.grasp_pcd.colors, point_size=self.point_size)

    def _scene_interaction_cb(self, input_value):
        return {"visibility": input_value}

    def _grasp_interaction_cb(self, visible, x, y, z, Rx, Ry, Rz):
        # https://vtk.org/doc/nightly/html/classvtkProp3D.html#a1c44f66f6ce311d9f38b9e1223f9cee5
        # Orientation is specified as X,Y and Z rotations in that order, but they are performed as RotateZ, RotateX, and finally RotateY.        
        orns = Rotation.from_rotvec([[0., 0., Rz/180*np.pi]]) * Rotation.from_rotvec([[Rx/180*np.pi, 0., 0.]]) * Rotation.from_rotvec([[0., Ry/180*np.pi, 0.]])
        orns = orns.as_quat()
        self._target_pose = SE3.from_numpy(positions=np.array([[x,y,z]]), orns=orns, versor_last_input=True)
        target_pose = self._target_pose.poses[0].numpy()
        target_pose = ["("] + [f"{target_pose[i]: .3f},   " for i in range(7)] + [")"]
        target_pose = "".join(target_pose)        
        output = [target_pose]
        output.append({"position": [x, y, z], 'orientation': [Rx, Ry, Rz], "visibility": visible})
        return tuple(output)
    
    def _slider_update_cb(self, x, y, z, Rx, Ry, Rz):
        return f"{x}", f"{y}", f"{z}", f"{Rx}", f"{Ry}", f"{Rz}"
    
    def _scene_update_cb(self, n):
        if self._scene_pcd_updated:
            self._scene_pcd_updated = False
            points, colors = self.scene_pcd.points, self.scene_pcd.colors
            assert points.device == colors.device == torch.device('cpu')
            return points.ravel(), colors.ravel() * 255
        else:
            return dash.no_update, dash.no_update
        
    def _grasp_update_cb(self, n):
        if self._grasp_pcd_updated:
            self._grasp_pcd_updated = False
            points, colors = self.grasp_pcd.points, self.grasp_pcd.colors
            assert points.device == colors.device == torch.device('cpu')
            return points.ravel(), colors.ravel() * 255
        else:
            return dash.no_update, dash.no_update
        
    def _submit_button_cb(self, n_click):
        if not self._request_response_flag or n_click < 1:
            return f""
        else:
            self._user_response = SE3(poses=self._target_pose.poses.detach().clone())
            self._request_response_flag, self._user_response_ready = False, True
            return f"Submitted response at {datetime.now()}."
        
    def _reset_button_cb(self, n_click):
        if not self._request_response_flag or n_click < 1:
            return f""
        else:
            self._user_response = RESET
            self._request_response_flag, self._user_response_ready = False, True
            return f"Sent reset signal at {datetime.now()}."
        
    def _robot_state_msg_cb(self, n):
        if self._robot_state_msg_update_flag:
            self._robot_state_msg_update_flag = False
            return f"{self.robot_state_msg}"
        else:
            return dash.no_update
            

    def get_user_response(self) -> Union[SE3, str]:
        self._request_response_flag = True
        
        while not self._user_response_ready:
            time.sleep(0.1)
        self._user_response_ready = False
        return self._user_response
    
    def update_robot_state(self, msg: str):
        self.robot_state_msg = msg
        self._robot_state_msg_update_flag = True