from abc import ABCMeta, abstractmethod
from typing import Tuple, Iterable, Any, Dict, Optional, TypeVar, List, Union
import warnings

from edf.data import SE3

RobotStateType = TypeVar('RobotStateType')
PlanType = TypeVar('PlanType')
ObjectType = TypeVar('ObjectType')


PLAN_FAIL = 'PLAN_FAIL'
EXECUTION_FAIL = 'EXECUTION_FAIL'
SUCCESS = 'SUCCESS'
RESET = 'RESET'
FEASIBLE = 'FEASIBLE'
INFEASIBLE = 'INFEASIBLE'


class EdfInterfaceBase(metaclass=ABCMeta):
    def __init__(self):
        self.in_grasp: bool = False

    @abstractmethod
    def move_plans(self, targets: Iterable[Tuple[SE3, Dict]], start_state: Optional[RobotStateType] = None) -> Tuple[List[bool], List[PlanType]]:
        pass
    
    @abstractmethod
    def execute_plans(self, plans: Iterable[PlanType]) -> List[bool]:
        pass

    @abstractmethod
    def grasp(self) -> bool:
        pass
    
    @abstractmethod
    def release(self) -> bool:
        pass

    @abstractmethod
    def attach(self, obj: ObjectType):
        pass

    @abstractmethod
    def attach_placeholder(self):
        pass

    @abstractmethod
    def detach(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def move_simple(self, target_poses: SE3, planner_kwargs: Dict = {'planner_name': 'default'}) -> str:
        ###### Plan ######
        plan_targets = [(target_pose, planner_kwargs) for target_pose in target_poses]
                    
        plan_results, plans = self.move_plans(targets=plan_targets)
        plan_success = plan_results[-1]
        if not plan_success:
            return PLAN_FAIL, 'MOVE_SIMPLE_PLAN_FAIL'
        
        ###### Execution ###### 
        execution_results = self.execute_plans(plans)
        execution_success = execution_results[-1]
        if not execution_success:
            return EXECUTION_FAIL, 'MOVE_SIMPLE_EXECUTION_FAIL'
        
        return SUCCESS, 'MOVE_SIMPLE_SUCCESS'
    
    def pick_plan(self, pre_pick_pose: SE3, pick_pose: SE3, 
                  pre_pick_kwargs: Dict = {'planner_name': 'default'}, 
                  pick_kwargs: Dict = {'planner_name': 'cartesian', 'cartesian_step': 0.001, 'cspace_step_thr': 10., 'avoid_collision': False, 'success_fraction': 0.95},
                  ) -> Tuple[str, Union[List[PlanType], str]]:
        if self.in_grasp:
            warnings.warn("EdfInterfaceBase: pick_plan() called, but the robot is seemingly in grasp. Please call .release() before calling .pick_plan() method.")

        plan_targets = [(pre_pick_pose, pre_pick_kwargs),
                        (pick_pose, pick_kwargs),
                        ]
        plan_results, plans = self.move_plans(targets=plan_targets)
        plan_success = plan_results[-1]

        if not plan_success:
            return PLAN_FAIL, 'PICK_PLAN_FAIL'
        else:
            return SUCCESS, plans

    def pick_execute(self, plans: List[PlanType], post_pick_pose: SE3, 
                     post_pick_kwargs: Dict = {'planner_name': 'cartesian', 'cartesian_step': 0.001, 'cspace_step_thr': 10., 'avoid_collision': False, 'success_fraction': 0.95},
                     ) -> Tuple[str, str]:
        if self.in_grasp:
            warnings.warn("EdfInterfaceBase: pick_execute() called, but the robot is seemingly in grasp. Please call .release() before calling .pick_plan() method.")

        ###### Pre-pick Execution ######
        execution_results = self.execute_plans(plans)
        execution_success = execution_results[-1]
        if not execution_success:
            return EXECUTION_FAIL, 'PICK_EXECUTION_FAIL'

        ###### Grasp ######
        grasp_result = self.grasp()
        if not grasp_result:
            return EXECUTION_FAIL, 'GRASP_FAIL'

        ###### Post-pick Plan ######
        plan_targets = [(post_pick_pose, post_pick_kwargs),
                        ]
        plan_results, plans = self.move_plans(targets=plan_targets)
        plan_success = plan_results[-1]
        if not plan_success:
            return EXECUTION_FAIL, 'POST_PICK_PLAN_FAIL'
        
        ###### Post-pick Execution ######
        execution_results = self.execute_plans(plans)
        execution_success = execution_results[-1]
        if not execution_success:
            return EXECUTION_FAIL, 'POST_PICK_EXECUTION_FAIL'
        
        return SUCCESS, 'PICK_SUCCESS'

    def place_plan(self, pre_place_pose: SE3, place_pose: SE3, 
                   pre_place_kwargs: Dict = {'planner_name': 'default'}, 
                   place_kwargs: Dict = {'planner_name': 'cartesian', 'cartesian_step': 0.01, 'cspace_step_thr': 10., 'avoid_collision': False, 'success_fraction': 0.95},
                   ) -> Tuple[str, Union[List[PlanType], str]]:
        
        ###### Pre-place Plan ######
        plan_targets = [(pre_place_pose, pre_place_kwargs),
                        (place_pose, place_kwargs),
                        ]
        plan_results, plans = self.move_plans(targets=plan_targets)
        plan_success = plan_results[-1]
        if not plan_success:
            return PLAN_FAIL, 'PLACE_PLAN_FAIL'
        else:
            return SUCCESS, plans

    def place_execute(self, plans: List[PlanType], post_place_pose: SE3, 
                      post_place_kwargs: Dict = {'planner_name': 'cartesian', 'cartesian_step': 0.01, 'cspace_step_thr': 10., 'avoid_collision': False, 'success_fraction': 0.95},
                      ) -> Tuple[str, str]:

        ###### Pre-place Execution ######
        execution_results = self.execute_plans(plans)
        execution_success = execution_results[-1]
        if not execution_success:
            return EXECUTION_FAIL, 'PLACE_EXECUTION_FAIL'

        ###### Release ######
        self.detach()
        release_result = self.release()
        if not release_result:
            return EXECUTION_FAIL, 'RELEASE_FAIL'

        ###### Post-place Plan ######
        plan_targets = [(post_place_pose, post_place_kwargs),
                       ]
        plan_results, plans = self.move_plans(targets=plan_targets)
        plan_success = plan_results[-1]
        if not plan_success:
            return EXECUTION_FAIL, 'POST_PLACE_PLAN_FAIL'
        
        ###### Post-place Execution ######
        execution_results = self.execute_plans(plans)
        execution_success = execution_results[-1]
        if not execution_success:
            return EXECUTION_FAIL, 'POST_PLACE_EXECUTION_FAIL'
        
        return SUCCESS, 'PLACE_SUCCESS'
