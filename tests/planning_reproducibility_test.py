import os
import os.path as osp
import random
from pathlib import Path

from ndf_robot.utils.eval_gen_utils import get_ee_offset

NDF_ROOT = Path(__file__).parent.parent
os.environ["NDF_SOURCE_DIR"] = str(NDF_ROOT / "src" / "ndf_robot")
os.environ["PB_PLANNING_SOURCE_DIR"] = str(NDF_ROOT / "pybullet-planning")


import numpy as np
from airobot import Robot, log_critical, log_debug, log_info, log_warn, set_log_level
from airobot.utils.common import euler2quat
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.utils import path_util, trimesh_util, util
from ndf_robot.utils.franka_ik import FrankaIK

# def get_plan(seed):
#     obj_class = "mug"
#     exp = "grasp_rim_hang_handle_gaussian_precise_w_shelf"
#     demo_load_dir = osp.join(path_util.get_ndf_data(), "demos", obj_class, exp)

#     # get filenames of all the demo files
#     demo_filenames = os.listdir(demo_load_dir)
#     assert len(demo_filenames), "No demonstrations found in path: %s!" % demo_load_dir

#     # strip the filenames to properly pair up each demo file
#     grasp_demo_filenames_orig = [
#         osp.join(demo_load_dir, fn) for fn in demo_filenames if "grasp_demo" in fn
#     ]  # use the grasp names as a reference

#     place_demo_filenames = []
#     grasp_demo_filenames = []
#     for i, fname in enumerate(grasp_demo_filenames_orig):
#         shapenet_id_npz = fname.split("/")[-1].split("grasp_demo_")[-1]
#         place_fname = osp.join(
#             "/".join(fname.split("/")[:-1]), "place_demo_" + shapenet_id_npz
#         )
#         if osp.exists(place_fname):
#             grasp_demo_filenames.append(fname)
#             place_demo_filenames.append(place_fname)
#         else:
#             log_warn(
#                 "Could not find corresponding placement demo: %s, skipping "
#                 % place_fname
#             )

#     grasp_demo_filenames = grasp_demo_filenames[:12]
#     place_demo_filenames = place_demo_filenames[:12]

#     cfg = get_eval_cfg_defaults()
#     robot = Robot(
#         "franka",
#         pb_cfg={"gui": False, "realtime": False},
#         arm_cfg={"self_collision": False, "seed": seed},
#     )
#     ik_helper = FrankaIK(gui=False)
#     table_z = cfg.TABLE_Z
#     # reset
#     robot.arm.reset(force_reset=True)
#     robot.cam.setup_camera(
#         focus_pt=[0.4, 0.0, table_z], dist=0.9, yaw=45, pitch=-25, roll=0
#     )
#     grasp_demo_fn = grasp_demo_filenames[i]
#     grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

#     cams = MultiCams(cfg.CAMERA, robot.pb_client, n_cams=cfg.N_CAMERAS)
#     cam_info = {}
#     cam_info["pose_world"] = []
#     for cam in cams.cams:
#         cam_info["pose_world"].append(util.pose_from_matrix(cam.cam_ext_mat))

#     # put table at right spot
#     table_ori = euler2quat([0, 0, np.pi / 2])

#     # this is the URDF that was used in the demos -- make sure we load an identical one
#     tmp_urdf_fname = osp.join(
#         path_util.get_ndf_descriptions(), "hanging/table/table_rack_tmp.urdf"
#     )
#     open(tmp_urdf_fname, "w").write(grasp_data["table_urdf"].item())
#     table_id = robot.pb_client.load_urdf(
#         tmp_urdf_fname, cfg.TABLE_POS, table_ori, scaling=cfg.TABLE_SCALING
#     )

#     rack_link_id = 0
#     shelf_link_id = 1

#     robot.arm.go_home(ignore_physics=True)

#     robot.pb_client.set_step_sim(False)
#     robot.arm.move_ee_xyz([0, 0, 0.2])
#     robot.pb_client.set_step_sim(True)
#     arm_state1 = robot.arm.get_jpos()

#     pre_grasp_ee_pose = [
#         0.5412378070932139,
#         -0.444533159663362,
#         1.128684750848088,
#         -0.38332738636033026,
#         0.9100620427603427,
#         -0.07526680804176965,
#         0.13849946137163732,
#     ]
#     pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)

#     pre_pre_grasp_ee_pose = util.pose_stamped2list(
#         util.transform_pose(
#             pose_source=util.list2pose_stamped(pre_grasp_ee_pose),
#             pose_transform=util.list2pose_stamped(pregrasp_offset_tf),
#         )
#     )
#     jnt_pos = ik_helper.get_feasible_ik(pre_pre_grasp_ee_pose)

#     plan1 = ik_helper.plan_joint_motion(robot.arm.get_jpos(), jnt_pos)

#     return jnt_pos, plan1


def test_motion_planning_repro():
    def compute_motion_plan(seed):
        jnt_pos = np.asarray(
            [
                -0.03422427,
                0.52081948,
                -0.78973238,
                -1.86084466,
                0.81015252,
                2.3547484,
                1.86654342,
            ]
        )

        robot = Robot(
            "franka",
            pb_cfg={"gui": False, "realtime": False},
            arm_cfg={"self_collision": False, "seed": seed},
        )
        ik_helper = FrankaIK(gui=False)

        robot.arm.go_home(ignore_physics=True)
        # robot.pb_client.set_step_sim(False)
        # robot.arm.move_ee_xyz([0, 0, 0.2])
        # robot.pb_client.set_step_sim(True)

        plan = ik_helper.plan_joint_motion(
            robot.arm.get_jpos(),
            jnt_pos,
            max_time=float("inf"),
            max_iterations=1000000000,
        )

        return plan

    seed = 0
    random.seed(seed)
    plan1 = np.asarray(compute_motion_plan(seed))
    random.seed(seed)
    plan2 = np.asarray(compute_motion_plan(seed))

    # breakpoint()

    assert np.allclose(plan1, plan2)


def test_ik_reproducible():
    def compute_ik(seed):
        random.seed(seed)

        robot = Robot(
            "franka",
            pb_cfg={"gui": False, "realtime": False},
            arm_cfg={"self_collision": False, "seed": seed},
        )
        ik_helper = FrankaIK(gui=False)
        pre_grasp_ee_pose = [
            0.5412378070932139,
            -0.444533159663362,
            1.128684750848088,
            -0.38332738636033026,
            0.9100620427603427,
            -0.07526680804176965,
            0.13849946137163732,
        ]
        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)

        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(pre_grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf),
            )
        )
        jnt_pos = ik_helper.get_feasible_ik(pre_pre_grasp_ee_pose)
        return jnt_pos

    seed = 0

    np.random.seed(seed)
    jnt_pos1 = np.asarray(compute_ik(seed))

    np.random.seed(seed)

    jnt_pos2 = np.asarray(compute_ik(seed))

    assert np.allclose(jnt_pos1, jnt_pos2)

    np.random.seed(seed + 1)

    jnt_pos3 = np.asarray(compute_ik(seed))
    assert not np.allclose(jnt_pos1, jnt_pos3)

    breakpoint()
