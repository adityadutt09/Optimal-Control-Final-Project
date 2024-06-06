import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet


class WorldConfig():

    def __init__(self,):
        self.world_config = {"cylinder": {}, "cuboid": {}, "mesh": {}, "capsule": {}}
    """
    Generate Base Curobo World Config Function from MJCF Physics Model
    """
    def generate_world_config(self, 
                              physics: Any, 
                              mesh_dict: dict = None, 
                              geom_group: int = 3, 
                              mesh_test: bool = False, 
                              skip_robot_name: str = ""
                            ):

        collision_geom_idx = np.where(physics.model.geom_group == geom_group)[0]
        
        ### Looping over Robot Meshes for world_config ####
        for i in collision_geom_idx:
            
            # Access the geom model and data
            geom = physics.model.geom(i)
            geom_data = physics.data.geom(i)

            if(geom.name.split("/")[0] == skip_robot_name):
                continue
            
            ## Check for different geom types ##

            ##  Mujoco Description: Sphere 
            #   Centered at geom's position 
            #   Size Params: radius of sphere

            if geom.type[0] == 2: #Sphere
                self.world_config["sphere"][geom.name] = {   "radius": float(geom.size[0]),
                                                        "pose": np.concatenate([geom_data.xpos, [1, 0, 0, 0]])
                                                    }
            
            ##  Mujoco Description: Capsule is a cylinder capped with two half-spheres. 
            #   Orientation along z-axis of geom's frame. 
            #   Size Params: radius, half-height of cylinder 
            #   Note: However capsules as well as cylinders can also be thought of as connectors, 
            #   allowing an alternative specification with the fromto attribute below. 
            #   In that case only one size parameter is required, namely the radius of the capsule.

            if geom.type[0] ==  3: #Capsule       
                self.world_config["capsule"][geom.name] = {  "radius": float(geom.size[0]),
                                                        "base": np.array([0.0, 0.0, 0.0]),
                                                        "tip": np.array([0., 0.0, 2.0*float(geom.size[1])]),
                                                        "pose": np.concatenate([geom_data.xpos, [1, 0, 0, 0]])
                                                    }

            ##  Mujoco Description: Cylinder
            #   Orientation along z-axis of geom's frame
            #   Size Params: radius, half-height 
            #   Note: can be specified with the fromto attribute

            if geom.type[0] == 5: #Cylinder
                self.world_config["cylinder"][geom.name] = { "radius": float(geom.size[0]),
                                                        "height": 2.0*float(geom.size[1]),
                                                        "pose" : np.concatenate([geom_data.xpos, [1, 0, 0, 0]])
                                                    }
            
            ##  Mujoco Description: Box
            #   Size Params: half-sizes of X, Y, Z along the xyz axes of the geom's frame
            if geom.type[0] ==  6: #Cuboid
                # print(i)
                self.world_config["cuboid"][geom.name] = {   "dims": list(geom.size*2),
                                                        "pose": np.concatenate([geom_data.xpos, [1, 0,0,0]])
                                                    }

            ## Mujoco Description: Mesh
            #   A mesh file has to be provided 
            #   Geom sizes are ignored!!!

            if geom.type[0] ==  7: #Mesh
                if mesh_dict is None:
                    print(f"Mesh Dictionary not provided but Geoms {geom.name} contain Meshes. Skipping")
                    continue

                mesh_file = ""
                #Find the key for the mesh file
                for key, values in mesh_dict.items():
                    if geom.name in values:
                        mesh_file = key
                
                if mesh_file == None:
                    print("Geom: ", geom.name, "could not be found in mesh_dict")
                    continue

                self.world_config["mesh"][geom.name] = { "file_path": mesh_file,
                                                    "pose": np.concatenate([geom_data.xpos, [1, 0, 0, 0]])
                                                }

        ### Testing Meshes Work ###
        if mesh_test:
            for key, value in self.world_config["mesh"].items():
                print(key, ":", value["file_path"])
    
    """
    Transform Object Pose to Robot Frame:
    
    Notes:
    - params are using Mujoco Convention for Orientation when passed in: (wxyz)
    - returned in Mujoco Convention
    """
    def transform_pose_robot(self, object_pose: np.ndarray, robot_pose: np.ndarray, verbose: bool = False):
        robot_pose_x = robot_pose[:3]
        robot_pose_q = robot_pose[[4,5,6,3]]
        object_pose_x = object_pose[:3]
        object_pose_q = object_pose[[4,5,6,3]]
        
        robot_r = R.from_quat(robot_pose_q)
        object_r = R.from_quat(object_pose_q)
        
        new_pose_x = robot_r.inv().apply(object_pose_x - robot_pose_x)
        new_quat = robot_r.inv()*object_r
        
        if verbose:
            print(object_pose_q,robot_pose_q,robot_r.inv().as_matrix(),object_r.as_matrix())
        
        return np.concatenate(( new_pose_x, -new_quat.as_quat()[[3,0,1,2]] ))
    
    """
    Update the Curobo World Config after making Changes to MJCF Environment
    """
    def update_curobo_world(self, physics, world_config = None, robot_name = ""):
        
        robot_pos = np.concatenate((physics.named.data.xpos[robot_name],
                                    physics.named.data.xquat[robot_name]), axis=0)
        
        if world_config is None:
            world_config = self.world_config 

        for object_type in world_config:
            
            print("Updating Type: ", object_type)

            object_dict = world_config[object_type]
            
            if object_type == "mesh":
                for name in object_dict:
                    new_xpos = physics.named.data.xpos[physics.named.model.geom_bodyid[name]].copy()
                    new_quat = physics.named.data.xquat[physics.named.model.geom_bodyid[name]]
                    xpos_robot_frame = self.transform_pose_robot(np.concatenate((new_xpos,new_quat), axis=0).copy(),robot_pos.copy())
                    object_dict[name]['pose'] = xpos_robot_frame.copy()

            else:
                for name in object_dict:
                    new_xpos = physics.named.data.geom_xpos[name].copy()
                    new_quat = R.from_matrix(physics.named.data.geom_xmat[name].reshape(3,3)).as_quat()[[3,0,1,2]]
                    xpos_robot_frame = self.transform_pose_robot(np.concatenate((new_xpos,new_quat), axis=0).copy(),robot_pos.copy())
                    object_dict[name]['pose'] = xpos_robot_frame.copy()
                




