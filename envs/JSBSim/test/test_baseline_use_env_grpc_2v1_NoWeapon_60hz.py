from abc import ABC
import sys
import os
import pandas as pd
import time
import math
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir, LLA2NEU, NEU2LLA, body_ned_to_world_ned, hit_rate, damage_rate, get_AO_TA_R, world_ned_to_body_ned, cal_azi_ele_from_euler
from scripts.udp_comm.udp_client import UDPClient, TacviewTCPServer
# from scripts.udp_comm.udp_server import UDPServer 

# keyboard interrupt
import threading

# grpc communication
import grpc
import ADEX_pb2
import ADEX_pb2_grpc

# upload dll file
import ctypes

FEET2METER = 0.3048
METER2FEET = 1 / 0.3048
DEG2RAD = 3.14159265/180
RAD2DEG = 180/3.14159265
G2FEET = 9.80665 / 0.3048
METER2MACH = 1/340.29

dll_path = "./x64/Weapon_Dynamic.dll"
weapon_dll = ctypes.CDLL(dll_path)

class category_type(ctypes.c_int):
    UTILITY = 0
    LIGHT_FIGHTER = 1
    HEAVY_FIGHTER = 2
    TRANSPORT = 3
    HELICOPTER = 4

class entity_status_type(ctypes.c_int):
    DEACTIVATED = 0
    ACTIVATED = 1
    EXPLODED = 2
    REMOVED = 3
    NUM_ENTITY_STATUS = 4
    GROUND_EXPLODED = 5
    AA_MISSILE_EXPLODED = 6
    CLUST_EXPLODED = 7

# rad rad ft
class position_type(ctypes.Structure):
    _fields_ = [
        ("Longitude", ctypes.c_double), 
        ("Latitude", ctypes.c_double), 
        ("Altitude", ctypes.c_double)
    ]

class xyz_type(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double)
    ]

# rad rad rad
class attitude_type(ctypes.Structure):
    _fields_ = [
        ("Roll", ctypes.c_double),
        ("Pitch", ctypes.c_double),
        ("Yaw", ctypes.c_double)
    ]

class Player_data_type(ctypes.Structure):
    _fields_ = [
        ("Entity_Index", ctypes.c_int),
        ("Category", category_type),
        ("Position", position_type),
        ("Velocity_NED", xyz_type), # ft/s
        ("Velocity_Body", xyz_type),
        ("Acceleration_NED", xyz_type), # ft/s2
        ("Acceleration_Body", xyz_type), 
        ("Attitude", attitude_type), # rad
        ("True_Heading", ctypes.c_double), # rad
        ("Normal_Load_Factor", ctypes.c_double), #  
        ("Normal_Acceleration", ctypes.c_double), # 
        ("CAS", ctypes.c_double),
        ("Mach_Number", ctypes.c_double),
        ("Angle_Of_Attack", ctypes.c_double),
        ("After_Burner_On", ctypes.c_bool)
    ]

class ejection_type(ctypes.Structure):
    _fields_ = [
        ("Azimuth_Angle", ctypes.c_double),
        ("Elevation_Angle", ctypes.c_double),
        ("Offset_Position", xyz_type)
    ]

class atmospheric_model_type(ctypes.c_int):
    STD_DAY = 0
    HOT_DAY = 1
    COLD_DAY = 2

class steady_wind_info_type(ctypes.Structure):
    _fields_ = [
        ("Steady_Wind_Enable", ctypes.c_bool),
        ("Steady_Wind_Altitude", ctypes.c_double),
        ("Steady_Wind_Direction", ctypes.c_double),
        ("Steady_Wind_Vertical_Speed", ctypes.c_double),
        ("Steady_Wind_Horizontal_Speed", ctypes.c_double)
    ]

#############
weapon_dll.Gun_Class_new.argtypes = [ctypes.c_int, ctypes.c_int, Player_data_type, ejection_type]
weapon_dll.Gun_Class_new.restype = ctypes.c_void_p

weapon_dll.Update_new.argtypes = [ctypes.c_void_p]
weapon_dll.Delete_Gun_Class.argtypes = [ctypes.c_void_p]
weapon_dll.Read_Data_new.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
weapon_dll.Initialize_new.argtypes = [ctypes.c_void_p]

weapon_dll.Get_Position.argtypes = [ctypes.c_void_p]
weapon_dll.Get_Position.restype = position_type

weapon_dll.Get_Weapon.argtypes = [ctypes.c_void_p]
weapon_dll.Get_Weapon.restype = ctypes.c_int

weapon_dll.Get_Status.argtypes = [ctypes.c_void_p]
weapon_dll.Get_Status.restype = ctypes.c_int

weapon_dll.Get_Velocity.argtypes = [ctypes.c_void_p]
weapon_dll.Get_Velocity.restype = xyz_type

weapon_dll.Get_Velocity_NED.argtypes = [ctypes.c_void_p]
weapon_dll.Get_Velocity_NED.restype = xyz_type

weapon_dll.Get_Acceleration.argtypes = [ctypes.c_void_p]
weapon_dll.Get_Acceleration.restype = xyz_type

weapon_dll.Get_Acceleration_NED.argtypes = [ctypes.c_void_p]
weapon_dll.Get_Acceleration_NED.restype = xyz_type

weapon_dll.Get_Attitude.argtypes = [ctypes.c_void_p]
weapon_dll.Get_Attitude.restype = attitude_type

#############
weapon_dll.Global_Class_new.restype = ctypes.c_void_p

weapon_dll.Set_Atmosphere_new.argtypes = [ctypes.c_void_p, atmospheric_model_type]
weapon_dll.Set_MainFrameTime.argtypes = [ctypes.c_void_p, ctypes.c_double]
weapon_dll.Set_SteadyWind.argtypes = [ctypes.c_void_p, steady_wind_info_type]
#############
weapon_dll.Database_Class_new.restype = ctypes.c_void_p
weapon_dll.Read_All_Entity_Data_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
#############

class Database_Class:
    def __init__(self):
        self.obj = weapon_dll.Database_Class_new()

    def Read_All_Entity_Data_New(self, path):
        weapon_dll.Read_All_Entity_Data_New(self.obj, path)

class Global_Class:
    def __init__(self):
        self.obj = weapon_dll.Global_Class_new()
    
    # void Set_Atmosphere_new(Global_Class* gc, atmospheric_model_type Model);
	# void Set_MainFrameTime(Global_Class* gc, double time);
    # void Set_SteadyWind(Global_Class* gc, steady_wind_info_type* wind);

    def Set_Atmosphere_new(self, model):
        weapon_dll.Set_Atmosphere_new(self.obj, model)

    def Set_MainFrameTime(self, time):
        weapon_dll.Set_MainFrameTime(self.obj, time)

    def Set_SteadyWind(self, wind):
        weapon_dll.Set_SteadyWind(self.obj, wind)

class Gun_Class:
    def __init__(self, gun_id, gun_idx, weapon_player_input, ejection_input):
        self.position = -1
        self.ID = -1
        self.status = -1

        self.velocity_body = -1
        self.velocity_NED = -1
        self.acceleration_body = -1
        self.acceleration_NED = -1
        self.attitude = -1

        self.id = gun_id
        self.idx = gun_idx

        self.obj = weapon_dll.Gun_Class_new(
            gun_id, gun_idx, weapon_player_input, ejection_input
        )
    
	# _GUN_CLASS_DLL_ void Read_Data_new(Gun_Class* gc, void* db);
	# _GUN_CLASS_DLL_ void Initialize_new(Gun_Class* gc);
	# _GUN_CLASS_DLL_ void Update_new(Gun_Class* gc);
	# _GUN_CLASS_DLL_ void Delete_Gun_Class(Gun_Class* gc);
    # _GUN_CLASS_DLL_ position_type Get_Position(Gun_Class* gc);
	# _GUN_CLASS_DLL_ int Get_Weapon(Gun_Class* gc);
	# _GUN_CLASS_DLL_ entity_status_type Get_Status(Gun_Class* gc);
    
	# _GUN_CLASS_DLL_ xyz_type Get_Velocity(Gun_Class* gc);
	# _GUN_CLASS_DLL_ xyz_type Get_Velocity_NED(Gun_Class* gc);
	# _GUN_CLASS_DLL_ xyz_type Get_Acceleration(Gun_Class* gc);
	# _GUN_CLASS_DLL_ xyz_type Get_Acceleration_NED(Gun_Class* gc);
	# _GUN_CLASS_DLL_ AttitudeType Get_Attitude(Gun_Class* gc);

    def Read_Data_new(self, db):
        weapon_dll.Read_Data_new(self.obj, db)

    def Initialize_new(self):
        weapon_dll.Initialize_new(self.obj)

    def Update_new(self):
        weapon_dll.Update_new(self.obj)
    
    def Delete_Gun_Class(self):
        weapon_dll.Delete_Gun_Class(self.obj)

    def Get_Position(self):
        self.position = weapon_dll.Get_Position(self.obj)

    def Get_Weapon(self):
        self.ID = weapon_dll.Get_Weapon(self.obj)

    def Get_Status(self):
        self.status = weapon_dll.Get_Status(self.obj)

    def Get_Velocity(self):
        self.velocity_body = weapon_dll.Get_Velocity(self.obj)

    def Get_Velocity_NED(self):
        self.velocity_NED = weapon_dll.Get_Velocity_NED(self.obj)

    def Get_Acceleration(self):
        self.acceleration_body = weapon_dll.Get_Acceleration(self.obj)

    def Get_Acceleration_NED(self):
        self.acceleration_NED = weapon_dll.Get_Acceleration_NED(self.obj)

    def Get_Attitude(self):
        self.attitude = weapon_dll.Get_Attitude(self.obj)

    def Get_method(self):
        self.Get_Position()
        # self.Get_Weapon()
        self.Get_Status()
        # self.Get_Velocity()
        self.Get_Velocity_NED()
        # self.Get_Acceleration()
        # self.Get_Acceleration_NED()
        # self.Get_Attitude()

# function_names = [func for func in dir(weapon_dll) if callable(getattr(weapon_dll, func))]
# print("Functions in the DLL:")
# for func_name in function_names:
#     print(f"- {func_name}")

# xyz_s = xyz_type(x = 1, y = 2, z = 3)
# position_s1 = position_type(Longitude = 127 * DEG2RAD, Latitude = 36 * DEG2RAD, Altitude = 10000)
# position_s2 = position_type(Longitude = 123 * DEG2RAD, Latitude = 40 * DEG2RAD, Altitude = 10000)
# att_s = attitude_type(Roll = 1 * DEG2RAD, Pitch = 1 * DEG2RAD, Yaw = 1  * DEG2RAD)

# player_data1 = Player_data_type(
#     Entity_Index = 1,
#     Category = category_type.UTILITY,
#     Position = position_s1,
#     Velocity_NED = xyz_s,
#     Velocity_Body = xyz_s,
#     Acceleration_NED = xyz_s,
#     Acceleration_Body = xyz_s,
#     Attitude = att_s,
#     True_Heading = 1 * DEG2RAD,
#     Normal_Load_Factor = 3.0,
#     Normal_Acceleration = 4.0,
#     CAS = 5.0,
#     Mach_Number = 6.0,
#     Angle_Of_Attack = 7.0,
#     After_Burner_On = False
# )

# player_data2 = Player_data_type(
#     Entity_Index = 1,
#     Category = category_type.HEAVY_FIGHTER,
#     Position = position_s2,
#     Velocity_NED = xyz_s,
#     Velocity_Body = xyz_s,
#     Acceleration_NED = xyz_s,
#     Acceleration_Body = xyz_s,
#     Attitude = att_s,
#     True_Heading = 1 * DEG2RAD,
#     Normal_Load_Factor = 3.0,
#     Normal_Acceleration = 4.0,
#     CAS = 5.0,
#     Mach_Number = 6.0,
#     Angle_Of_Attack = 7.0,
#     After_Burner_On = False
# )

# eject_type = ejection_type(Azimuth_Angle = 1.0, Elevation_Angle = 2.0, Offset_Position = xyz_s)

global_class = Global_Class()
global_class.Set_Atmosphere_new(atmospheric_model_type.STD_DAY)
global_class.Set_MainFrameTime(1.0/60.0)

db_class = Database_Class()
db_class.Read_All_Entity_Data_New(b"C:")
#db_class.Read_All_Entity_Data_New(b"C:/Users/kai_dwkim/Desktop/KDH/Project/23AI/CloseAirCombat/envs/JSBSim/test/x64")

# gun_class1 = Gun_Class(208, 1, player_data1, eject_type)
# gun_class1.Read_Data_new(db_class.obj)
# gun_class1.Initialize_new()

# gun_class2 = Gun_Class(208, 1, player_data2, eject_type)
# gun_class2.Read_Data_new(db_class.obj)
# gun_class2.Initialize_new()


# for i in range(60):
#     for j in range(60):
#         gun_class1.Update_new()
#         gun_class2.Update_new()

#     gun_class1.Get_Position()
#     gun_class1.Get_Weapon()
#     gun_class1.Get_Status()

#     print(" ")
#     print("GUN CLASS 1")
#     print("iteration : " + str(i))
#     print(gun_class1.position.Longitude * RAD2DEG)
#     print(gun_class1.position.Latitude * RAD2DEG)
#     print(gun_class1.position.Altitude)
#     print(gun_class1.status)
#     print(gun_class1.ID) 

#     gun_class2.Get_Position()
#     gun_class2.Get_Weapon()
#     gun_class2.Get_Status()

#     print(" ")
#     print("GUN CLASS 2")
#     print("iteration : " + str(i))
#     print(gun_class2.position.Longitude * RAD2DEG)
#     print(gun_class2.position.Latitude * RAD2DEG)
#     print(gun_class2.position.Altitude)
#     print(gun_class2.status)
#     print(gun_class2.ID) 


#     time.sleep(1)


#########################################################################
keyboard_input = 's'
gun_list = []
gun_lock = threading.Lock()

class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
        self.model_path = get_root_dir() + '/model/baseline_model.pt'

        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path))
        self.actor.eval()
        self.agent_id = agent_id
        self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.velocities_u_mps,                 #  5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  8. vc        (unit: m/s)
            c.position_h_sl_m                   #  9. altitude  (unit: m)
        ]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod
    def set_delta_value(self, env, task):
        raise NotImplementedError

    def get_observation(self, env, task, delta_value):
        uid = list(env.agents.keys())[self.agent_id]
        obs = env.agents[uid].get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = obs[9] / 5000                  #  3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(obs[3])                 #  4. ego_roll_sin
        norm_obs[5] = np.cos(obs[3])                 #  5. ego_roll_cos
        norm_obs[6] = np.sin(obs[4])                 #  6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[4])                 #  7. ego_pitch_cos
        norm_obs[8] = obs[5] / 340                   #  8. ego_v_x   (unit: mh)
        norm_obs[9] = obs[6] / 340                   #  9. ego_v_y    (unit: mh)
        norm_obs[10] = obs[7] / 340                  #  10. ego_v_z    (unit: mh)
        norm_obs[11] = obs[8] / 340                  #  11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs

    # #@profile
    def get_action(self, env, task, received_data):
        delta_value = self.set_delta_value(env, task, received_data)
        observation = self.get_observation(env, task, delta_value)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()

        return action


class PursueAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    # #@profile
    def set_delta_value(self, env, task, received_data):
        # NOTE: only adapt for 1v1
        ego_uid, enm_uid = list(env.agents.keys())[self.agent_id], list(env.agents.keys())[(self.agent_id+1)%2] 
        
        # (deg, deg, ft,rad, rad, rad, ft/sec) 
        # print(received_data)
        enm_lon, enm_lat, enm_alt, enm_pitch, enm_roll, enm_yaw, enm_v_x, enm_v_y, enm_v_z, enm_vn, enm_ve, enm_vd = received_data 

        # LLA2NEU : input (deg, deg, meter)
        enm_x, enm_y, enm_z = LLA2NEU(enm_lon, enm_lat, enm_alt * FEET2METER, env.agents[enm_uid].lon0, env.agents[enm_uid].lat0, env.agents[enm_uid].alt0)
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()

        # print(ego_x, ego_y, ego_z, enm_x, enm_y, enm_z)

        # delta altitude
        delta_altitude = enm_z - ego_z

        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag

        # enm_v_x need to be meter UNIT
        delta_velocity = enm_v_x * FEET2METER - env.agents[ego_uid].get_property_value(c.velocities_u_mps)

        # AI set property
        env.agents[enm_uid]._position[:] = np.array([enm_y, enm_x, enm_z])
        env.agents[enm_uid]._geodetic[:] = np.array([enm_lon, enm_lat, enm_alt * FEET2METER])
        env.agents[enm_uid]._posture[:] = np.array([enm_roll, enm_pitch, enm_yaw])

        return np.array([delta_altitude, delta_heading, delta_velocity])


def cal_WVR_damage(ego_feature, recv_data, env, damaged_munition_idx_list):
    std_lon, std_lat, std_alt = env.agents["B0100"].lon0, env.agents["B0100"].lat0, env.agents["B0100"].alt0
    enm_state_list = []
    ego_damage_dict = {agent_name : 0 for agent_name in env.agents.keys() if agent_name[0] == "B"}
    ego_damage = 0

    damage_ratio = 10

    for enm_idx in range(len(recv_data)):    
        # 이미 맞은 munition 은 제외
        if (recv_data[enm_idx].Entity_Index in damaged_munition_idx_list):
            continue

        munition_idx = recv_data[enm_idx].Entity_Index
        enm_pos = [recv_data[enm_idx].Geocentri_Position_Y, recv_data[enm_idx].Geocentri_Position_X, recv_data[enm_idx].Geocentri_Position_Z * FEET2METER]
        enm_n, enm_e, enm_u = LLA2NEU(*enm_pos, std_lon, std_lat, std_alt)
        enm_state_list.append([munition_idx, enm_n, enm_e, enm_u])

    for agent_id in env.agents.keys():
        if (agent_id[0] == "B"): # AI
            ego_lat, ego_lon, ego_alt = ego_feature[:3]
            ego_n, ego_e, ego_u = LLA2NEU(ego_lon, ego_lat, ego_alt, std_lon, std_lat, std_alt)
        
            for enm_state in enm_state_list:
                munition_idx, enm_n, enm_e, enm_u = enm_state
                dist = math.sqrt((ego_n - enm_n)**2 + (ego_e - enm_e)**2 + (ego_u - enm_u)**2)
                if (dist < 10):
                    ego_damage_dict[agent_id] += damage_ratio
                    ego_damage += damage_ratio
                    damaged_munition_idx_list.append(munition_idx)
    
    return ego_damage, damaged_munition_idx_list


# #@profile
def check_overlap_corn(ego_feature, recv_data, env):
    std_lon, std_lat, std_alt = env.agents["B0100"].lon0, env.agents["B0100"].lat0, env.agents["B0100"].alt0
    
    enm_state_dict = {}
    for enm_idx in range(len(recv_data)):
        enm_pos = [recv_data[enm_idx].Geocentri_Position_Y, recv_data[enm_idx].Geocentri_Position_X, recv_data[enm_idx].Geocentri_Position_Z * FEET2METER]
        enm_v = [recv_data[enm_idx].Velocity_NED_X * FEET2METER, recv_data[enm_idx].Velocity_NED_Y * FEET2METER, recv_data[enm_idx].Velocity_NED_Z * FEET2METER]

        enm_neu = LLA2NEU(*enm_pos, std_lon, std_lat, std_alt)
        ego_lat, ego_lon, ego_alt = ego_feature[:3]
        ego_n, ego_e, ego_u = LLA2NEU(ego_lon, ego_lat, ego_alt, std_lon, std_lat, std_alt)
        enm_n, enm_e, enm_u = enm_neu
        dt = math.sqrt((enm_n - ego_n)**2 + (enm_e - ego_e)**2 + (enm_u - ego_u)**2) / 2.5 * 340.29 # 총알의 속도는 2.5 mach 

        enm_n = enm_n + enm_v[0] * dt
        enm_e = enm_e + enm_v[1] * dt
        enm_u = enm_u + -enm_v[2] * dt

        enm_lon, enm_lat, enm_alt = NEU2LLA(enm_n, enm_e, enm_u, std_lon, std_lat, std_alt)
        enm_state_dict[enm_idx] =  [enm_lat, enm_lon, enm_alt] + enm_v
    
    ego_hit = hit_rate(*ego_feature[:3], *ego_feature[6:9], *enm_state_dict[0]) + hit_rate(*ego_feature[:3], *ego_feature[6:9], *enm_state_dict[1])

    return ego_hit

def launch_gun(gun_idx, agent, ego_state):
    """
    gun 을 발사하는 함수

    Args : 
        gun_idx `int` : gun index (unique 한 값)
        agent `BaseSimulator` : Aircraft class
        ego_state `list` : [lat, lon, alt, roll, pitch, yaw, vn, ve, vd, vbx, vby, vbz, an, ae, ad]

    Returns : 
        gun_class `gun_class` : gun 객체
    
    """
    # deg, deg, m, rad, rad, rad, m/s(3), m/s(3), ft/s2(3)
    lat, lon, alt, roll, pitch, yaw, vn, ve, vd, vbx, vby, vbz, an, ae, ad = ego_state
    position_s = position_type(Longitude = lon * DEG2RAD, Latitude = lat * DEG2RAD, Altitude = alt * METER2FEET)
    attitude_s = attitude_type(Roll = roll, Pitch = pitch, Yaw = yaw)
    vel_ned_s = xyz_type(x = vn * METER2FEET, y = ve * METER2FEET, z = vd * METER2FEET)
    vel_body_s = xyz_type(x = vbx * METER2FEET, y = vby * METER2FEET, z = vbz * METER2FEET)
    abx, aby, abz = world_ned_to_body_ned(an, ae, ad, roll, pitch, yaw)
    acc_ned_s = xyz_type(x = an, y = ae, z = ad)
    acc_body_s = xyz_type(x = abx, y = aby, z = abz)

    mach_val = math.sqrt(vn**2 + ve**2 + vd**2) * METER2MACH 
    normal_acc = math.sqrt(an**2 + ae**2 + ad**2)

    aero_alpha = agent.get_property_value(c.aero_alpha_deg) # deg
    load_factor = agent.get_property_value(c.force_load_factor) # 단위가 뭔지 모르겠음
    cas = agent.get_property_value(c.velocities_vc_fps) # ft/s

    #################
    # # for test
    # xyz_s = xyz_type(x = 1, y = 2, z = 3)
    # position_s = position_type(Longitude = 127 * DEG2RAD, Latitude = 36 * DEG2RAD, Altitude = 10000)
    # att_s = attitude_type(Roll = 1 * DEG2RAD, Pitch = 1 * DEG2RAD, Yaw = 1  * DEG2RAD)

    # player_data = Player_data_type(
    #     Entity_Index = gun_idx,
    #     Category = category_type.HEAVY_FIGHTER,
    #     Position = position_s,
    #     Velocity_NED = xyz_s,
    #     Velocity_Body = xyz_s,
    #     Acceleration_NED = xyz_s,
    #     Acceleration_Body = xyz_s,
    #     Attitude = att_s,
    #     True_Heading = 1 * DEG2RAD,
    #     Normal_Load_Factor = 3.0,
    #     Normal_Acceleration = 4.0,
    #     CAS = 5.0,
    #     Mach_Number = 6.0,
    #     Angle_Of_Attack = 7.0,
    #     After_Burner_On = False
    # )
    #################

    player_data = Player_data_type(
        Entity_Index = gun_idx,
        Category = category_type.HEAVY_FIGHTER,
        Position = position_s,              # rad rad ft
        Velocity_NED = vel_ned_s,           # ft/s
        Velocity_Body = vel_body_s,         # ft/s
        Acceleration_NED = acc_ned_s,       # ft/s2
        Acceleration_Body = acc_body_s,     # ft/s2
        Attitude = attitude_s,              # rad
        True_Heading = yaw,                 # rad
        Normal_Load_Factor = load_factor,   
        Normal_Acceleration = normal_acc,   # ft/s2
        CAS = cas,                          # ft/s
        Mach_Number = mach_val,             # mach
        Angle_Of_Attack = aero_alpha,       # deg
        After_Burner_On = False
    )

    offset_s = xyz_type(x = 0, y = 0, z = 0)

    azimute, elevation = cal_azi_ele_from_euler(roll, pitch, yaw)
    # eject_data = ejection_type(Azimuth_Angle = azimute, Elevation_Angle = elevation, Offset_Position = offset_s)
    eject_data = ejection_type(Azimuth_Angle = 0, Elevation_Angle = 0, Offset_Position = offset_s)

    gun_class = Gun_Class(208, gun_idx, player_data, eject_data)
    gun_class.Read_Data_new(db_class.obj)
    gun_class.Initialize_new()

    gun_class.Update_new()
    gun_class.Get_Status()
    return gun_class


class get_sim_data_response_class():
    def __init__(self):
        self.Sim_Munition_Data_Array = []
        self.Sim_Player_Data_Array = []

class player_data_class():
    def __init__(self):
        self.Entity_ID = 0
        self.Entity_Index = 0
        self.Category = 0
        self.SubCategory = 0
        self.Entity_Status = 0
        
        self.Geocentri_Position_X = 37
        self.Geocentri_Position_Y = 120
        self.Geocentri_Position_Z = 10000

        self.Attitude_Roll = 0
        self.Attitude_Pitch = 0
        self.Attitude_Yaw = 0

        self.Velocity_NED_X = 0
        self.Velocity_NED_Y = 0
        self.Velocity_NED_Z = 0

        self.Velocity_U = 0
        self.Velocity_V = 0
        self.Velocity_W = 0

        self.Acceleration_BODY_X = 0
        self.Acceleration_BODY_Y = 0
        self.Acceleration_BODY_Z = 0

class munition_data_class():
    def __init__(self):
        self.Entity_ID = 0
        self.Entity_Index = 0
        self.Category = 0
        self.SubCategory = 0
        self.Entity_Status = 0
        
        self.Geocentri_Position_X = 0
        self.Geocentri_Position_Y = 0
        self.Geocentri_Position_Z = 0

        self.Attitude_Roll = 0
        self.Attitude_Pitch = 0
        self.Attitude_Yaw = 0

        self.Velocity_NED_X = 0
        self.Velocity_NED_Y = 0
        self.Velocity_NED_Z = 0

        self.Acceleration_NED_X = 0
        self.Acceleration_NED_Y = 0
        self.Acceleration_NED_Z = 0
        

def while_main_loop(env, multi_state_var, ego_hit_damage, agent1, step_cnt, damaged_munition_idx_list, cur_time, gun_idx):
    start_time = time.time()

    ego_feature = [-1 for i in range(len(multi_state_var))]
    for agent_name, agent in env.agents.items():
        if (agent_name[0] == "B"):
            ego_feature = env.agents[agent_name].get_property_values(multi_state_var)

    # grpc publish
    # env agent 가 모든 agent 가 있는게 아니라 AI 만 고려해야함
    get_sim_data_response = get_sim_data_response_class()
    get_sim_data_response.Sim_Player_Data_Array = [player_data_class(), player_data_class()]
    get_sim_data_response.Sim_Munition_Data_Array = []

    ego_damage, damaged_munition_idx_list = cal_WVR_damage(ego_feature, get_sim_data_response.Sim_Munition_Data_Array, env, damaged_munition_idx_list)
    ego_hit_damage[1] += ego_damage
    ego_hit_damage[1] = min(90, ego_hit_damage[1])

    ego_hit_value = check_overlap_corn(ego_feature, get_sim_data_response.Sim_Player_Data_Array, env)
    sim_data = get_sim_data_response.Sim_Player_Data_Array[0]
    if (sim_data.Entity_Status != 1 ):
        sim_data = get_sim_data_response.Sim_Player_Data_Array[1]
    
    std_lon, std_lat, std_alt = env.agents["B0100"].lon0, env.agents["B0100"].lat0, env.agents["B0100"].alt0
    enm_lat, enm_lon, enm_alt = sim_data.Geocentri_Position_X, sim_data.Geocentri_Position_Y, sim_data.Geocentri_Position_Z * FEET2METER
    enm_vn, enm_ve, enm_vd = sim_data.Velocity_NED_X, sim_data.Velocity_NED_Y, sim_data.Velocity_NED_Z

    ego_lat, ego_lon, ego_alt = ego_feature[:3]
    ego_vn, ego_ve, ego_vd = ego_feature[6:9]

    enm_n, enm_e, enm_u = LLA2NEU(enm_lon, enm_lat, enm_alt, std_lon, std_lat, std_alt)
    ego_n, ego_e, ego_u = LLA2NEU(ego_lon, ego_lat, ego_alt, std_lon, std_lat, std_alt)

    # 
    temp_ego_feature = [ego_n, ego_e, -ego_u, ego_vn, ego_ve, ego_vd]
    temp_enm_feature = [enm_n, enm_e, -enm_u, enm_vn, enm_ve, enm_vd]
    AO, _, R = get_AO_TA_R(temp_ego_feature, temp_enm_feature)

    gun_print_cnt = 0
    for gun in gun_list:
        if (gun.status == -1 or gun.status != entity_status_type.ACTIVATED):
            continue

        gun.Update_new()

        if (gun_print_cnt == 0):
            gun.Get_Position()
            gun.Get_Weapon()
            gun.Get_Status()

            print(" ")
            print(gun.position.Longitude * RAD2DEG)
            print(gun.position.Latitude * RAD2DEG)
            print(gun.position.Altitude)
            print(gun.status)
            print(gun.ID) 

            gun_print_cnt += 1

    # if (
    #     -0.5236 <= AO <= 0.5236 and
    #     math.sqrt((enm_n - ego_n)**2 + (enm_e - ego_e)**2 + (enm_u - ego_u)**2) < 2 * 1000 and
    #     step_cnt % 2 == 0
    # ): # 20km
    if (True):
        agent_name = [agent_name for agent_name in env.agents.keys() if (agent_name[0] == "B")][0]
        gun_instance = launch_gun(gun_idx, env.agents[agent_name], ego_feature)
        
        gun_list.append(gun_instance)
        gun_idx += 1

    if (time.time() - cur_time > 1):
        print("Normal detect!!")
        cur_time = time.time()

        for agent_name in env.agents.keys():
            if (agent_name[0] == "B"): # AI
                lon, lat, alt = env.agents[agent_name].get_geodetic() # [lon, lat, alt] = [deg, deg, m]
                print("Agent Name : " + str(agent_name) + " / " + str(lat) + " " + str(lon) + " " + str(alt * METER2FEET) + " / " + str(step_cnt) + " / " + str(ego_hit_damage[1]))

        for player_data in get_sim_data_response.Sim_Player_Data_Array:
            lat, lon, alt = player_data.Geocentri_Position_X, player_data.Geocentri_Position_Y, player_data.Geocentri_Position_Z
            print("Agent name : " + str(player_data.Entity_Index) + " / " + str(lat) + " " + str(lon) + " " + str(alt) + " / " + str(step_cnt) + " / " + str(ego_hit_damage[1]))

        gun_cnt = 0
        for gun in gun_list:
            if (gun.status != entity_status_type.ACTIVATED):
                continue

            gun_cnt += 1
        print("GUN COUNT : ", gun_cnt, " / len GUN LIST : ", len(gun_list))

    # received_data = (lon, lat, alt, p, r, yaw, vx, vy, vz, vn, ve, vd) - (deg, deg, ft, rad, rad, rad, ft/sec, ft/sec) 
    
    player_data = get_sim_data_response.Sim_Player_Data_Array[0]
    if (player_data.Entity_Status != 1): # ACTIVATED = 1
        player_data = get_sim_data_response.Sim_Player_Data_Array[1]

    enm_lat, enm_lon, enm_alt = player_data.Geocentri_Position_X, player_data.Geocentri_Position_Y, player_data.Geocentri_Position_Z
    enm_roll, enm_pitch, enm_yaw = player_data.Attitude_Roll, player_data.Attitude_Pitch, player_data.Attitude_Yaw
    enm_vx, enm_vy, enm_vz = player_data.Velocity_U, player_data.Velocity_V, player_data.Velocity_W
    enm_vn, enm_ve, enm_vd = 0, 0, 0

    received_data = [enm_lon, enm_lat, enm_alt, enm_pitch, enm_roll, enm_yaw, enm_vx, enm_vy, enm_vz, enm_vn, enm_ve, enm_vd]

    # action0 = agent0.get_action(env, env.task)
    action1 = agent1.get_action(env, env.task, received_data)
    actions = [action1, action1]

    obs, reward, done, info = env.step(actions)

    # blue : AI, red : Excel
    # env.render(filepath="control.txt.acmi", aircraft_id = "B0100")
    
    step_cnt += 1
    end_time = time.time()

    return step_cnt, cur_time, gun_idx
   

# @profile
def test_maneuver():
    global keyboard_input, gun_list

    prev_time = time.time()
    multi_state_var = [
        c.position_lat_geod_deg,            # 1. latitude   (unit: °)
        c.position_long_gc_deg,             # 0. lontitude  (unit: °)
        c.position_h_sl_m,                  # 2. altitude   (unit: m)
        c.attitude_roll_rad,                # 3. roll       (unit: rad)
        c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
        c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
        c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
        c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
        c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
        c.velocities_u_mps,                 # 9. v_u        (unit: m/s)
        c.velocities_v_mps,                 # 10. v_v        (unit: m/s)
        c.velocities_w_mps,                 # 11. v_w        (unit: m/s)
        c.accelerations_udot_ft_sec2,       # 12. a_n       (unit: ft/s2)
        c.accelerations_vdot_ft_sec2,       # 13. a_e       (unit: ft/s2)
        c.accelerations_wdot_ft_sec2        # 14. a_d       (unit: ft/s2)
    ]

    cur_time = time.time()

    while True:
        env = SingleCombatEnv(config_name='1v1/NoWeapon/test/opposite')
        obs = env.reset()
        agent1 = PursueAgent(agent_id=1)

        ego_hit_damage = [0, 0]
        step_cnt = 0

        # gun index
        gun_idx = 1
        damaged_munition_idx_list = [] # 이미 기 계산된 (맞은) 총알 index list
        
        gun_lock.acquire()
        gun_list = []
        gun_lock.release()

        def run_at_frequency(frequency, duration, step_cnt, cur_time, gun_idx):
            interval = 1.0 / frequency  # 실행 간격
            count = 0  # 실행 횟수 카운트
            start_time = time.perf_counter()

            while time.perf_counter() - start_time < duration:
                expected_time = start_time + (count + 1) * interval
                step_cnt, cur_time, gun_idx = while_main_loop(env, multi_state_var, ego_hit_damage, agent1, step_cnt, damaged_munition_idx_list, cur_time, gun_idx)
                count += 1
                sleep_time = expected_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            return count
    
        cnt = run_at_frequency(60, duration = 5, step_cnt = step_cnt, cur_time = cur_time, gun_idx = gun_idx)
        print("Final hz : " + str(cnt))
        break

 
if __name__ == '__main__':
    test_maneuver()
    