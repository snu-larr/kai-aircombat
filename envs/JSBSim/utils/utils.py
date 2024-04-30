import os
import yaml
import pymap3d
import numpy as np
import math

KNOT2METER = 0.514444
DEG2RAD = 3.14159265/180

def parse_config(filename):
    """Parse JSBSim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def LLA2NEU(lon, lat, alt, lon0=124.0, lat0=37.0, alt0=0):
    """Convert from Geodetic Coordinate System to NEU Coordinate System.

    Args:
        lon, lat, alt (float): target geodetic lontitude(°), latitude(°), altitude(m)
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (North, East, Up), unit: m
    """
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d])


def NEU2LLA(n, e, u, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from NEU Coordinate System to Geodetic Coordinate System.

    Args:
        n, e, u (float): target relative position w.r.t. North, East, Down
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (lon, lat, alt), unit: °, °, m
    """
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h])


def LLA2ECEF(lon, lat, alt):
    """
        lla to ecef (earth centered, earth fixed)
    
    """
    x, y, z = pymap3d.geodetic2ecef(lat, lon, alt)
    return np.array([x, y, z])

 
def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    # ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    ego_v = math.sqrt(ego_vx ** 2 + ego_vy ** 2 + ego_vz ** 2)
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    # enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    enm_v = math.sqrt(enm_vx ** 2 + enm_vy ** 2 + enm_vz ** 2)
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    # R = np.linalg.norm([delta_x, delta_y, delta_z])
    R = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    # ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    ego_AO = math.acos(max(min(proj_dist / (R * ego_v + 1e-8), 1), -1))

    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    # ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))
    ego_TA = math.acos(max(min(proj_dist / (R * enm_v + 1e-8), 1), -1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        # side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        temp = ego_vx * delta_y - ego_vy * delta_x
        if (temp == 0):
            side_flag = 0
        elif (temp > 0):
            side_flag = 1
        else:
            side_flag = -1
        return ego_AO, ego_TA, R, side_flag


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def world_ned_to_body_ned(world_n, world_e, world_d, roll, pitch, yaw):
    roll_mat = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    pitch_mat = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])

    yaw_mat = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    rot_matrix = np.dot(np.dot(roll_mat, pitch_mat), yaw_mat)
    return np.dot(rot_matrix, np.array([world_n, world_e, world_d]))

def body_ned_to_world_ned(body_x, body_y, body_z, roll, pitch, yaw):
    roll_mat = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    pitch_mat = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    yaw_mat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    rot_matrix = np.dot(np.dot(yaw_mat, pitch_mat), roll_mat)
    return np.dot(rot_matrix, np.array([body_x, body_y, body_z]))

def cal_azi_ele_from_euler(roll, pitch, yaw):
    """
    euler angle 을 받아서 azimute, elevation 을 만들어내는 함수

    Args : 
        roll, pitch, yaw `double` : rad
    
    Returns :
        azimute, elevation `double` : rad
    
    """
    roll_mat = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    pitch_mat = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    yaw_mat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    rot_matrix = np.dot(np.dot(yaw_mat, pitch_mat), roll_mat)

    elevation = np.arcsin(rot_matrix[2, 0])
    azimute = 0
    if (np.cos(elevation) != 0):
        azimute = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    
    return azimute, elevation

# ego 를 통해서 enm 에게 damage 를 입히는 정도
def hit_rate(ego_lat, ego_lon, ego_alt, ego_vn, ego_ve, ego_vd, enm_lat, enm_lon, enm_alt, enm_vn, enm_ve, enm_vd):
    std_lat, std_lon, std_alt = 37.36000, 127.91000, 0

    # print("1111", ego_alt, enm_alt)

    # deg deg m -> m m m
    ego_position = LLA2NEU(ego_lon, ego_lat, ego_alt, std_lon, std_lat, std_alt)
    enm_position = LLA2NEU(enm_lon, enm_lat, enm_alt, std_lon, std_lat, std_alt)

    ego_position[2] = -ego_position[2]
    enm_position[2] = -enm_position[2]

    ego_feature = np.hstack([ego_position, np.array([ego_vn, ego_ve, ego_vd])])
    enm_feature = np.hstack([enm_position, np.array([enm_vn, enm_ve, enm_vd])])
    
    AO, _, R = get_AO_TA_R(ego_feature, enm_feature)

    # print(AO, R)

    return _orientation_fn_custom(AO) * _distance_fn(R/1000)

# enm에 의해 ego 가 damage 를 입는 정도
def damage_rate(ego_lat, ego_lon, ego_alt, ego_vn, ego_ve, ego_vd, enm_lat, enm_lon, enm_alt, enm_vn, enm_ve, enm_vd):
    std_lat, std_lon, std_alt = 37.36000, 127.91000, 0

    # deg deg m -> m m m
    ego_position = LLA2NEU(ego_lon, ego_lat, ego_alt, std_lon, std_lat, std_alt)
    enm_position = LLA2NEU(enm_lon, enm_lat, enm_alt, std_lon, std_lat, std_alt)

    ego_feature = np.hstack([ego_position, np.array([ego_vn, ego_ve, ego_vd])])
    enm_feature = np.hstack([enm_position, np.array([enm_vn, enm_ve, enm_vd])])
    
    AO, _, R = get_AO_TA_R(enm_feature, ego_feature)
    return _orientation_fn(AO) * _distance_fn(R/1000)

def _orientation_fn(AO):
    if AO >= 0 and AO <= 0.5236:  # [0, pi/6]
        return 1 - AO / 0.5236
    elif AO >= -0.5236 and AO <= 0: # [-pi/6, 0]
        return 1 + AO / 0.5236
    return 0

def _orientation_fn_custom(AO):
    if AO >= 0 and AO <= 0.5236:  # [0, pi/4]
        return 1 - AO / 0.5236
    elif AO >= -0.5236 and AO <= 0: # [-pi/4, 0]
        return 1 + AO / 0.5236
    return 0

def _distance_fn(R):
    if R <=1: # [0, 1km]
        return 1
    elif R > 1 and R <= 10: # [1km, 10km] # [1, 3]
        return (10 - R) / 2.
    else:
        return 0