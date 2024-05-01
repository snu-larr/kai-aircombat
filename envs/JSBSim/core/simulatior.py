import os
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import Literal, Union, List

import jsbsim
from .catalog import Property, Catalog
from ..utils.utils import get_root_dir, LLA2NEU, NEU2LLA
import re
            
TeamColors = Literal["Red", "Blue", "Green", "Violet", "Orange"]


class BaseSimulator(ABC):

    def __init__(self, uid: str, color: TeamColors, dt: float):
        """Constructor. Creates an instance of simulator, initialize all the available properties.

        Args:
            uid (str): 5-digits hexadecimal numbers for unique identification.
            color (TeamColors): use different color strings to represent diferent teams
            dt (float): simulation timestep. Default = `1 / 60`.
        """
        self.__uid = uid
        self.__color = color
        self.__dt = dt
        self.model = ""
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        self._accelerations = np.zeros(3)
        logging.debug(f"{self.__class__.__name__}:{self.__uid} is created!")

    @property
    def uid(self) -> str:
        return self.__uid

    @property
    def color(self) -> str:
        return self.__color

    @property
    def dt(self) -> float:
        return self.__dt

    def get_geodetic(self):
        """(lontitude, latitude, altitude), unit: °, m"""
        return self._geodetic

    def get_position(self):
        """(north, east, up), unit: m"""
        return self._position

    def get_rpy(self):
        """(roll, pitch, yaw), unit: rad"""
        return self._posture

    def get_velocity(self):
        """(v_north, v_east, v_up), unit: m/s"""
        return self._velocity
    
    def get_acceleration(self):
        """(a_north, a_east, a_down), unit: G"""
        return self._accelerations


    def reload(self):
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        self._accelerations = np.zeros(3)

    @abstractmethod
    def run(self, **kwargs):
        pass

    def log(self):
        lon, lat, alt = self.get_geodetic()
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"
        log_msg += f"Color={self.color}"
        return log_msg

    @abstractmethod
    def close(self):
        pass

    def __del__(self):
        logging.debug(f"{self.__class__.__name__}:{self.uid} is deleted!")

class UnControlAircraftSimulator(BaseSimulator):

    ALIVE = 0
    CRASH = 1       # low altitude / extreme state / overload
    SHOTDOWN = 2    # missile attack

    def __init__(self,
                 uid: str = "R0100",
                 color: TeamColors = "Red",
                 model: str = 'f16',
                 init_state: dict = {},
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60, **kwargs):
        
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.mode = "Rule"
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self.__status = UnControlAircraftSimulator.ALIVE
        for key, value in kwargs.items():
            if key == 'num_missiles':
                self.num_missiles = value  # type: int
                self.num_left_missiles = self.num_missiles  # type: int
        
        # fixed simulator links
        self.partners = []  # type: List[AircraftSimulator]
        self.enemies = []   # type: List[AircraftSimulator]
        
        # dictionary
        self.property_dict = {}

        # ac state
        self.ac_id_state = {}
        self.ac_id_lla = {}

        self.nearest_munition = []

        # initialize simulator
        self.reload()

    @property
    def is_alive(self):
        return self.__status == UnControlAircraftSimulator.ALIVE

    @property
    def is_crash(self):
        return self.__status == UnControlAircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        return self.__status == UnControlAircraftSimulator.SHOTDOWN

    def crash(self):
        self.__status = UnControlAircraftSimulator.CRASH

    def shotdown(self):
        self.__status = UnControlAircraftSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """Reload aircraft simulator
        """
        super().reload()

        # reset temp simulator links
        self.bloods = 100
        self.__status = UnControlAircraftSimulator.ALIVE
        self.num_left_missiles = self.num_missiles

        # assign new properties
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin

    def run(self):
        pass

    def close(self):
        self.partners = []
        self.enemies = []

    def get_property_values(self, props):
        ret = []
        for prop in props:
            try:
                ret.append(self.property_dict[prop.name_jsbsim])
            except:
                print("Key : " + str([prop.name_jsbsim]))
        
        return ret
    
    def set_ac_state(self, data, id):
        self.ac_id_state[id] = data

    def set_ac_lla(self, data, id):
        self.ac_id_lla[id] = data

    def _update_lla_properties(self, agent_id = None):
        if (agent_id == None):
            raise Exception("NO AGNET ID")

        lon, lat, alt = self.ac_id_lla[agent_id]

        self.property_dict[Catalog.position_long_gc_deg.name_jsbsim] = float(lon)
        self.property_dict[Catalog.position_lat_geod_deg.name_jsbsim] = float(lat)
        self.property_dict[Catalog.position_h_sl_m.name_jsbsim] = float(alt)
        self._geodetic[:] = [lon, lat, alt]
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)

        self.property_dict[Catalog.attitude_roll_rad.name_jsbsim] = 0
        self.property_dict[Catalog.attitude_pitch_rad.name_jsbsim] = 0
        self.property_dict[Catalog.attitude_heading_true_rad.name_jsbsim] = 0
        self._posture[:] = [0, 0, 0]

        self.property_dict[Catalog.velocities_v_north_mps.name_jsbsim] = 0
        self.property_dict[Catalog.velocities_v_east_mps.name_jsbsim] = 0
        self.property_dict[Catalog.velocities_v_down_mps.name_jsbsim] = 0
        self._velocity[:] = [0, 0, 0]

        self.property_dict[Catalog.velocities_u_mps.name_jsbsim] = 0
        self.property_dict[Catalog.velocities_v_mps.name_jsbsim] = 0
        self.property_dict[Catalog.velocities_w_mps.name_jsbsim] = 0
        self.property_dict[Catalog.velocities_vc_mps.name_jsbsim] = 0

    def _update_properties(self, agent_id = None):
        ################################

        if (agent_id == None):
            raise Exception("NO AGNET ID")

        lon, lat, alt, r, p, y, vn, ve, vd, vbx, vby, vbz, vc = self.ac_id_state[agent_id]

        self.property_dict[Catalog.position_long_gc_deg.name_jsbsim] = float(lon)
        self.property_dict[Catalog.position_lat_geod_deg.name_jsbsim] = float(lat)
        self.property_dict[Catalog.position_h_sl_m.name_jsbsim] = float(alt)
        self._geodetic[:] = [lon, lat, alt]
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)

        self.property_dict[Catalog.attitude_roll_rad.name_jsbsim] = float(r)
        self.property_dict[Catalog.attitude_pitch_rad.name_jsbsim] = float(p)
        self.property_dict[Catalog.attitude_heading_true_rad.name_jsbsim] = float(y)
        self._posture[:] = [r, p, y]

        self.property_dict[Catalog.velocities_v_north_mps.name_jsbsim] = float(vn)
        self.property_dict[Catalog.velocities_v_east_mps.name_jsbsim] = float(ve)
        self.property_dict[Catalog.velocities_v_down_mps.name_jsbsim] = float(vd)
        self._velocity[:] = [vn, ve, vd]

        self.property_dict[Catalog.velocities_u_mps.name_jsbsim] = float(vbx)
        self.property_dict[Catalog.velocities_v_mps.name_jsbsim] = float(vby)
        self.property_dict[Catalog.velocities_w_mps.name_jsbsim] = float(vbz)
        self.property_dict[Catalog.velocities_vc_mps.name_jsbsim] = float(vc)

        # self.property_dict[Catalog.accelerations_n_pilot_x_norm.name_jsbsim] = float(an)
        # self.property_dict[Catalog.accelerations_n_pilot_y_norm.name_jsbsim] = float(ae)
        # self.property_dict[Catalog.accelerations_n_pilot_z_norm.name_jsbsim] = float(ad)
        # self._accelerations[:] = [an, ae, ad]
        

class AircraftSimulator(BaseSimulator):
    """A class which wraps an instance of JSBSim and manages communication with it.
    """

    ALIVE = 0
    CRASH = 1       # low altitude / extreme state / overload
    SHOTDOWN = 2    # missile attack

    def __init__(self,
                 uid: str = "A0100",
                 color: TeamColors = "Red",
                 model: str = 'f16',
                 init_state: dict = {},
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60, **kwargs):
        """Constructor. Creates an instance of JSBSim, loads an aircraft and sets initial conditions.

        Args:
            uid (str): 5-digits hexadecimal numbers for unique identification. Default = `"A0100"`.
            color (TeamColors): use different color strings to represent diferent teams
            model (str): name of aircraft to be loaded. Default = `"f16"`.
                model path: './data/aircraft_name/aircraft_name.xml'
            init_state (dict): dict mapping properties to their initial values. Input empty dict to use a default set of initial props.
            origin (tuple): origin point (longitude, latitude, altitude) of the Global Combat Field. Default = `(120.0, 60.0, 0.0)`
            sim_freq (int): JSBSim integration frequency. Default = `60`.
        """
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.mode = "AI"
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        for key, value in kwargs.items():
            if key == 'num_missiles':
                self.num_missiles = value  # type: int
                self.num_left_missiles = self.num_missiles  # type: int
        # fixed simulator links
        self.partners = []  # type: List[AircraftSimulator]
        self.enemies = []   # type: List[AircraftSimulator]
        # temp simulator links
        # initialize simulator

        self.nearest_munition = []
        self.reload()

    @property
    def is_alive(self):
        return self.__status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        return self.__status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        return self.__status == AircraftSimulator.SHOTDOWN

    def crash(self):
        self.__status = AircraftSimulator.CRASH

    def shotdown(self):
        self.__status = AircraftSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """Reload aircraft simulator
        """
        super().reload()

        # reset temp simulator links
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self.num_left_missiles = self.num_missiles

        # load JSBSim FDM
        self.jsbsim_exec = jsbsim.FGFDMExec(os.path.join(get_root_dir(), 'data'))
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.model)
        Catalog.add_jsbsim_props(self.jsbsim_exec.query_property_catalog(""))
        self.jsbsim_exec.set_dt(self.dt)
        self.clear_defalut_condition()

        # test
        # self.jsbsim_exec.print_property_catalog()

        # assign new properties
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)
        success = self.jsbsim_exec.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        # propulsion init running
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()
        # update inner property
        self._update_properties()

    def clear_defalut_condition(self):
        default_condition = {
            Catalog.ic_long_gc_deg: 120.0,  # geodesic longitude [deg]
            Catalog.ic_lat_geod_deg: 60.0,  # geodesic latitude  [deg]
            Catalog.ic_h_sl_ft: 20000,      # altitude above mean sea level [ft]
            Catalog.ic_psi_true_deg: 0.0,   # initial (true) heading [deg] (0, 360)
            Catalog.ic_u_fps: 800.0,        # body frame x-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_v_fps: 0.0,          # body frame y-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_w_fps: 0.0,          # body frame z-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_p_rad_sec: 0.0,      # roll rate  [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_q_rad_sec: 0.0,      # pitch rate [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_r_rad_sec: 0.0,      # yaw rate   [rad/s]  (-2 * pi, 2 * pi)
            Catalog.ic_roc_fpm: 0.0,        # initial rate of climb [ft/min]
            Catalog.ic_terrain_elevation_ft: 0,

            Catalog.wind_north_fps : 0.0,
            Catalog.wind_east_fps : 0.0,
            Catalog.wind_down_fps : 0.0,
        }
        for prop, value in default_condition.items():
            self.set_property_value(prop, value)

    def run(self):
        """Runs JSBSim simulation until the agent interacts and update custom properties.

        JSBSim monitors the simulation and detects whether it thinks it should
        end, e.g. because a simulation time was specified. False is returned
        if JSBSim termination criteria are met.

        Returns:
            (bool): False if sim has met JSBSim termination criteria else True.
        """
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()
            result = self.jsbsim_exec.run()
            if not result:
                raise RuntimeError("JSBSim failed.")
            self._update_properties()
            return result
        else:
            return True

    def close(self):
        """ Closes the simulation and any plots. """
        if self.jsbsim_exec:
            self.jsbsim_exec = None
        self.partners = []
        self.enemies = []

    def set_ac_state(self, data, id):
        pass

    def set_ac_lla(self, data, id):
        pass

    def _update_lla_properties(self, agent_id = None):
        pass

    def _update_properties(self):
        # update position
        self._geodetic[:] = self.get_property_values([
            Catalog.position_long_gc_deg,
            Catalog.position_lat_geod_deg,
            Catalog.position_h_sl_m
        ])
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        # update posture
        self._posture[:] = self.get_property_values([
            Catalog.attitude_roll_rad,
            Catalog.attitude_pitch_rad,
            Catalog.attitude_heading_true_rad,
        ])
        # update velocity
        self._velocity[:] = self.get_property_values([
            Catalog.velocities_v_north_mps,
            Catalog.velocities_v_east_mps,
            Catalog.velocities_v_down_mps,
        ])

        # update acc
        self._accelerations[:] = self.get_property_values([
            Catalog.accelerations_udot_ft_sec2,
            Catalog.accelerations_vdot_ft_sec2,
            Catalog.accelerations_wdot_ft_sec2,
        ])

    def get_sim_time(self):
        """ Gets the simulation time from JSBSim, a float. """
        return self.jsbsim_exec.get_sim_time()

    def get_property_values(self, props):
        """Get the values of the specified properties

        :param props: list of Properties

        : return: NamedTupl e with properties name and their values
        """
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        """Set the values of the specified properties

        :param props: list of Properties

        :param values: list of float
        """
        if not len(props) == len(values):
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        """Get the value of the specified property from the JSBSim simulation

        :param prop: Property

        :return : float
        """
        if isinstance(prop, Property):
            if prop.access == "R":
                if prop.update:
                    prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def set_property_value(self, prop, value):
        """Set the values of the specified property

        :param prop: Property

        :param value: float
        """
        # set value in property bounds
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max

            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)

            if "W" in prop.access:
                if prop.update:
                    prop.update(self)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

class UnControlSAMSimulator(BaseSimulator):

    ALIVE = 0
    CRASH = 1       # low altitude / extreme state / overload
    SHOTDOWN = 2    # missile attack

    def __init__(self,
                 uid: str = "S0100",
                 color: TeamColors = "Red",
                 model: str = 'SAM',
                 init_state: dict = {},
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60, **kwargs):
        
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.mode = "Rule"
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self.__status = UnControlSAMSimulator.ALIVE
        
        # dictionary
        self.property_dict = {}

        # recv data
        self.sam_id_state = {}

        # initialize simulator
        self.reload()

    @property
    def is_alive(self):
        return self.__status == UnControlSAMSimulator.ALIVE

    @property
    def is_crash(self):
        return self.__status == UnControlSAMSimulator.CRASH

    @property
    def is_shotdown(self):
        return self.__status == UnControlSAMSimulator.SHOTDOWN

    def crash(self):
        self.__status = UnControlSAMSimulator.CRASH

    def shotdown(self):
        self.__status = UnControlSAMSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """Reload aircraft simulator
        """
        super().reload()

        # reset temp simulator links
        self.bloods = 100
        self.__status = UnControlSAMSimulator.ALIVE

        # assign new properties
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin

    def run(self):
        pass

    def close(self):
        pass

    def get_property_values(self, props):
        ret = []
        for prop in props:
            try:
                ret.append(self.property_dict[prop.name_jsbsim])
            except:
                print("Key : " + str([prop.name_jsbsim]))
        
        return ret
    
    def set_sam_state(self, data, sam_id):
        self.sam_id_state[sam_id] = data

    def _update_properties(self, sam_id = None):
        ################################
        if (sam_id == None):
            raise Exception("NO SAM ID")

        lon, lat, alt = self.sam_id_state[sam_id]

        self.property_dict[Catalog.position_long_gc_deg.name_jsbsim] = float(lon)
        self.property_dict[Catalog.position_lat_geod_deg.name_jsbsim] = float(lat)
        self.property_dict[Catalog.position_h_sl_m.name_jsbsim] = float(alt)
        self._geodetic[:] = [lon, lat, alt]
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)

    def log(self):
        lon, lat, alt = self.get_geodetic()
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{0.0}|{0.0}|{0.0},"
        log_msg += "Name=ZU-23," # TODO : SAM model?
        log_msg += f"Color={self.color}"
        return log_msg