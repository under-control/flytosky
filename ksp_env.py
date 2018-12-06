import time
from gym import spaces
import numpy as np
import math

'''    
    Written in Whiteaster by Piotr Kubica
    Code could be done better, but the purpose was to make it similar to Open AI Gym environments  
    Another algorithms working on Gym should not make us do many changes
'''

turn_start_altitude = 250
turn_end_altitude = 45000
MAX_ALT = 45000

CONTINUOUS = True


def get_observation_space():

    low = np.array([
        0,
        - 1,
        - 1,
    ])

    high = np.array([
        1,
        1,
        1,
    ])

    observation_space = spaces.Box(low, high)
    return observation_space


def get_action_space():

    action_low = np.array([
        -1,
        -1
    ])

    action_high = np.array([
        1,
        1
    ])

    action_space = spaces.Box(action_low, action_high, dtype=np.float32)

    return action_space


class GameEnv(object):

    observation_space = get_observation_space()
    action_space = get_observation_space()

    def __init__(self, conn):
        self.conn = conn
        self.vessel = conn.space_center.active_vessel

        # Setting up streams for telemetry
        self.ut = conn.add_stream(getattr, conn.space_center, 'ut')
        self.altitude = conn.add_stream(getattr, self.vessel.flight(), 'mean_altitude')
        self.apoapsis = conn.add_stream(getattr, self.vessel.orbit, 'apoapsis_altitude')
        self.periapsis = conn.add_stream(getattr, self.vessel.orbit, 'periapsis_altitude')
        self.stage_2_resources = self.vessel.resources_in_decouple_stage(stage=2, cumulative=False)
        self.srb_fuel = conn.add_stream(self.stage_2_resources.amount, 'SolidFuel')
        self.pitch = conn.add_stream(getattr, self.vessel.flight(), 'pitch')
        self.heading = conn.add_stream(getattr, self.vessel.flight(), 'heading')
        self.roll = conn.add_stream(getattr, self.vessel.flight(), 'roll')
        self.g_force = conn.add_stream(getattr, self.vessel.flight(), 'g_force')
        self.frame = self.vessel.orbit.body.reference_frame
        self.vert_speed = conn.add_stream(getattr, self.vessel.flight(self.frame), 'vertical_speed')
        self.speed = conn.add_stream(getattr, self.vessel.flight(), 'velocity')
        self.lift = conn.add_stream(getattr, self.vessel.flight(), 'lift')
        self.crew = conn.add_stream(getattr, self.vessel, 'crew_count')
        self.parts = conn.add_stream(getattr, self.vessel.parts, 'all')

        self.prev_pitch = 90

        # Pre-launch setup
        self.vessel.control.sas = False
        self.vessel.control.rcs = False

        self.counter = 0
        self.altitude_max = 0

    def rotation_matrix(self):
        """
            changing quaternions to rotation matrix
            http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
            :param frame: reference_frame
            :return: [m00, m01, m02, m10, m11, m12, m20, m21, m22]
        """

        X, Y, Z, W = self.vessel.rotation(self.frame)

        xx = X * X
        xy = X * Y
        xz = X * Z
        xw = X * W

        yy = Y * Y
        yz = Y * Z
        yw = Y * W

        zz = Z * Z
        zw = Z * W

        m00 = 1 - 2 * (yy + zz)
        m01 = 2 * (xy - zw)
        m02 = 2 * (xz + yw)

        m10 = 2 * (xy + zw)
        m11 = 1 - 2 * (xx + zz)
        m12 = 2 * (yz - xw)

        m20 = 2 * (xz - yw)
        m21 = 2 * (yz + xw)
        m22 = 1 - 2 * (xx + yy)

        return m00, m01, m02, m10, m11, m12, m20, m21, m22

    def step(self, action):
        done = False

        action = action.tolist()

        self.conn.ui.message(str(action), duration=0.5)

        start_act = self.ut()

        '''
        https://krpc.github.io/krpc/python/api/space-center/control.html
        possible continuous actions: yaw[-1:1], pitch[-1:1], roll[-1:1], throttle[0:1],
        other: forward[-1:1], up[-1:1], right[-1:1], wheel_throttle[-1:1], wheel_steering[-1:1],
        '''

        if CONTINUOUS:
            self.vessel.control.pitch = action[0]
            self.vessel.control.yaw = action[1]

        else:
            self.vessel.control.pitch = 0
            self.vessel.control.yaw = 0
            self.vessel.control.roll = 0

            if action == 0:
                # do nothing action, wait
                pass
            if action == 1:
                self.vessel.control.pitch = -1
            if action == 2:
                self.vessel.control.pitch = 1
            if action == 3:
                self.vessel.control.yaw = -1
            if action == 4:
                self.vessel.control.yaw = 1

        while self.ut() - start_act <= 0.1:
            continue

        '''
        available states:
        https://krpc.github.io/krpc/python/api/space-center/flight.html
        https://krpc.github.io/krpc/python/api/space-center/orbit.html
        https://krpc.github.io/krpc/python/api/space-center/reference-frame.html
        '''

        state = self.get_state()

        reward = self.turn_reward()

        reward, done = self.epoch_ending(reward, done)

        self.conn.ui.message(str(state) + "reward: "+str(reward), duration=0.5)

        self.counter += 1

        if done:
            self.counter = 0
            reward -= 500 * (1 - self.altitude()/MAX_ALT)  # part of traveled distance

        if self.altitude() > self.altitude_max:
            self.altitude_max = self.altitude()

        return state, reward, done, {}

    def epoch_ending(self, reward, done):

        if self.altitude() >= MAX_ALT:                      # expected altitude
            reward += 1000
            done = True
            print('reached max altitude at: ', self.altitude())

        elif self.periapsis() > 150000:                     # extra win
            reward += 1000000
            done = True
            print('you won, congrats!')

        elif self.crew() == 0:                              # check if crew is alive
            done = True
            print('crew is dead :(')

        elif self.counter >= 200000:                        # finish after 200k steps
            done = True
            print('finished after 200k steps')

        elif self.altitude() < 20:                          # finish if rocket was down
            done = True
            print('altitude < 20')

        elif self.pitch() < 60 and self.altitude() <= 5000:
            done = True
            print('pitch < 60 and altitude <= 5000')

        elif self.pitch() < -5 and self.altitude() <= 45000:
            done = True
            print('pitch < -5 and altitude <= 45000')

        elif len(self.parts()) <= 1:                        # if self.lift() == (-0.0, 0.0, -0.0):
            done = True
            print('parts = ', len(self.parts()))

        elif self.vert_speed() < - 1000:
            done = True
            print('you started falling really fast')

        elif self.counter >= 60 and self.altitude() < 88:   # check if rocket started
            done = True
            print('rocket did not start within 60 moves')

        return reward, done

    def reset(self, conn):
        """
        revivekerbals is a save file /GOG/KSP/game/saves/kill
        to run the code you will need to dowload it from
        https://drive.google.com/file/d/1khVq9FyAcpDxxF5_71i_3nXNvDf79OWl
        :param conn: krpc.connection
        :return: state
        """
        self.altitude_max = 0

        quicksave_name = "revivekerbals"

        try:
            self.conn.space_center.load(quicksave_name)
        except Exception as ex:
            print("Error:", ex)
            print("Add \"kill\" save to your saves directory")
            exit("You have no quicksave named {}. Terminating.".format(quicksave_name))

        time.sleep(3)

        self.__init__(conn)  # need to reset

        self.conn.space_center.physics_warp_factor = 0

        state = self.get_state()

        return state

    def get_state(self):

        state = [
            ((self.altitude() + 0.2) / MAX_ALT) / 1.2,
            math.sin(math.radians(self.heading())) * (90 - self.pitch()) / 90,
            math.cos(math.radians(self.heading())) * (90 - self.pitch()) / 90,
        ]

        return state

    def _normalize(self, feature):
        return 1 / (1 + round(math.pow(math.e, -feature), 5))

    def difference(self):
        fractal = (self.altitude() - turn_start_altitude) \
                  / (turn_end_altitude - turn_start_altitude)
        turn_angle = 90 - fractal * 90
        deviation = abs(turn_angle - self.pitch())
        return deviation

    def turn_reward(self):

        reward = 0

        if str(self.vessel.situation) == "VesselSituation.flying":

            deviation = self.difference()

            reward = 10 * (self.pitch() - self.prev_pitch)
            if deviation < 10:
                reward += 10 - deviation
            else:
                reward -= 1

            self.prev_pitch = self.pitch()

        return reward

    def add_throttle(self, reward):
        return reward + self.vessel.control.throttle*10

    def include_velocity(self, reward):
        return reward * self.vert_speed()**(1/3)

    def include_heading(self, reward):
        if 75 <= self.heading() <= 90:
            reward = reward + (self.heading() - 75) / 10
        elif 90 < self.heading() <= 105:
            reward = reward + (105 - self.heading()) / 10
        return reward

    def include_orbit(self, reward):
        if str(self.vessel.situation) == "VesselSituation.orbital" or\
                str(self.vessel.situation) == "VesselSituation.sub_orbital":
            reward = self.periapsis()**(1/9)
        return reward

    def lower_first_rewards(self, reward):
        if self.counter < 10:
            reward = reward / (10 - self.counter)
        return reward

    def activate_engine(self):
        self.vessel.control.throttle = 1.0
        self.vessel.control.activate_next_stage()
        time.sleep(3.5)

    def get_altitude(self):
        return round(self.altitude_max, 2)


