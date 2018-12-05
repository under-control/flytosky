import krpc
import time
from ksp_env import GameEnv
import math
import numpy as np
import socket

# ip can be also local 127.0.0.1
ip = socket.gethostbyname(socket.gethostname())

conn = krpc.connect(name='Tracker', address=ip)
env = GameEnv(conn)
vessel = env.vessel

frame = vessel.orbit.body.reference_frame
vert_speed = conn.add_stream(getattr, vessel.flight(frame), 'vertical_speed')


def states():
        message = ('sin',round(math.sin((math.radians(env.heading()))) * (90 - env.pitch()) / 90, 2),
                   'cos', round(math.cos((math.radians(env.heading()))) * (90 - env.pitch()) / 90, 2),
                   'sinr', round(math.sin((math.radians(env.heading() + env.roll()))) * (90 - env.pitch()) / 90, 2),
                   'cosr', round(math.cos((math.radians(env.heading() + env.roll()))) * (90 - env.pitch()) / 90, 2),
                   "p", round(env.pitch(), 2),
                   "h", round(env.heading(), 2),
                   "r", round(env.roll(), 2),
                   " a " + str(round(env.altitude(), 2)),
                   "time " + str(round(env.ut(), 2))
                   )
        return str(message)


def velocities():
    print('(%.1f, %.1f, %.1f)' % vessel.position(vessel.orbit.body.reference_frame))
    print('(%.2f, %.2f, %.2f, %.2f)' % vessel.rotation(vessel.orbit.body.reference_frame))
    print('(%.1f, %.1f, %.1f)' % vessel.velocity(vessel.orbit.body.reference_frame))
    print('(%.1f, %.1f, %.1f)' % vessel.angular_velocity(vessel.orbit.body.reference_frame))
    print('(%.1f, %.1f, %.1f)' % vessel.direction(vessel.orbit.body.reference_frame))


def tracking_to_csv():
    with open('flight.csv', mode='a', newline='') as f:
        f.write(str(env.ut()) + ',' +
                str(env.altitude()) + ',' +
                str(env.heading()) + ',' +
                str(env.roll()) +
                "\n")


def more_info_printer():
    print("t", round(env.ut(),2),
          "a",round(env.altitude(),2),
          "p",round( env.pitch(),2),
          "h",round(env.heading(),2),
          "r",round( env.roll(),2),
          "k",round( env.altitude_keeper,2),
          "apo", round(vessel.orbit.apoapsis,2),
          "per", round(vessel.orbit.periapsis,2),
          "apo_alt", round(vessel.orbit.apoapsis_altitude, 2),
          "per_alt", round(vessel.orbit.periapsis_altitude, 2),
          "g_force", round(vessel.flight().g_force, 2),
          "vertspeed", round(vert_speed(), 2),
          "parts", len(env.parts()),
          # "stage", vessel.parts.in_stage(2),
          "cp", env.vessel.control.pitch,
          "cy", env.vessel.control.yaw,
          "cr", env.vessel.control.roll,
          vessel.situation,
          # angle, d, v
          )

    print("rotation ",vessel.rotation(frame))
    print("direction ", vessel.direction(frame))
    print("prograde ", vessel.prograde(frame))


def angle_of_attack():
    d = vessel.direction(vessel.orbit.body.reference_frame)
    v = vessel.velocity(vessel.orbit.body.reference_frame)

    d = list(np.around(np.array(d), 2))
    v = list(np.around(np.array(v), 2))

    # Compute the dot product of d and v
    dotprod = d[0] * v[0] + d[1] * v[1] + d[2] * v[2]

    # Compute the magnitude of v
    vmag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    # Note: don't need to magnitude of d as it is a unit vector

    # Compute the angle between the vectors
    angle = 0
    if dotprod > 0:
        angle = abs(math.acos(dotprod / vmag) * (180.0 / math.pi))

    # print('Angle of attack = %.1f degrees' % angle, 'direction ', d, 'velocity', v )

    return d, v, angle


if __name__ == "__main__":
    conn.ui.message('you are being tracked ', duration=5.0)
    print(ip, "vessel.orbit.body: ", vessel.orbit.body)
    while True:
        msg = states()
        # here you can all other methods to see and understand more data
        print(msg)
        conn.ui.message(msg)
        time.sleep(1)





