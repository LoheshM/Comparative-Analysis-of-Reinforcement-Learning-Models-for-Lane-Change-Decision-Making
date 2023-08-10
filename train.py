import sys
import glob
import os
import time
import random
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
# import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from datetime import datetime
from controller import VehiclePIDController

from DQN import *

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

VEHICLE_VEL = 45
LEFT_LANE = 0
CENTER_LANE = 1
RIGHT_LANE = 2
CELL_LEN = 2.0
STEER_AMT = 1.0
offset = 1

class Player():
    STEER_AMT = 1.0

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town05")
        self.blp_lib = self.world.get_blueprint_library()
        self.model_3 = self.blp_lib.filter("model3")[0]
        # # Spawning ego car
        self.spectator = self.world.get_spectator()
        self.initial_transform = carla.Transform(carla.Location(x=-22.89981460571289, y=-203.97962951660156, z=2.79974365234375),
                                                 carla.Rotation(pitch=0.000000, yaw=179.757080078125, roll=0.000000))

        self.vehicle = self.world.try_spawn_actor(self.model_3, self.initial_transform)

        self.lane_index = CENTER_LANE
        self.vel_ref = VEHICLE_VEL
        self.current_pos = self.vehicle.get_transform().location
        self.past_pos = self.vehicle.get_transform().location
        self.vehicle_list = []
        self.actor_list = []
        self.collision_hist = []
        self.steps_since_last_vehicle_spawn = 0
        self.pos = None
        self.location_history = [self.vehicle.get_location()]
        self.actor_location_history = []
        self.tm_port = 9500
        tm_is_initialized = False
        while not tm_is_initialized:
            try:
                self.tm = self.client.get_trafficmanager(self.tm_port)
                self.tm.global_percentage_speed_difference(10.0)  # set the global speed limitation
                self.tm.set_synchronous_mode(True)
                tm_is_initialized = True
            except Exception as err:
                # print("Caught exception during traffic manager creation: ")
                # print(err)
                self.tm_port += 1
                # print("Trying with port {}...".format(self.tm_port))
        # self.state = None

    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.vehicle_list = []
        self.lane_index = CENTER_LANE
        self.vel_ref = VEHICLE_VEL
        self.waypointsList = []
        self.actor_location_history = []
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 1/20
        self.world.apply_settings(self.settings)

        self.world = self.client.load_world("Town05")
        self.world.tick()
        
        self.spectator = self.world.get_spectator()
        self.vehicle = None
        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.initial_transform)
        self.world.tick()
        self.location_history = [self.vehicle.get_location()]

        # self.current_pos = self.vehicle.get_transform().location
        # self.past_pos = self.vehicle.get_transform().location

        # self.vehicle_list.append(self.vehicle)
        self.actor_list.append(self.vehicle)
        self.actor_list.append(self.spectator)
        self.steps_since_last_vehicle_spawn = 0

        # collision sensor
        col_sensor = self.blp_lib.find('sensor.other.collision')
        col_transform = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))
        self.ego_col = self.world.spawn_actor(col_sensor, col_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.ego_col.listen(lambda event: self.collision_data(event))

        # lane invasion sensor
        # lane_bp = self.blp_lib.find('sensor.other.lane_invasion')
        # lane_transform = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))
        # self.ego_lane = self.world.spawn_actor(lane_bp, lane_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        # self.ego_lane.listen(lambda lane: lane_callback(lane))

        # obstacle sensor
        # obs_bp = self.blp_lib.find('sensor.other.obstacle')
        # obs_bp.set_attribute("only_dynamics", str(True))
        # obs_transform = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))
        # self.ego_obs = self.world.spawn_actor(obs_bp, obs_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        # self.ego_obs.listen(lambda obs: obs_callback(obs))

        # while self.state is None:
        #     time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        grid = np.ones((51, 3))
        ego_loc = self.vehicle.get_location()
        ego_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
        for actor in self.vehicle_list:
            actor_vel = actor.get_velocity()
            actor_speed = (3.6 * math.sqrt(actor_vel.x ** 2 + actor_vel.y ** 2 + actor_vel.z ** 2)) / 100.0
            actor_loc = actor.get_location()
            dist = actor_loc.distance(ego_loc)
            if dist > 60: continue
            
            cell_delta = int(dist // CELL_LEN)

            previous_reference_loc = ego_wp.previous(15)[0].transform.location
            is_behind = actor_loc.distance(previous_reference_loc) <= 15
            if is_behind: cell_delta *= -1

            # find actor's lane
            actor_lane = None
            left_lane_wp = None
            center_lane_wp = None
            right_lane_wp = None

            if self.lane_index == LEFT_LANE:
                left_lane_wp = ego_wp
                center_lane_wp = ego_wp.get_right_lane()
                right_lane_wp = center_lane_wp.get_right_lane()
            elif self.lane_index == CENTER_LANE:
                left_lane_wp = ego_wp.get_left_lane()
                center_lane_wp = ego_wp
                right_lane_wp = ego_wp.get_right_lane()
            elif self.lane_index == RIGHT_LANE:
                left_lane_wp = ego_wp.get_left_lane().get_left_lane()
                center_lane_wp = ego_wp.get_left_lane()
                right_lane_wp = ego_wp

            left_lane_next_wp = left_lane_wp.previous(30)[0]
            center_lane_next_wp = center_lane_wp.previous(30)[0]
            right_lane_next_wp = right_lane_wp.previous(30)[0]


            # TODO: we can speed this up by using "dist" for the range, but we need to handle next/previous search differently
            for i in range(1, 95):
                # print(left_lane_next_wp.transform.location.x, left_lane_next_wp.transform.location.y)
                # print(left_lane_next_wp.transform.location)
                # print(actor_loc.distance(left_lane_next_wp.transform.location))
                if actor_loc.distance(left_lane_next_wp.transform.location) < 1:
                    actor_lane = LEFT_LANE
                    break
                elif actor_loc.distance(center_lane_next_wp.transform.location) < 1:
                    actor_lane = CENTER_LANE
                    break
                elif actor_loc.distance(right_lane_next_wp.transform.location) < 1:
                    actor_lane = RIGHT_LANE
                    break
            
                left_lane_next_wp = left_lane_next_wp.next(1)[0]
                center_lane_next_wp = center_lane_next_wp.next(1)[0]
                right_lane_next_wp = right_lane_next_wp.next(1)[0]
                
            # if we didn't find the actor's lane, it must >30m behind the ego car
            if actor_lane == None:
                continue
            grid[(31 - cell_delta):(35 - cell_delta), actor_lane] = actor_speed
            # TODO: Fill in the grid with actors' velocities
        vel = self.vehicle.get_velocity()
        grid[31:35, self.lane_index] = (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)) / 100.0

        self.state = np.zeros((45, 3))
        self.state[:, :] = grid[3:48, :]

        self.tm_port = 9500
        tm_is_initialized = False
        while not tm_is_initialized:
            try:
                self.tm = self.client.get_trafficmanager(self.tm_port)
                self.tm.global_percentage_speed_difference(10.0)  # set the global speed limitation
                self.tm.set_synchronous_mode(True)
                tm_is_initialized = True
            except Exception as err:
                # print("Caught exception during traffic manager creation: ")
                # print(err)
                self.tm_port += 1
                # print("Trying with port {}...".format(self.tm_port))

        return self.state

    def step(self, action):
        dt = 1.0 / 20.0
        args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
        args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}
        offset = 1
        max_throt = 0.75
        max_brake = 0.3
        max_steer = 0.8
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        reward = 0
        
        if len(self.collision_hist) != 0:
            # print("step func detected collision")
            done = True
            reward -= 10
            return self.state, self.pos, reward, done, None
        elif kmh < VEHICLE_VEL:
            done = False
            reward -= 1
        else:
            done = False
            reward += 3

        self.controller = VehiclePIDController(self.vehicle,
                                        args_lateral=args_lateral_dict,
                                        args_longitudinal=args_longitudinal_dict,
                                        # offset=offset,
                                        max_throttle=max_throt,
                                        max_brake=max_brake,
                                        max_steering=max_steer)

        if action == 0:
            self.do_follow_lane()
        elif action == 1:
            if (self.lane_index != LEFT_LANE):
                self.do_left_lane_change()
            else:
                self.do_follow_lane()
                reward -= 3
        elif action == 2:
            if (self.lane_index != RIGHT_LANE):
                self.do_right_lane_change()
            else:
                self.do_follow_lane()
                reward -= 3

        if self.steps_since_last_vehicle_spawn >= 4:
            self.spawn_vehicle()
            self.steps_since_last_vehicle_spawn = 0
        self.steps_since_last_vehicle_spawn += 1
        new_state = self.get_state_representation(self.vehicle_list)

        if new_state is None:
            done = True
            return self.state, self.pos, reward, done, None
        else:
            self.state = new_state

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        ego_speed = player.vehicle.get_velocity()
        ego_lane = player.lane_index

        self.pos = [ego_speed / 25, 0, 0]
        if ego_lane == LEFT_LANE:
            self.pos = [ego_speed / 25, 1, 0]
        elif ego_lane == CENTER_LANE:
            self.pos = [ego_speed / 25, 1, 1]
        elif ego_lane == RIGHT_LANE:
            self.pos = [ego_speed / 25, 0, 1]
        self.pos = np.reshape(self.pos, [1, 3])

        return self.state, self.pos, reward, done, None

    def spawn_vehicle(self):
        ego_loc = self.vehicle.get_location()
        ego_wp = self.world.get_map().get_waypoint(ego_loc)
        spawn_lane = random.choice([LEFT_LANE, CENTER_LANE, RIGHT_LANE])
        if self.lane_index == LEFT_LANE:
            center_lane_wp = ego_wp.get_right_lane()
        elif self.lane_index == CENTER_LANE:
            center_lane_wp = ego_wp
        elif self.lane_index == RIGHT_LANE:
            center_lane_wp = ego_wp.get_left_lane()

        try:
            if spawn_lane == LEFT_LANE:
                spawn_wp = center_lane_wp.get_left_lane().next(70)[0]
            elif spawn_lane == CENTER_LANE:
                spawn_wp = center_lane_wp.next(70)[0]
            elif spawn_lane == RIGHT_LANE:
                spawn_wp = center_lane_wp.get_right_lane().next(70)[0]

            spawn_transform = spawn_wp.transform
            _loc = spawn_transform.location
            spawn_transform.location = carla.Location(_loc.x, _loc.y, _loc.z + 0.5)
            actor_vehicle = self.world.spawn_actor(self.model_3, spawn_transform)
            # print("spawned actor")
            self.vehicle_list.append(actor_vehicle)
            self.actor_list.append(actor_vehicle)
            actor_vehicle.set_autopilot(True, self.tm_port)  # you can get those functions detail in carla document
            # self.tm.ignore_lights_percentage(actor_vehicle, 0)
            self.tm.distance_to_leading_vehicle(actor_vehicle, 50)
            # self.tm.vehicle_percentage_speed_difference(actor_vehicle, -20)
        except:
            # print('failed appending new vehicle actor!')  # if failed, print the hints.
            pass

    # sensor print functions
    def collision_data(self, event):
        # print("Collision detected:\n" + str(event) + '\n')
        self.collision_hist.append(event)
        # print(event)
        # print(event.other_actor)
        # print(event.other_actor.type_id)
        self.ego_col.stop()

    def lane_callback(lane):
        pass
        # print("Lane invasion detected:\n"+str(lane)+'\n')

    def obs_callback(obs):
        pass
        # print("Obstacle detected:\n"+str(obs)+'\n')

    def dist_to_waypoint(self, waypoint):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_x = vehicle_transform.location.x
        vehicle_y = vehicle_transform.location.y
        waypoint_x = waypoint.transform.location.x
        waypoint_y = waypoint.transform.location.y
        return math.sqrt((vehicle_x - waypoint_x)**2 + (vehicle_y - waypoint_y)**2)
    
    def go_to_waypoint(self, waypoint, draw_waypoint = True, threshold = 2):
        if draw_waypoint : 
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                                       color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                                       persistent_lines=True)
        
        current_pos_np = np.array([self.current_pos.x,self.current_pos.y])
        past_pos_np = np.array([self.past_pos.x,self.past_pos.y])
        waypoint_np = np.array([waypoint.transform.location.x,waypoint.transform.location.y])
        vec2wp = waypoint_np - current_pos_np
        motion_vec = current_pos_np - past_pos_np
        dot = np.dot(vec2wp, motion_vec)
        # if (dot >=0):
        if True:
            while(self.dist_to_waypoint(waypoint) > threshold):
                control_signal = self.controller.run_step(self.vel_ref,waypoint) 
                self.vehicle.apply_control(control_signal)
                if len(self.collision_hist) != 0:
                    return
                self.location_history.append(self.vehicle.get_location())
                self.update_spectator()

    def get_left_lane_waypoints(self, offset = VEHICLE_VEL):
        # TODO: Check if lane direction is the same as current direction
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        left_lane_target = current_waypoint.get_left_lane().next(offset)[0]
        left_lane_follow = left_lane_target.next(offset)[0]
        self.waypointsList = [left_lane_target, left_lane_follow]

    def get_right_lane_waypoints(self, offset = VEHICLE_VEL):
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        right_lane_target = current_waypoint.get_right_lane().next(offset)[0]
        right_lane_follow = right_lane_target.next(offset)[0]
        self.waypointsList = [right_lane_target, right_lane_follow]
    
    def get_current_lane_waypoints(self, offset = VEHICLE_VEL):
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        target_wp = current_waypoint.next(offset)[0]
        follow_wp = target_wp.next(offset)[0]
        self.waypointsList = [target_wp, follow_wp]
    
    def do_left_lane_change(self):
        self.lane_index -= 1
        self.get_left_lane_waypoints()
        self.actor_location_history += [a.get_location() for a in self.vehicle_list]
        for i in range(len(self.waypointsList)-1):
            self.current_pos = self.vehicle.get_location()
            self.go_to_waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()

    def do_right_lane_change(self):
        self.lane_index += 1
        self.get_right_lane_waypoints()
        self.actor_location_history += [a.get_location() for a in self.vehicle_list]
        for i in range(len(self.waypointsList)-1):
            self.current_pos = self.vehicle.get_location()
            self.go_to_waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()
            
    def do_follow_lane(self):
        self.get_current_lane_waypoints()
        ego_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
        for i in range(len(self.waypointsList)-1):
            # check if we need to slow down
            # set next wp at half the distance to vehicle ahead
            # go there
            # return
            self.current_pos = self.vehicle.get_location()
            self.go_to_waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()

    def update_spectator(self):
        new_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        spectator_transform =  self.vehicle.get_transform()
        spectator_transform.location += carla.Location(x = -10*math.cos(new_yaw), y= -10*math.sin(new_yaw), z = 5.0)
        
        self.spectator.set_transform(spectator_transform)
        self.world.tick()

    def is_waypoint_in_direction_of_motion(self,waypoint):
        current_pos = self.vehicle.get_location()

    def draw_waypoints(self):
        for waypoint in self.waypointsList:
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                                       color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                                       persistent_lines=True)

    def get_state_representation(self, vehicle_list = []):
        grid = np.ones((51, 3))
        ego_loc = self.vehicle.get_location()
        ego_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
        for actor in vehicle_list:
            actor_vel = actor.get_velocity()
            actor_speed = (3.6 * math.sqrt(actor_vel.x ** 2 + actor_vel.y ** 2 + actor_vel.z ** 2)) / 100.0
            actor_loc = actor.get_location()
            dist = actor_loc.distance(ego_loc)
            if dist > 60: continue
            
            cell_delta = int(dist // CELL_LEN)

            previous_reference_loc = ego_wp.previous(15)[0].transform.location
            is_behind = actor_loc.distance(previous_reference_loc) <= 15
            if is_behind: cell_delta *= -1

            # find actor's lane
            actor_lane = None
            left_lane_wp = None
            center_lane_wp = None
            right_lane_wp = None

            if self.lane_index == LEFT_LANE:
                # print("left")
                left_lane_wp = ego_wp
                center_lane_wp = ego_wp.get_right_lane()
                right_lane_wp = center_lane_wp.get_right_lane()
            elif self.lane_index == CENTER_LANE:
                # print("center")
                left_lane_wp = ego_wp.get_left_lane()
                center_lane_wp = ego_wp
                right_lane_wp = ego_wp.get_right_lane()
            elif self.lane_index == RIGHT_LANE:
                # print("right")
                left_lane_wp = ego_wp.get_left_lane().get_left_lane()
                center_lane_wp = ego_wp.get_left_lane()
                right_lane_wp = ego_wp

            if left_lane_wp is None or center_lane_wp is None or right_lane_wp is None:
                return None

            left_lane_next_wp = left_lane_wp.previous(30)[0]
            center_lane_next_wp = center_lane_wp.previous(30)[0]
            right_lane_next_wp = right_lane_wp.previous(30)[0]

            # TODO: we can speed this up by using "dist" for the range, but we need to handle next/previous search differently
            for i in range(1, 95):
                # print(left_lane_next_wp.transform.location.x, left_lane_next_wp.transform.location.y)
                # print(left_lane_next_wp.transform.location)
                # print(actor_loc.distance(left_lane_next_wp.transform.location))
                if actor_loc.distance(left_lane_next_wp.transform.location) < 1:
                    actor_lane = LEFT_LANE
                    break
                elif actor_loc.distance(center_lane_next_wp.transform.location) < 1:
                    actor_lane = CENTER_LANE
                    break
                elif actor_loc.distance(right_lane_next_wp.transform.location) < 1:
                    actor_lane = RIGHT_LANE
                    break
            
                left_lane_next_wp = left_lane_next_wp.next(1)[0]
                center_lane_next_wp = center_lane_next_wp.next(1)[0]
                right_lane_next_wp = right_lane_next_wp.next(1)[0]
            
            # if we didn't find the actor's lane, it must >30m behind the ego car
            if actor_lane == None:
                continue
            grid[(31 - cell_delta):(35 - cell_delta), actor_lane] = actor_speed
            # TODO: Fill in the grid with actors' velocities
        vel = self.vehicle.get_velocity()
        grid[31:35, self.lane_index] = (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)) / 100.0

        state = np.zeros((45, 3))
        state[:, :] = grid[3:48, :]
        return state

######## Training Loop ########
if __name__ == '__main__':
    run_start = datetime.now().strftime("%d_%m_%Y_%H_%M")
    state_height = 45
    state_width = 3
    action_size = 3
    EPISODES = 1000
    batch_size = 16
    SECONDS_PER_EPISODE = 6 * 60
    count = 0
    
    agent = DuelingDoubleDQN(state_height, state_width, action_size)
    player = Player()

    ego_speed = player.vehicle.get_velocity()
    ego_lane = player.lane_index

    pos = [ego_speed / 25, 0, 0]
    if ego_lane == LEFT_LANE:
        pos = [ego_speed / 25, 1, 0]
    elif ego_lane == CENTER_LANE:
        pos = [ego_speed / 25, 1, 1]
    elif ego_lane == RIGHT_LANE:
        pos = [ego_speed / 25, 0, 1]
    pos = np.reshape(pos, [1, 3])

    action = 0

    scores = []
    avg_scores = []
    print('Initializing training loop: ')
    for episode in tqdm(range(1, EPISODES + 1), unit='episode'):
        # print('Starting a new episode!')
        player.collision_hist = []

        score = 0
        step = 1

        current_state = player.reset()
        # print('Current state: \n', current_state)
        
        done = False
        episode_start = time.time()
        # player.do_follow_lane()
        ego_speed = player.vehicle.get_velocity()
        ego_lane = player.lane_index

        pos = [ego_speed / 25, 0, 0]
        if ego_lane == LEFT_LANE:
            pos = [ego_speed / 25, 1, 0]
        elif ego_lane == CENTER_LANE:
            pos = [ego_speed / 25, 1, 1]
        elif ego_lane == RIGHT_LANE:
            pos = [ego_speed / 25, 0, 1]
        pos = np.reshape(pos, [1, 3])
        current_pos = pos
        try:
            while True:
                current_state = np.reshape(current_state, [-1, 1, state_height, state_width])
                current_state = np.array(current_state)
                action = agent.act([current_state, current_pos])
                # print('action taken: ', action)

                new_state, new_pos, reward, done, _ = player.step(action)

                if action != 0:
                    agent.remember1(current_state, action, reward, new_state, done)
                else:
                    agent.remember2(current_state, action, reward, new_state, done)
                score += reward
                
                current_state = new_state
                # print('new state: \n', current_state)
                current_pos = new_pos
                step += 1

                if done:
                    agent.save("models/" + str(episode) + ".h5")
                    # print("weight saved")
                    print("episode: {}, epsilon: {}".format(episode, agent.epsilon))
                    with open("models/train_c.txt", "a") as f:
                        f.write(" episode {} epsilon {}\n".format(episode, agent.epsilon))
                    with open("models/trainexp1_c.pkl", "wb") as exp1:
                        pickle.dump(agent.memory1, exp1)
                    with open("models/exp2_c.pkl", "wb") as exp2:
                        pickle.dump(agent.memory2, exp2)

                    episode = episode + 1
                    if episode == 41:
                        agent.epsilon_min = 0.10
                    if episode == 71:
                        agent.epsilon_min = 0.03
                    if episode == 6:
                        agent.epsilon_decay = 0.99985  # start epsilon decay
                    break

                count += 1
                if count == 10:
                    agent.update_target_model()
                    # print('target model updated')
                    count = 0

                if len(agent.memory1) > batch_size and len(agent.memory2) > batch_size:
                    agent.replay(batch_size)
        except Exception as e:
            print(e)
            pass
            
        gw = player.world.get_map().generate_waypoints(20)
        X_W = [w.transform.location.x for w in gw]
        Y_W = [w.transform.location.y for w in gw]

        plt.figure()
        plt.scatter(X_W, Y_W, c="black", s=0.25, alpha=0.1)
        locations = player.location_history
        X = [loc.x for loc in locations]
        Y = [loc.y for loc in locations]

        actor_hist = player.actor_location_history
        actor_hist_x = [loc.x for loc in actor_hist]
        actor_hist_y = [loc.y for loc in actor_hist]

        plt.plot(X, Y, c="green", linewidth=0.5)
        plt.scatter(X[0], Y[0], c="red", s=1)
        plt.scatter(X[-1], Y[-1], c="blue", s=1)
        plt.scatter(actor_hist_x, actor_hist_y, c="purple", s=0.5, alpha=0.5)
        
        actor_locations_x = [actor.get_location().x for actor in player.vehicle_list if actor.is_alive]
        actor_locations_y = [actor.get_location().y for actor in player.vehicle_list if actor.is_alive]
        plt.scatter(actor_locations_x, actor_locations_y, c="purple", s=1)
    
        plt.savefig(r"C:\Users\Lohesh\Downloads\Notes\Research\lane-changing-carla_d3qn\eval_visualizations\\" + run_start + "_" + str(episode) + ".png", dpi=300, transparent=True)
        plt.close()

        for actor in player.actor_list:
            try:
                actor.destroy()
            except:
                pass

        scores.append(score)
        avg_scores.append(np.mean(scores[-10:]))

        # print('episode: ', episode, 'score %.2f' % score)

    print('Scores List: ', scores)
    print('Avg. Scores List: ', avg_scores)

    with open("scores.txt", "a") as f:
        f.write(" Scores List: {}\n".format(scores))

    with open("avg_scores.txt", "a") as f:
        f.write(" Scores List: {}\n".format(avg_scores))