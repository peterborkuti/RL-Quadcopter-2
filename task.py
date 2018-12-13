import numpy as np
from physics_sim import PhysicsSim
np.seterr(all='raise')

def reward_function(pose, target_pos, angular_velo):
    pose_reward = 1.-.003*(np.abs(pose[:3] - target_pos)).sum()
    ang_velo = np.abs(angular_velo[:3]).sum()
    
    return pose_reward - ang_velo

def reward_function_v8(distance, time, done=False, max_distance=10.0, min_distance=5.0, max_time=5.0):
    goal_reward = - (distance/(max_distance/2))**0.4 + 2.5

    time_reward = time ** 0.2
    premature_end_penalty = 0
    
    reward = goal_reward * time_reward + premature_end_penalty - 1.5

    return reward

def reward_function_v7(distance, time, done=False, max_distance=10.0, min_distance=5.0, max_time=5.0):
    # extra points when distance is small
    """
    if (distance < min_distance):
        goal_reward = 1.1
    else:
        # goal reward is positive, between 0 and 50 and has a steep gradient
        # near 0 distance
        # the smaller the distance the nearer to 1
        # from 50, this is negative
    """
    goal_reward = - distance**0.4 + 2.5

    # time reward is between 0 and 1
    #time_reward = 1.0 - (max(time, 0.1)/(max_time-1))**(-0.4)
    time_reward = time ** 0.2
    premature_end_penalty = 0

    #if done:
    #    if time < max_time:
    #        premature_end_penalty = -0.1
        #else:
        #    premature_end_penalty = 200 - goal_distance
    
    reward = goal_reward * time_reward + premature_end_penalty - 1.5

    return reward

def reward_function_v6(distance, time, done=False, max_distance=10.0, min_distance=5.0, max_time=5.0):
    # extra points when distance is small
    """
    if (distance < min_distance):
        goal_reward = 1.1
    else:
        # goal reward is positive, between 0 and 50 and has a steep gradient
        # near 0 distance
        # the smaller the distance the nearer to 1
        # from 50, this is negative
    """
    goal_reward = 1.0 - (max(0.1, distance)/min_distance)**0.4

    # time reward is between 0 and 1
    #time_reward = 1.0 - (max(time, 0.1)/(max_time-1))**(-0.4)
    time_reward = 1 + (max(time, 0.1)/(max_time + 1))

    premature_end_penalty = 0
    """
    if done:
        if time < max_time:
            premature_end_penalty = -0.1
        #else:
        #    premature_end_penalty = 200 - goal_distance
    """
    reward = goal_reward * time_reward + premature_end_penalty + 0.5

    return reward


def reward_function_v5(distance, time, done=False, max_distance=10.0, min_distance=5.0, max_time=5.0):
    # extra points when distance is small
    """
    if (distance < min_distance):
        goal_reward = 1.1
    else:
        # goal reward is positive, between 0 and 50 and has a steep gradient
        # near 0 distance
        # the smaller the distance the nearer to 1
        # from 50, this is negative
    """
    goal_reward = 1.0 - (max(0.1, distance)/max_distance)**0.4

    # time reward is between 0 and 1
    #time_reward = 1.0 - (max(time, 0.1)/(max_time-1))**(-0.4)
    time_reward = (max(time, 0.1)/(max_time + 1))

    premature_end_penalty = 0
    """
    if done:
        if time < max_time:
            premature_end_penalty = -0.1
        #else:
        #    premature_end_penalty = 200 - goal_distance
    """
    reward = goal_reward * time_reward + premature_end_penalty + 0.5

    return reward


def reward_function_v4(distance, time, done=False, max_distance=10.0, min_distance=5.0, max_time=5.0):
    # extra points when distance is small
    """
    if (distance < min_distance):
        goal_reward = 1.1
    else:
        # goal reward is positive, between 0 and 50 and has a steep gradient
        # near 0 distance
        # the smaller the distance the nearer to 1
        # from 50, this is negative
    """
    goal_reward = 1.0 - (max(0.1, distance)/max_distance)**0.2

    # time reward is between 0 and 1
    time_reward = 1.0 - (max(time, 0.1)/(max_time-1))**(-0.2)

    premature_end_penalty = 0
    """
    if done:
        if time < max_time:
            premature_end_penalty = -0.1
        #else:
        #    premature_end_penalty = 200 - goal_distance
    """
    reward = goal_reward + time_reward + premature_end_penalty +0.3

    return reward

def reward_function_v3(distance, time, done=False, max_distance=200.0, min_distance=5.0, max_time=5.0):
    # max(0,1, distance): when distance < 0.0004, floating point error
    goal_reward = 1.0 - ((max(0.1, distance))/(max_distance))**0.2
    
    reward = goal_reward
    
    return reward


def reward_function_v2(distance, time, done=False, max_distance=200.0, min_distance=5.0, max_time=5.0):
    # max(0,1, distance): when distance < 0.0004, floating point error
    goal_reward = 1.0 - ((max(0.1, distance)-0.1)/(max_distance/3.0))**0.2
    
    # time reward is between 0 and 1
    #time_reward = (time/max_time)**0.2
    time_reward = 1-(max(time, 0.1)/(max_time-1))**(-0.2)

    premature_end_penalty = 0

    reward = goal_reward + time_reward + premature_end_penalty

    return reward
    
def reward_function_v1(distance, time, done=False, max_distance=200.0, min_distance=5.0, max_time=5.0):
    # extra points when distance is small
    if (distance < min_distance):
        goal_reward = 10
    else:
        # goal reward is positive, between 0 and 50 and has a steep gradient
        # near 0 distance
        # the smaller the distance the nearer to 1
        # from 50, this is negative
        goal_reward = 1 - (distance/(max_distance/4))**0.2

    # time reward is between 0 and 1
    time_reward = (time/max_time)**0.2

    premature_end_penalty = 0
    if done:
        if time < max_time:
            premature_end_penalty = -1000
        else:
            premature_end_penalty = 200 - goal_distance

    reward = goal_reward + time_reward + premature_end_penalty - 1

    return reward

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 7 #4 # 6 ... x,y,z, time
        self.action_low = 50
        self.action_high = 400
        self.action_size = 4
        self.runtime = runtime
        self.agent_distance = -np.inf
        self.raw_reward = 0.0
        self.counter = 0
        self.avg_dist = 0
        self.sum_dist = 0

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def adjust_range(self, angles):
        return ((angles + np.pi) % (2 * np.pi))-np.pi
    
    def set_self_distance(self):
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

        # euclidan-distance in 3D
        self.agent_distance = np.linalg.norm(self.sim.pose[:3]-self.target_pos)
        
    
    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        #reward = np.tanh(3 + 0.1 * self.sim.v[2] - 0.005 * 
        #    np.square(self.adjust_range(self.sim.pose[3:5])).sum() - 0.001 *
        #    np.square(self.sim.angular_v).sum())
        #original reward: reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #dz = abs(self.sim.pose[2] - self.target_pos[2]) ** 2
        #dx = abs(self.sim.pose[0] - self.target_pos[0]) ** 2
        #dy = abs(self.sim.pose[1] - self.target_pos[1]) ** 2
        #time = (self.runtime - self.sim.time) ** 2 # BAD: its called in every step, not only at the end
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

        reward = reward_function(self.sim.pose, self.target_pos, self.sim.angular_v)
        
        self.raw_reward = reward

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            self.set_self_distance()
            done = done or self.agent_distance > 10
            reward += self.get_reward(done)
            #pose_all.append([*self.sim.pose[0:3], self.sim.time]) # only x,y,z,time
            pose_all.append([*self.sim.pose, self.sim.time])
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = [*self.sim.pose[0:3], self.sim.time] * self.action_repeat # only x,y,z, time
        state = [*self.sim.pose, self.sim.time] * self.action_repeat
        
        return state