import pygame as pg
import numpy as np
import random, copy, threading, time, sys, queue
from math import sqrt, pi, cos, sin
from PIL import Image, ImageOps
from wfc import wfc_control as wfc
from pygame.time import Clock

# WFC Variables
input_size = 16
background_size = 64
wfc_image = './images/samples/environment.png'

# Pygame Variables
project_name = 'CAP4630 Introduction to Artificial Intelligence - Multi-robot Exploration Under the Constraints of Wireless Networking'
background_scale = 2
background_border = 1
background_color = (255, 255, 255)
background_resolution = (background_size + background_border) * background_scale
tries_number = 5
t = 0

# Robot Variables
constraint = True
robot_number = 2
robot_radius = 1
vision_multiplier = 3
vision_radius = robot_radius * vision_multiplier
step_size = 1
movements = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
wireless_distance = background_resolution*1.5/2
robot_color = (255, 0, 0)
vision_color = (173, 216, 230)

# Base Station Variables
start_radius = robot_number * robot_radius
clear_size = start_radius * 3
base_station = [clear_size//2, background_resolution - clear_size//2]
base_station_color = (0, 255, 0)

# Environment Variables
obstacle_color = (0, 0, 0)
visited_color = (255, 255, 0)

# Genetic Algorithm Variables
population_size = 10
generations = 1
mutation_rate = 0.1
wall_penalty_factor = 5
exploration_factor = 0.0001

class Project:
    def __init__(self):
        self.running = True
        self.fps = 30
        self.collision_array = np.zeros((background_resolution, background_resolution), dtype=bool)
        self.collision_surface = None
        self.visited_array = np.zeros((background_resolution, background_resolution), dtype=bool)
        self.visited_surface = pg.Surface((background_resolution, background_resolution))
        self.visited_surface.fill(background_color)
        self.open_space = 0
        self.screen = pg.display.set_mode([background_resolution, background_resolution])
        self.base_station = BaseStation(base_station[0], base_station[1])
        self.robot_number = robot_number
        self.configuration = Configuration(self.initial_population())
        pg.init()
        pg.display.set_caption(project_name)
        self.screen.fill(background_color)
        self.load_wfc()
        self.count_open_space()
        self.run()

    def run_genetic_algorithm(self):
        while self.running:
            genetic_algorithm = GeneticAlgorithm(self, population_size, generations, mutation_rate)
            best_configuration = genetic_algorithm.run()
            self.configuration = best_configuration

    def load_wfc(self):
        # Create a greyscale white image
        self.collision_array = np.ones([background_resolution, background_resolution], dtype=np.float32)
        self.collision_array[:] = 0.0

        # Create a surface from the greyscale white image
        self.collision_surface = pg.surfarray.make_surface(self.collision_array.astype('uint8') * 255)
        self.collision_surface.set_palette([background_color, obstacle_color])
        pg.display.update()
        #with Image.open(wfc_image) as image_file:
        #    data = image_file.convert('RGB')
        #    data = np.array(data)
        #    data = data[:, :, :3]
        #    wfc_output = wfc.execute_wfc(image=data)
        #    image = Image.fromarray(wfc_output)
        #    image = ImageOps.expand(image, border=background_border, fill=background_color)
        #    image = image.resize((background_resolution, background_resolution))
        #    pixels = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3).transpose((1, 0, 2))
        #    gray_threshold = 128
        #    gray_mask = ((pixels >= gray_threshold).all(axis=-1))
        #    self.collision_array = ~gray_mask
        #    self.collision_array[:clear_size, background_resolution-clear_size:] = False
        #    self.collision_surface = pg.surfarray.make_surface(self.collision_array.astype('uint8'))
        #    self.collision_surface.set_palette([background_color, obstacle_color])
        #    pg.display.update()

    def count_open_space(self):
        background_resolution = self.collision_array.shape[0]
        open_space = np.zeros((background_resolution, background_resolution), dtype=bool)

        q = queue.Queue()

        for i in range(background_resolution):
            for j in [0, background_resolution - 1]:
                q.put((i, j))
        for j in range(1, background_resolution - 1):
            for i in [0, background_resolution - 1]:
                q.put((i, j))

        while not q.empty():
            x, y = q.get()
            if self.collision_array[x, y] == False and open_space[x, y] == False:
                open_space[x, y] = True

                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < background_resolution and 0 <= ny < background_resolution:
                        q.put((nx, ny))

        open_space_surface = pg.surfarray.make_surface(open_space.astype('uint8'))
        open_space_surface.set_palette([obstacle_color, background_color])
        self.collision_surface.blit(open_space_surface, (0, 0))

    def display_wfc(self):
        self.screen.blit(self.collision_surface, (0, 0))
        pg.display.update()

    def mark_visited(self, position):
        mask = pg.Surface((vision_radius * 2, vision_radius * 2), pg.SRCALPHA)
        pg.draw.circle(mask, (255, 255, 255, 255), (vision_radius, vision_radius), vision_radius)

        for i in range(-vision_radius, vision_radius):
            for j in range(-vision_radius, vision_radius):
                if mask.get_at((i + vision_radius, j + vision_radius))[3] == 255:
                    x, y = int(position[0]) + i, int(position[1]) + j
                    if 0 <= x < background_resolution and 0 <= y < background_resolution:
                        if self.collision_array[x][y] == False:
                            self.visited_array[x][y] = True
                            self.visited_surface.set_at((x, y), visited_color)

    def visited(self):
        while self.running:
            self.create_visited()
            time.sleep(0.1)

    def create_visited(self):
        for robot in self.configuration.population:
                self.mark_visited((robot.x, robot.y))

    def update(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                pg.quit()
                quit()
        self.display_wfc()
        self.draw_visited()
        self.draw_base_station()
        self.draw_robots()
        pg.display.update()

    def run(self):
        clock = Clock()

        ga_thread = threading.Thread(target=self.run_genetic_algorithm)
        ga_thread.start()
        visited_thread = threading.Thread(target=self.visited)
        visited_thread.start()

        while True:
            self.update()
            if not self.running:
                break
            clock.tick(self.fps)
        sys.exit()

    def draw_robots(self):
        for robot in self.configuration.population:
            robot.draw(self.screen)

    def draw_base_station(self):
        self.base_station.draw(self.screen)

    def draw_visited(self):
        self.visited_surface.set_colorkey(background_color)
        self.screen.blit(self.visited_surface, (0, 0))
        pg.display.update()

    def initial_population(self):
        population = []
        for _ in range(robot_number):
            angle = pi / 2
            x = base_station[0] + start_radius * cos(angle)
            y = base_station[1] + start_radius * sin(angle)
            angle += 2 * pi / robot_number
            population.append(Robot(x, y))
        for robot in population:
            self.mark_visited((robot.x, robot.y))
        return population

class BaseStation:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def draw(self, screen):
        pg.draw.circle(screen, base_station_color, (self.x, self.y), robot_radius)

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def draw(self, screen):
        pg.draw.circle(screen, robot_color, (self.x, self.y), robot_radius)
        
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
    
    def distance_to(self, other_object):
        dx = self.x - other_object.x
        dy = self.y - other_object.y
        return sqrt(dx**2 + dy**2)
    
    def check_collision(self, robot2):
        # Check if the robots are at the same position.
        if self.x == robot2.x and self.y == robot2.y:
            return True
        return False
    
class WirelessTree:
    def __init__(self, root, parent=None):
        self.root = root
        self.parent = parent
        self.children = []

    def add_child(self, child_tree):
        self.children.append(child_tree)

    def is_connected(self, robot):
        if self.root.distance_to(robot) <= wireless_distance:
            return True
        for child in self.children:
            if child.is_connected(robot):
                return True
        return False

    def is_connected_to_base(self):
        if self.parent is None:
            return True
        if self.root.distance_to(self.parent.root) <= wireless_distance:
            return self.parent.is_connected_to_base()
        return False

    def find_robot_tree(self, robot):
        if self.root == robot:
            return self
        for child in self.children:
            found = child.find_robot_tree(robot)
            if found:
                return found
        return None

class Configuration:
    def __init__(self, population):
        self.population = population
        self.wireless_tree = WirelessTree(Robot(base_station[0], base_station[1]))

    def check_wireless_constraint(self):
        for robot in self.population:
            robot_tree = self.wireless_tree.find_robot_tree(robot)
            if not robot_tree or not robot_tree.is_connected_to_base():
                return False
        return True

    def fitness(self, project):
        score = 0
        background_resolution = project.collision_array.shape[0]
        epsilon = 1e-6  # Add a small constant to avoid division by zero

        for robot in self.population:
            x, y = int(robot.x), int(robot.y)

            min_distance_up = background_resolution
            min_distance_down = background_resolution
            min_distance_left = background_resolution
            min_distance_right = background_resolution

            penalty_up = 0
            penalty_down = 0
            penalty_left = 0
            penalty_right = 0

            unvisited_up = False
            unvisited_down = False
            unvisited_left = False
            unvisited_right = False

            for i in range(y, -1, -1):
                if not project.collision_array[x][i] and not project.visited_array[x][i]:
                    min_distance_up = abs(i - y)
                    unvisited_up = True
                elif (project.collision_array[x][i] or i == 0) and not unvisited_up:
                    penalty_up = wall_penalty_factor * (1 / (abs(i - y) + epsilon))
                    break

            for i in range(y, background_resolution):
                if not project.collision_array[x][i] and not project.visited_array[x][i]:
                    min_distance_down = abs(i - y)
                    unvisited_down = True
                elif (project.collision_array[x][i] or i == background_resolution - 1) and not unvisited_down:
                    penalty_down = wall_penalty_factor * (1 / (abs(i - y) + epsilon))
                    break

            for i in range(x, -1, -1):
                if not project.collision_array[i][y] and not project.visited_array[i][y]:
                    min_distance_left = abs(i - x)
                    unvisited_left = True
                elif (project.collision_array[i][y] or i == 0) and not unvisited_left:
                    penalty_left = wall_penalty_factor * (1 / (abs(i - x) + epsilon))
                    break

            for i in range(x, background_resolution):
                if not project.collision_array[i][y] and not project.visited_array[i][y]:
                    min_distance_right = abs(i - x)
                    unvisited_right = True
                elif (project.collision_array[i][y] or i == background_resolution - 1) and not unvisited_right:
                    penalty_right = wall_penalty_factor * (1 / (abs(i - x) + epsilon))
                    break

            score += 1 / (min_distance_up + min_distance_down + min_distance_left + min_distance_right + penalty_up + penalty_down + penalty_left + penalty_right)

        return score

    def isValid(self, project):
        for robot in self.population:
            if robot.x < 0 or robot.x >= background_resolution or \
               robot.y < 0 or robot.y >= background_resolution or \
               project.collision_array[int(robot.x)][int(robot.y)] or \
               (project.base_station.x == robot.x and project.base_station.y == robot.y):
                return False

            if not self.is_robot_connected(robot, project):
                return False

        return True

    def is_robot_connected(self, robot, project):
        if robot.distance_to(project.base_station) <= wireless_distance:
            return True

        for other_robot in self.population:
            if robot != other_robot and robot.distance_to(other_robot) <= wireless_distance:
                if other_robot.distance_to(project.base_station) <= wireless_distance:
                    return True
                for third_robot in self.population:
                    if other_robot != third_robot and other_robot.distance_to(third_robot) <= wireless_distance:
                        if third_robot.distance_to(project.base_station) <= wireless_distance:
                            return True
        return False

    def randomize_population(self, project):
        for robot in self.population:
            #tries = 0
            old_x, old_y = robot.x, robot.y
            #while old_x == robot.x and old_y == robot.y and tries < tries_number:
            while old_x == robot.x and old_y == robot.y:
                direction = random.choice(movements)
                robot.move(direction[0], direction[1])
                if not self.isValid(project):
                    robot.x, robot.y = old_x, old_y
                elif constraint == True:
                    self.update_wireless_tree(project, robot)
                    break
                #tries += 1

    def update_wireless_tree(self, project, robot):
        # Remove the robot from its current position in the wireless tree
        robot_tree = self.wireless_tree.find_robot_tree(robot)
        if robot_tree:
            if robot_tree in robot_tree.parent.children:
                robot_tree.parent.children.remove(robot_tree)
            for child in robot_tree.children:
                robot_tree.parent.add_child(child)

        # Find the closest robot or base station within wireless_distance
        closest_connection = project.base_station
        min_distance = robot.distance_to(project.base_station)
        for r in self.population:
            if robot != r and robot.distance_to(r) <= wireless_distance:
                distance = robot.distance_to(r)
                if distance < min_distance:
                    min_distance = distance
                    closest_connection = r

        # Connect the robot to the closest connection
        if isinstance(closest_connection, Robot):
            r_tree = self.wireless_tree.find_robot_tree(closest_connection)
            if r_tree:
                r_tree.add_child(WirelessTree(robot, r_tree))
        else:
            self.wireless_tree.add_child(WirelessTree(robot, self.wireless_tree))

class GeneticAlgorithm:
    def __init__(self, project, population_size = 10, generations=100, mutation_rate=0.1):
        self.project = project
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
    def run(self):
        population = [copy.deepcopy(self.project.configuration) for _ in range(self.population_size)]
        
        best_configuration = None
        best_fitness = -1

        for configuration in population:
            configuration.randomize_population(self.project)
        
        fitness_scores = [Configuration.fitness(configuration, self.project) for configuration in population]

        for idx, score in enumerate(fitness_scores):
            if score > best_fitness:
                best_fitness = score
                best_configuration = population[idx]
        
        time.sleep(0.1)
        return best_configuration

Project()