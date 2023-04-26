import pygame as pg
import numpy as np
import random
from math import sqrt, pi, cos, sin
from PIL import Image, ImageOps
from wfc import wfc_control as wfc

# WFC Variables
input_size = 16  # Size of the input to the WFC algorithm
background_size = 64  # Size of the output from the WFC algorithm
wfc_image = './images/samples/environment.png'  # Path to the image to use as input to the WFC algorithm

# Pygame Variables
project_name = 'CAP4630 Introduction to Artificial Intelligence - Multi-robot Exploration Under the Constraints of Wireless Networking' # Project name
background_scale = 10  # Scaling factor for the background image
background_border = 1  # Width of the border around the background image
background_color = (255, 255, 255)  # Color of the background
background_resolution = (background_size + background_border) * background_scale  # Resolution of the background image
t = 0  # Current time step

# Robot Variables
robot_number = 5
robot_radius = 3  # Radius of the robots
step_size = 1
movements = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]  # Possible directions for the robots to move
wireless_distance = 50  # The maximum distance a robot can be from either the base station or another robot in the chain
robot_color = (255, 0, 0)

# Base Station Variables
start_radius = robot_number * robot_radius  # Radius of the area around the base station that robots should start on
clear_size = start_radius * 3  # Size of the area around the base station that should be free of obstacles
base_station = [clear_size//2, background_resolution - clear_size//2]  # Set the base station's position
base_station_color = (0, 0, 255)

# Environment Variables
obstacle_color = (0, 0, 0)
visited_color = (0, 255, 0)

# Genetic Algorithm Variables
population_size = 10 # Number of potential solutions to generate
generations = 100 # Number of generations to run the genetic algorithm for
mutation_rate = 0.1 # The rate at which mutations occur

class Project:
    def __init__(self):
        self.collision_array = np.zeros((background_resolution, background_resolution), dtype=bool)
        self.collision_surface = None
        self.open_space = 0
        self.screen = pg.display.set_mode([background_resolution, background_resolution])
        self.visited = set()
        self.robot_number = robot_number
        self.configuration = Configuration()
        pg.init()
        pg.display.set_caption(project_name)
        self.screen.fill(background_color)
        self.load_wfc()
        self.count_open_space()
        self.run()
    def load_wfc(self):
        with Image.open(wfc_image) as image_file:
            data = image_file.convert('RGB')
            data = np.array(data)
            data = data[:, :, :3]
            wfc_output = wfc.execute_wfc(image=data)
            image = Image.fromarray(wfc_output)
            image = ImageOps.expand(image, border=background_border, fill=background_color)
            image = image.resize((background_resolution, background_resolution))
            pixels = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3).transpose((1, 0, 2))
            gray_threshold = 128
            gray_mask = ((pixels >= gray_threshold).all(axis=-1))
            self.collision_array = ~gray_mask
            self.collision_array[:clear_size, background_resolution-clear_size:] = False
            self.collision_surface = pg.surfarray.make_surface(self.collision_array.astype('uint8'))
            self.collision_surface.set_palette([background_color, obstacle_color])
            pg.display.update()
    def count_open_space(self):
        for i in range(background_resolution):
            for j in range(background_resolution):
                if self.collision_array[i][j] == False:
                    self.open_space += 1
    def display_wfc(self):
        self.screen.blit(self.collision_surface, (0, 0))
        pg.display.update()
    def mark_visited(self, position):
        self.visited.add((int(position[0]), int(position[1])))
        pg.draw.rect(self.screen, visited_color, pg.Rect(int(position[0]), int(position[1]), 1, 1))
    def update(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
        self.display_wfc()
        #self.move_robots()
        self.draw_robots()
        pg.display.update()
    def run(self):
        while True:
            genetic_algorithm = GeneticAlgorithm(self, population_size, generations, mutation_rate)
            best_configuration = genetic_algorithm.run()
            self.configuration = best_configuration
            self.update()
    def draw_robots(self):
        for robot in self.configuration.population:
            robot.draw(self.screen)

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
    def find_robot_tree(self, robot):
        if self.root == robot:
            return self
        for child in self.children:
            found = child.find_robot_tree(robot)
            if found:
                return found
        return None

    
class Configuration:
    def __init__(self):
        self.population = self.initial_population()
        self.wireless_tree = WirelessTree(Robot(base_station[0], base_station[1]))
    def initial_population(self):
        population = []
        for _ in range(robot_number):
            angle = pi / 2
            for i in range(robot_number):
                x = base_station[0] + start_radius * cos(angle)
                y = base_station[1] + start_radius * sin(angle)
                angle += 2 * pi / robot_number
                population.append(Robot(x, y))
        return population
    def check_wireless_constraint(self):
        for robot in self.population:
            if not self.wireless_tree.is_connected(robot):
                return False
        return True

    def fitness(self, project):
        score = 0
        if not self.check_wireless_constraint():
            return score
        for robot in self.population:
            x, y = int(robot.x), int(robot.y)
            if 0 <= x < background_resolution and 0 <= y < background_resolution:
                if not project.collision_array[x][y]:
                    score += 1
        return score
    
    def update_wireless_tree(self, robot):
        robot_tree = self.wireless_tree.find_robot_tree(robot)
        if robot_tree:
            robot_tree.parent.children.remove(robot_tree)
            for child in robot_tree.children:
                robot_tree.parent.add_child(child)

        for r in self.population:
            if robot.distance_to(r) <= wireless_distance:
                r_tree = self.wireless_tree.find_robot_tree(r)
                if r_tree:
                    r_tree.add_child(WirelessTree(robot, r_tree))
                    break


    
class GeneticAlgorithm:
    def __init__(self, project, population_size, generations=100, mutation_rate=0.1):
        self.project = project
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
    def run(self):
        population = [Configuration() for _ in range(self.population_size)]
        
        for _ in range(self.generations):
            # Evaluate the fitness of the population
            fitness_scores = [Configuration.fitness(configuration, self.project) for configuration in population]
            
            # Select parents based on their fitness
            parents = self.selection(population, fitness_scores)
            
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                offspring = self.crossover(parent1, parent2)
                self.mutation(offspring)
                next_generation.append(offspring)
            
            population = next_generation
        
        # Return the best configuration found
        best_index = np.argmax(fitness_scores)
        return population[best_index]
    
    def selection(self, population, fitness_scores):
        # Tournament selection
        parents = []
        for _ in range(self.population_size):
            candidates = random.sample(population, k=3)
            scores = [fitness_scores[population.index(c)] for c in candidates]
            best_index = np.argmax(scores)
            parents.append(candidates[best_index])
        return parents
    
    def crossover(self, parent1, parent2):
        # Uniform crossover
        offspring = []
        for r1, r2 in zip(parent1.population, parent2.population):
            if random.random() < 0.5:
                offspring.append(Robot(r1.x, r1.y))
            else:
                offspring.append(Robot(r2.x, r2.y))
        child = Configuration()
        child.population = offspring
        return child

    
    def mutation(self, configuration):
        for robot in configuration.population:
            if random.random() < self.mutation_rate:
                dx, dy = random.choice(movements)
                new_x, new_y = robot.x + dx * step_size, robot.y + dy * step_size
                if 0 <= new_x < background_resolution and 0 <= new_y < background_resolution:
                    robot.move(dx * step_size, dy * step_size)
                    configuration.update_wireless_tree(robot)  # Update the wireless tree


Project()