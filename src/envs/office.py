import numpy as np

class OfficeParams:
	def __init__(self):
		# dimension parameters
		self.grid_dimension = (12, 16)
		self.m = 12
		self.n = 16
		"""
		Game objects
		its a map from tuple of coordinates in grid
		to entity labeled by char.
		"""
		self.agent = (10,3) # agent
		self.objects = {}
		self.objects[(10,2)] = "A"	# A,B,C,D are locations to patroll
		self.objects[(10,14)] = "B"
		self.objects[(2,14)] = "C"
		self.objects[(2,2)] = "D"
		self.objects[(6,10)] = "m" 	# the mail
		self.objects[(3,5)] = "c" 	# coffee
		self.objects[(9,11)] = "c"
		self.objects[(6,6)] = "o" 	# office
		self.objects[(2,6)] = "*" 	# plant
		self.objects[(2,10)] = "*"
		self.objects[(6,2)] = "*"
		self.objects[(6,14)] = "*"
		self.objects[(10,6)] = "*"
		self.objects[(10,10)] = "*"

		self.map_symbols = {"A":1.0, "B":2.0, "C":3.0, "D":4.0, "m": 5.0, "c": 6.0, "o":7.0, "*":8.0, "x":9.0, "#":10.0, 0:11.0}

class Office:
	def __init__(self, params):
		self.env_game_over = False
		self.agent = params.agent
		self.initial_agent = params.agent
		self.params = params
		self.observation = np.zeros(self.params.m * self.params.n, dtype=np.float32)
		self._load_map()

	def _load_map(self):

		# initialize map as blanck (full of 0's)
		grid = []
		for row in range(self.params.m+1):
			grid.append([])
			for _ in range(self.params.n+1):
				grid[row].append(0)
		self.grid = grid

		## draw walls (12 3x3 boxes)
		# horizontal lines
		for row in [0,4,8,12]:
			for col in range(self.params.n+1):
				self.grid[row][col] = "#"

		# vertical lines
		for col in [0,4,8,12,16]:
			for row in range(self.params.m+1):
				self.grid[row][col] = "#"

		## break walls to make doors
		# horizontal doors
		for row in [2,10]:
			for col in [4,8,12]:
				self.grid[row][col] = 0

		# vertical doors 
		# (exception since room with o and m only can be accessed from the top)
		for col in [2,6,10,14]:
			self.grid[4][col] = 0
		for col in [2,14]:
			self.grid[8][col] = 0

		# load objects
		for coord_tuple, symbol in self.params.objects.items():
			i, j = coord_tuple
			self.grid[i][j] = symbol
		
		# add agent and its possible actions
		(x_agent, y_agent) = self.agent
		self.grid[x_agent][y_agent] = "x"
		self.actions = [0,1,2,3]
		self.__observation()

	def __observation(self):
		"""
		function transforms 2d list grid into 1d arra
		and symbols #, *, o,... are mapped into floats, to allow
		the dqn to learn
		"""
		k = 0
		for i in range(self.params.m):
			for j in range(self.params.n):
				self.observation[k] = self.params.map_symbols[self.grid[i][j]]
				k += 1

	def reset(self):
		"""
		reset environment
		"""
		self.agent = self.initial_agent
		self._load_map()
		return self.observation

	def step(self, action):
		x_agent, y_agent = self.agent

		if action == 0:
			y_agent += 1
		if action == 1:
			y_agent -= 1
		if action == 2:
			x_agent -= 1
		if action == 3:
			x_agent += 1
		
		# only valid action if not in wall
		if self.grid[x_agent][y_agent] != "#": 
			self.agent = (x_agent, y_agent)
			self.params.agent = (x_agent, y_agent)

			# reload map (update)
			self._load_map()
		return self.observation, np.random.random()

	def get_true_propositions(self):
		# retuns symbol of true propositions
		true_props = ""
		if self.agent in self.params.objects:
			true_props += self.params.objects[self.agent]
		return true_props
			

	def game_display(self):
		"""
		Displays the grid on the terminal
		Parameters
		---------------
		office: Office object
		"""
		for row in range(self.params.m+1):
			for col in range(self.params.n+1):

				# zeros in the grid represent empty space
				if self.grid[row][col] == 0:
					print(" ", end="")
				else:
					print(self.grid[row][col], end="")
			print()


"""
For debugging purposes. Shows the environment map.
"""
if __name__ == "__main__":

	import time
	t_start = time.time()
	print("---- Office Environment ----")
	params = OfficeParams() # office params
	office = Office(params) # office environment

	office.game_display()
	time.sleep(2)
	tour_sequence = [0,0,0,0,1,2,3]
	for action in tour_sequence:
		office.step(action)
		office.game_display()
		time.sleep(1)

	office.reset()
	office.game_display()

	print(f"---- finished {round(time.time()-t_start,5)} s ----")
	exit(0)