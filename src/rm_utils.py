"""
Reward Machines Util functions
"""
import json

def evaluate_dnf(formula, true_props):
	"""
	Evaluate the disjunctive normal form.
	Evaluates the formula assuming true_props are true propositions
	and the rest are false.

	example: evaluate_dnf("A&B|!C&D", "D") would return True
			as D (True) and A,B,C (False) the left condition "not C and D"
			is true (True and True -> True) and so the evaluation is True
	
	Note: Formula and true props have to use the symbol criteria
	symbols in Office environment: ["A", "B", "C", "D", "*", "c", "m", "o"]

	Parameters:
	--------------
	formula: string
		formula in string formula, example: "a&b|!c&d", "True", ...
	true_props: char
		char symbol of true proposition

	Returns:
	-------------
	boolean
	"""
	#print(formula, true_props)
    # split AND conditions 
	if "&" in formula:
		# evaluate subformulas
		for subformula in formula.split("&"):
			# if any subcondition is False then global is False (AND condition)
			if not evaluate_dnf(subformula, true_props):
				return False
		# if didnt trigger False then all subconditions are met
		return True

	# split OR conditions
	if "|" in formula:
		# evaluate subformulas
		for subformula in formula.split("|"):
			# if any subcondition is True then global is True (OR condition)
			if evaluate_dnf(subformula, true_props):
				return True 
		# if didnt trigger True then none subconditions are met
		return False

	# check for NOT
	if formula.startswith("!"):
		# invert proposition
		return not evaluate_dnf(formula[1:], true_props)

	# simple cases
	if (formula == "True"):
		return True
	if (formula == "False"):
		return False

	# atomized propositions
	if formula in true_props:
		return True
	if formula not in true_props:
		return False

def parse_json_reward_machine(json_file, env_idx):
	"""
	extract reward machine from json_file

	Parameters:
	---------------
	json_file: string
		path to json file for the environment

	env_idx: int
		index of the task to extract

	Returns:
	--------------
	env_data: dictionary
		dictionary with parsed content
		- env_data["env_id"]: index of task
		- env_data["env_name"]: name of class of environment
		- env_data["initial_state"]: u0, initial state of RM
		- env_data["n_rm_states"] total number of reward machines states used for DQRM/DQRM learning
        - env_data["rm_states"]: list of <u1, u2, delta_u, delta_r> reward machine states
								with defined delta_u[u1][u2] state transition function and 
								delta_r[u1][u2] reward transition function
	"""
	with open(json_file) as file:
		data = json.load(file)
	env_data = data["envs"][int(env_idx)-1]
	return env_data

# for debugging purposes
if __name__ == "__main__":
	import time

	start_t = time.time()
	print("---- Debugging -----")
	idx = input("function to debug: \n [0]: evaluate_dnf() \n [1]: parse_json_reward_machine() \n")

	if (idx == "0"):
		print("evaluate_dnf(formula, true_props)")
		formula = input("formula: ")
		true_props = input("true_props: ")
		print(f"Result: {evaluate_dnf(formula, true_props)}")

	if (idx == "1"):
		print("parse_json_reward_machine(json_file, task_idx)")
		json_file = input("json_file: ")
		task_idx = input("task_idx: ")
		print(f"Result: {parse_json_reward_machine(json_file, task_idx)}")

	print(f"---- Finished {time.time()-start_t} s ----")
