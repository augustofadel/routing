"""Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
"""
from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

import pandas as pd
import numpy as np
import os
import sys
import json

data_dir = sys.argv[1]
dat = pd.read_csv(os.path.join(data_dir, 'geocode.csv'))

###########################
# Problem Data Definition #
###########################
def create_data_model():
  """Stores the data for the problem"""
  data = {}
  data["tot_days"] = 22
  data["tot_work_hours"] = 12
  data["dist_matrix"] = np.loadtxt(os.path.join(data_dir, 'dist_matrix.txt'), dtype=float) * 1000
  data["locations"] = dat[['lat', 'lon']].values.tolist()
  data["num_locations"] = len(data["locations"])
  data["num_vehicles"] = 5
  depot_id = 0
  data["node_id"] = dat['id'].values.tolist()
  data["depot"] = data["node_id"].index(depot_id)
  data["demands"] = [1] * data["num_locations"]
  # data["vehicle_capacities"] = [2 * data["tot_days"]] * data["num_vehicles"]
  data["vehicle_capacities"] = [42] * data["num_vehicles"]
  data["time_windows"] = [(0, data["tot_work_hours"] * 60 * data["tot_days"])] * data["num_locations"]
  data["time_per_demand_unit"] = 4 * 60
  data["vehicle_speed"] = 40 * 1000 / 60
  # data["travel_time_matrix"] = np.loadtxt(os.path.join(data_dir, 'time_matrix.txt'), dtype=float)  # min
  data["travel_time_matrix"] = data["dist_matrix"] / data["vehicle_speed"]
  return data

#######################
# Problem Constraints #
#######################
def create_distance_callback(data):
  """Creates callback to return distance between points."""
  _distances = {}

  for from_node in range(data["num_locations"]):
    _distances[from_node] = {}
    for to_node in range(data["num_locations"]):
      _distances[from_node][to_node] = data["dist_matrix"][from_node, to_node]

  def distance_callback(from_node, to_node):
    """Returns the distance between the two nodes"""
    return _distances[from_node][to_node]

  return distance_callback

def create_demand_callback(data):
  """Creates callback to get demands at each location."""
  def demand_callback(from_node, to_node):
    return data["demands"][from_node]
  return demand_callback


def add_capacity_constraints(routing, data, demand_evaluator):
  """Adds capacity constraint"""
  capacity = "Capacity"
  routing.AddDimensionWithVehicleCapacity(
      demand_evaluator,
      0, # null capacity slack
      data["vehicle_capacities"], # vehicle maximum capacities
      True, # start cumul to zero
      capacity)

def create_time_callback(data):
  """Creates callback to get total times between locations."""
  def service_time(node):
    """Gets the service time for the specified location."""
    return data["demands"][node] * data["time_per_demand_unit"]

  def travel_time(from_node, to_node):
    """Gets the travel times between two locations."""
    travel_time = data["travel_time_matrix"][from_node, to_node]
    return travel_time

  def time_callback(from_node, to_node):
    """Returns the total time between the two nodes"""
    serv_time = service_time(from_node)
    trav_time = travel_time(from_node, to_node)
    return serv_time + trav_time

  return time_callback
def add_time_window_constraints(routing, data, time_callback):
  """Add Global Span constraint"""
  time = "Time"
  horizon = data["tot_work_hours"] * 60 * data["tot_days"]
  routing.AddDimension(
    time_callback,
    horizon, # allow waiting time
    horizon, # maximum time per vehicle
    False, # Don't force start cumul to zero. This doesn't have any effect in this example,
           # since the depot has a start window of (0, 0).
    time)
  time_dimension = routing.GetDimensionOrDie(time)
  for location_node, location_time_window in enumerate(data["time_windows"]):
        index = routing.NodeToIndex(location_node)
        time_dimension.CumulVar(index).SetRange(location_time_window[0], location_time_window[1])

###########
# Printer #
###########
def print_solution(data, routing, assignment):
  """Prints assignment on console"""
  # Inspect solution.
  capacity_dimension = routing.GetDimensionOrDie('Capacity')
  time_dimension = routing.GetDimensionOrDie('Time')
  total_dist = 0
  time_matrix = 0
  final_route = {}
  routes_distance = []

  for vehicle_id in range(data["num_vehicles"]):
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
    route_dist = 0
    vehicle_route = []
    while not routing.IsEnd(index):
      node_index = routing.IndexToNode(index)
      next_node_index = routing.IndexToNode(
        assignment.Value(routing.NextVar(index)))
      route_dist += data["dist_matrix"][node_index, next_node_index]
      load_var = capacity_dimension.CumulVar(index)
      route_load = assignment.Value(load_var)
      time_var = time_dimension.CumulVar(index)
      time_min = assignment.Min(time_var)
      time_max = assignment.Max(time_var)
      plan_output += ' {0} Load({1}) Time({2:.1f},{3:.1f}) ->'.format(
        data["node_id"][node_index],
        route_load,
        time_min / (60 * data["tot_work_hours"]), time_max / (60 * data["tot_work_hours"]))
      vehicle_route.append(data["node_id"][node_index])
      index = assignment.Value(routing.NextVar(index))

    node_index = routing.IndexToNode(index)
    load_var = capacity_dimension.CumulVar(index)
    route_load = assignment.Value(load_var)
    time_var = time_dimension.CumulVar(index)
    route_time = assignment.Value(time_var)
    time_min = assignment.Min(time_var)
    time_max = assignment.Max(time_var)
    total_dist += route_dist
    time_matrix += route_time
    vehicle_route.append(data["node_id"][node_index])
    final_route[vehicle_id] = vehicle_route
    routes_distance.append(route_dist / 1000)
    plan_output += ' {0} Load({1}) Time({2:.1f},{3:.1f})\n'.format(data["node_id"][node_index], route_load,
                                                           time_min / (60 * data["tot_work_hours"]), time_max / (60 * data["tot_work_hours"]))
    plan_output += 'Distance of the route: {0:.3f} km\n'.format(route_dist / 1000)
    plan_output += 'Load of the route: {0}\n'.format(route_load)
    plan_output += 'Minimun estimated time of the route: {0:.3f} days\n'.format(route_time / (60 * data["tot_work_hours"]))
    print(plan_output)
  print('Total distance of all routes: {0:.3f} km'.format(total_dist / 1000))
  # print('Total time of all routes: {0:.3f} days'.format(time_matrix / (60 * data["tot_work_hours"])))
  print('Mean minimun time per route: {0:.3f} days'.format(time_matrix / (60 * data["tot_work_hours"] * data["num_vehicles"])))
  print('Mean minimun time per location: {0:.3f} h\n'.format(time_matrix / (60 * (data["num_locations"] - 1))))
  print('The travel times were estimated considering the average speed of {0} km/h.\n'.format(round(data["vehicle_speed"] / 1000 * 60)))
  final_route["routes_distance"] = routes_distance
  with open(os.path.join(data_dir, 'final_route.json'), 'w') as json_file: 
    json.dump(final_route, json_file)


########
# Main #
########
def main():
  """Entry point of the program"""
  # Instantiate the data problem.
  data = create_data_model()

  # Create Routing Model
  routing = pywrapcp.RoutingModel(data["num_locations"], data["num_vehicles"], data["depot"])
  # Define weight of each edge
  distance_callback = create_distance_callback(data)
  routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
  # Add Capacity constraint
  demand_callback = create_demand_callback(data)
  add_capacity_constraints(routing, data, demand_callback)
  # Add Time Window constraint
  time_callback = create_time_callback(data)
  add_time_window_constraints(routing, data, time_callback)

  # Setting first solution heuristic (cheapest addition).
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  # Solve the problem.
  assignment = routing.SolveWithParameters(search_parameters)
  if assignment:
    printer = print_solution(data, routing, assignment)

if __name__ == '__main__':
  main()