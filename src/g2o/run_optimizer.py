"""
In this file we implement the pose graph optimization on the odometry and loop closure data using the G2O Framework.
The g2opy repository which is a python wrapper is cloned and build with the instruction specified on the github page.
The run_optimizer function is directly taken from the python/examples/simple_optimizer.py file and adapted accordingly
based on the project requirements.
https://github.com/RainerKuemmerle/g2o
https://github.com/uoip/g2opy
"""

import g2o
import os
from pathlib import Path
from src.g2o.csv_to_g2o_dataset import gen_g2o_dataset
from src.g2o.plot_optimization_graphs import plot_g2o_graphs
from src.g2o.g2o_pose_transforms_to_csv import store_opt_pose_tfs_to_csv
from utils.data_directory_from_json import get_data_directory_path


def slam2d_optimizer(path_input_g2o: Path, path_output_g2o: Path):
    # Define the Block solver to solve th matrix inverse and structure of sparce matrix
    solver = g2o.BlockSolverSE2(g2o.LinearSolverEigenSE2())
    # Using Levenberg-Marquardt algorithm to optimize the problem
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    # Declaring the Sparse Optimizer for solving the SLAM problem
    optimizer = g2o.SparseOptimizer()
    # Setting verbose to True to see the intermediate optimization steps
    optimizer.set_verbose(True)
    # Pass the defined solver to set the optimization algorithm
    optimizer.set_algorithm(solver)
    # Load the input g2o dataset file to the optimizer to form the pose graph
    optimizer.load(str(path_input_g2o))
    # Prints the number of vertices and edges of the pose graph
    print('num vertices:', len(optimizer.vertices()))
    print('num edges:', len(optimizer.edges()), end='\n\n')
    # Run the optimizer by first intializating the optimization problem
    optimizer.initialize_optimization()
    # The optimizer will run at most 100 iterations
    optimizer.optimize(100)
    # Save the optimized pose graph as the
    optimizer.save(str(path_output_g2o))


def run_optimization(plot_pose_graphs: bool = None):

    path_to_csvs = get_data_directory_path()

    # generate g2o dataset file from odometry and loop closure csv files
    path_gen_g2o = gen_g2o_dataset(path_to_csvs)
    # create the path to save the optimized pose graph g2o file
    path_optimized_g2o = os.path.dirname(path_gen_g2o) + '/optimized_pose_graph.g2o'
    # run the slam2d optimization on the g2o dataset
    slam2d_optimizer(path_input_g2o=path_gen_g2o, path_output_g2o=path_optimized_g2o)
    # plot the ground truth, unoptimized and optimized pose graphs
    if plot_pose_graphs is True:
        print("############### PLOTTING THE POSE GRPAHS ####################")
        plot_g2o_graphs(path_to_csvs)
    else:
        print("PLOTTING POSE GRAPHS SET TO FALSE")
    # Store the optimized pose transform to csv for creating joint point clouds
    store_opt_pose_tfs_to_csv(path_optimized_g2o)
    print("OPTIMIZATION ON POSE GRAPHS FINISHED")
