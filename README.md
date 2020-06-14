# GelSight Simulation

This repository contains the necessary ROS packages for executing the GelSight Simulation approach proposed in **(publication reference to be inserted here)**. These packages contain the drivers for running a GelSight sensor (in real world and simulation) and a FDM-Printer for carrying the described experiments (also in real world and simulation). Visit [danfergo.github.io/gelsight-simulation](https://danfergo.github.io/gelsight-simulation/) for more information.

### Index of Packages

| Package       | Description   |
| ------------- | ------------------|
| fdm_printer   | Contains two drivers for running the FDM printer in the real world: a standard ROS subscriber/publisher and a ROS Control Harware Interface. The drivers work by issuing g-code commands to the printer. | 
| fdm_printer_bringup     | Includes the file for launching the printer either in simulation or the real world (sim:=true for simulation)   | 
| fdm_printer_description | The URDF files and STL meshes describing the printer. |
| gelsight_description | The URDF files and STL meshes describing the GelSight sensor. The modeling is based on sensor proposed [here](https://ieeexplore.ieee.org/document/6943123). |
| **gelsight_gazebo** | This package contains the [driver](gelsight_gazebo/src/gelsight_driver.py) that implements the proposed approach, to be used in simulation. |
| gelsight_simulation | Scripts and Materials used to carry the experiments described in the paper. 

To run the collection of tactile images using the simulated setup.
```
roscore
roslaunch gelsight_simulation dc.launch sim:=true
rosrun gelsight_simulation data_collection.py
```