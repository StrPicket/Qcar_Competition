import subprocess

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    path_follower= Node(
        package ='qcar2_autonomy',
        executable ='path_follower',
        name ='path_follower'
    )

    yolo_system_detector = Node(
        package ='qcar2_autonomy',
        executable='yolo_detector',
        name = 'qcar2_yolo_detector'
    )
    
    traffic_system_detector = Node(
        package ='qcar2_autonomy',
        executable='traffic_system_detector',
        name = 'qcar2_traffic_system_detector'
    )

    ''' TODO: Once finished this launch file must also include
    - Lane detector to help smooth out tracking of lanes while driving
    - Planner server to coordinate which LEDs on the QCar should be on based on trip logic
    '''

    return LaunchDescription([
        path_follower,
        yolo_system_detector,
        traffic_system_detector
        ]
    )