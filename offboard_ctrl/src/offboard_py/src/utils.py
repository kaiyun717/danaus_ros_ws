def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5

def setpoint_reached(setpoint, current_pose, tolerance=0.1):
    return distance(setpoint.pose.position, current_pose.pose.position) < tolerance