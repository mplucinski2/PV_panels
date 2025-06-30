import airsim
import math



def orbit_pattern(center_x, center_y, altitude, radius, num_points, face_center=True):
    """
    classical circular orbit at constant altitude - deterministic version
    """
    poses = []
    center_x = round(center_x, 3)
    center_y = round(center_y, 3)
    altitude = round(altitude, 3)
    radius = round(radius, 3)
    
    for i in range(num_points):
        theta = round(2 * math.pi * i / num_points, 6)
        x = round(center_x + radius * math.cos(theta), 3)
        y = round(center_y + radius * math.sin(theta), 3)
        
        if face_center:
            yaw = round(math.atan2(center_y - y, center_x - x), 6)
        else:
            yaw = round(theta + math.pi/2, 6)
            
        quat = airsim.to_quaternion(0, 0, yaw)
        poses.append(airsim.Pose(airsim.Vector3r(x, y, -altitude), quat))
    
    return poses

def square_pattern(center_x, center_y, altitude, side_length, points_per_side, face_center=True):
    """
    square (moving to four corners in sequence)
    """
    poses = []
    half_side = side_length / 2
    corners = [
        (center_x - half_side, center_y - half_side),
        (center_x + half_side, center_y - half_side),
        (center_x + half_side, center_y + half_side),
        (center_x - half_side, center_y + half_side) 
    ]
    for i in range(4): 
        start_corner = corners[i]
        end_corner = corners[(i + 1) % 4]
        
        for j in range(points_per_side):
            if i == 3 and j == points_per_side - 1:
                break
                
            t = j / points_per_side
            x = start_corner[0] + t * (end_corner[0] - start_corner[0])
            y = start_corner[1] + t * (end_corner[1] - start_corner[1])
            
            if face_center:
                yaw = math.atan2(center_y - y, center_x - x)
            else:
                dx = end_corner[0] - start_corner[0]
                dy = end_corner[1] - start_corner[1]
                yaw = math.atan2(dy, dx)
            
            quat = airsim.to_quaternion(0, 0, yaw)
            poses.append(airsim.Pose(airsim.Vector3r(x, y, -altitude), quat))
    
    return poses

def lawnmower_pattern(min_x, max_x, min_y, max_y, altitude, spacing, face_forward=True):
    """
    simple lawnmover pattern, (back-and-forth) coverage of an area - deterministic version
    """
    poses = []
    min_x = round(min_x, 3)
    max_x = round(max_x, 3)
    min_y = round(min_y, 3)
    max_y = round(max_y, 3)
    altitude = round(altitude, 3)
    spacing = round(spacing, 3)
    
    y = min_y
    going_right = True
    
    while y <= max_y:
        y = round(y, 3)
        
        if going_right:
            x_start, x_end = min_x, max_x
            yaw = 0.0
        else:
            x_start, x_end = max_x, min_x
            yaw = round(math.pi, 6)
        
        line_length = round(abs(x_end - x_start), 3)
        num_points = max(2, int(round(line_length / 2, 0)))
        
        for i in range(num_points):
            t = round(i / (num_points - 1), 6) if num_points > 1 else 0.0
            x = round(x_start + t * (x_end - x_start), 3)
            
            if not face_forward:
                yaw = 0.0
            
            quat = airsim.to_quaternion(0, 0, yaw)
            poses.append(airsim.Pose(airsim.Vector3r(x, y, -altitude), quat))
        
        y += spacing
        going_right = not going_right
    
    return poses

def grid_pattern(center_x, center_y, altitude, width, height, rows, cols, face_center=True):
    """
    moving to points on a 2d grid - deterministic version
    """
    poses = []
    center_x = round(center_x, 3)
    center_y = round(center_y, 3)
    altitude = round(altitude, 3)
    width = round(width, 3)
    height = round(height, 3)
    
    for i in range(rows):
        for j in range(cols):
            if cols > 1:
                x = round(center_x - width/2 + (j * width / (cols - 1)), 3)
            else:
                x = center_x
                
            if rows > 1:
                y = round(center_y - height/2 + (i * height / (rows - 1)), 3)
            else:
                y = center_y
            
            if face_center:
                yaw = round(math.atan2(center_y - y, center_x - x), 6)
            else:
                yaw = 0.0
            
            quat = airsim.to_quaternion(0, 0, yaw)
            poses.append(airsim.Pose(airsim.Vector3r(x, y, -altitude), quat))
    
    return poses

def linear_pattern(start_x, start_y, end_x, end_y, altitude, num_points, face_forward=True):
    """
    just a motion along a straight line - deterministic version
    """
    poses = []
    start_x = round(start_x, 3)
    start_y = round(start_y, 3)
    end_x = round(end_x, 3)
    end_y = round(end_y, 3)
    altitude = round(altitude, 3)
    
    if face_forward:
        yaw = round(math.atan2(end_y - start_y, end_x - start_x), 6)
    else:
        yaw = 0.0
    
    for i in range(num_points):
        t = round(i / (num_points - 1), 6) if num_points > 1 else 0.0
        x = round(start_x + t * (end_x - start_x), 3)
        y = round(start_y + t * (end_y - start_y), 3)
        
        quat = airsim.to_quaternion(0, 0, yaw)
        poses.append(airsim.Pose(airsim.Vector3r(x, y, -altitude), quat))
    
    return poses

def figure_eight_pattern(center_x, center_y, altitude, width, height, num_points):
    poses = []
    
    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        x = center_x + (width/2) * math.sin(t)
        y = center_y + (height/2) * math.sin(2*t)
        dx = (width/2) * math.cos(t)
        dy = height * math.cos(2*t)
        yaw = math.atan2(dy, dx)
        quat = airsim.to_quaternion(0, 0, yaw)
        poses.append(airsim.Pose(airsim.Vector3r(x, y, -altitude), quat))
    
    return poses

def random_pattern(center_x, center_y, altitude, radius, num_points, seed=None):
    """
    random viewpoints inside a 2d circle
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    poses = []
    
    for _ in range(num_points):
        angle = random.uniform(0, 2 * math.pi)
        r = radius * math.sqrt(random.uniform(0, 1)) 
        
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        
        yaw = random.uniform(0, 2 * math.pi)
        
        quat = airsim.to_quaternion(0, 0, yaw)
        poses.append(airsim.Pose(airsim.Vector3r(x, y, -altitude), quat))
    
    return poses

def spiral_pattern(center_x, center_y, altitude, max_radius, num_points, rotations=2):
    """
    spiral motion around a center point
    """
    poses = []
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        theta = rotations * 2 * math.pi * t
        radius = max_radius * t
        
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        yaw = theta + math.pi/2
        
        quat = airsim.to_quaternion(0, 0, yaw)
        poses.append(airsim.Pose(airsim.Vector3r(x, y, -altitude), quat))
    
    return poses

PATTERN_EXAMPLES = {
    'orbit': lambda: orbit_pattern(0, 0, 40, 15, 36, face_center=True),
    'square': lambda: square_pattern(0, 0, 40, 30, 8, face_center=True),
    'lawnmower': lambda: lawnmower_pattern(-20, 20, -20, 20, 40, 5),
    'grid': lambda: grid_pattern(0, 0, 40, 30, 30, 5, 5, face_center=True),
    'linear': lambda: linear_pattern(-15, -15, 15, 15, 40, 20),
    'figure8': lambda: figure_eight_pattern(0, 0, 40, 20, 15, 50),
    'random': lambda: random_pattern(0, 0, 40, 20, 25, seed=42),
    'spiral': lambda: spiral_pattern(0, 0, 40, 20, 40, rotations=3)
} 