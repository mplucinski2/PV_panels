import airsim, os, math, time, datetime, json
from tqdm import tqdm
from pose_patterns import lawnmower_pattern

ALTITUDE_M = 40
AREA_SIZE = 20
SPACING_M = 4
CAM_NAME = "bottom_center"
OUT_DIR = "Dataset_No_Shadows_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SHADOW_CMD = "r.ShadowQuality 5"

WORLD_ORIGIN_LAT = 47.641468 
WORLD_ORIGIN_LON = -122.140165
METERS_PER_DEGREE_LAT = 111320.0
METERS_PER_DEGREE_LON = 85390.0  

def meters_to_gps(x_meters, y_meters, origin_lat, origin_lon):
    if math.isnan(x_meters) or math.isnan(y_meters):
        return float('nan'), float('nan')
    lat = origin_lat + (y_meters / METERS_PER_DEGREE_LAT)
    lon = origin_lon + (x_meters / METERS_PER_DEGREE_LON)
    return lat, lon

def save_metadata(client, pose, timestamp, view_index, metadata_dir, vehicle_name=""):
    position = pose.position
    orientation = pose.orientation
    current_pose = client.simGetVehiclePose(vehicle_name)
    
    roll, pitch, yaw = airsim.utils.to_eularian_angles(orientation)
    roll_deg, pitch_deg, yaw_deg = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    
    current_lat, current_lon = meters_to_gps(
        current_pose.position.x_val, 
        current_pose.position.y_val, 
        WORLD_ORIGIN_LAT, WORLD_ORIGIN_LON
    )
    
    metadata = {
        "timestamp": timestamp.isoformat(),
        "view_index": view_index,
        "coordinates": {
            "position": {
                "x_meters": current_pose.position.x_val,
                "y_meters": current_pose.position.y_val,
                "z_meters": current_pose.position.z_val,
                "gps": {
                    "latitude": current_lat,
                    "longitude": current_lon,
                    "altitude": abs(current_pose.position.z_val)
                }
            }
        },
        "orientation": {
            "quaternion": {
                "w": orientation.w_val,
                "x": orientation.x_val,
                "y": orientation.y_val,
                "z": orientation.z_val
            },
            "euler_degrees": {
                "roll": roll_deg,
                "pitch": pitch_deg,
                "yaw": yaw_deg
            },
            "euler_radians": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw
            }
        },
        "camera": {
            "name": CAM_NAME,
            "pattern_type": "lawnmower",
            "area_size_meters": AREA_SIZE,
            "spacing_meters": SPACING_M,
            "altitude_meters": ALTITUDE_M
        }
    }
    
    metadata_path = os.path.join(metadata_dir, f"view_{view_index:03d}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

os.makedirs(OUT_DIR, exist_ok=True)
images_dir = os.path.join(OUT_DIR, "images")
metadata_dir = os.path.join(OUT_DIR, "metadata")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)

client = airsim.MultirotorClient()
client.confirmConnection()

client.simRunConsoleCommand(SHADOW_CMD)

cx, cy = 0.0, 0.0
print(f"Using fixed center position: ({cx:.6f}, {cy:.6f})")

half_size = AREA_SIZE / 2
poses = lawnmower_pattern(cx - half_size, cx + half_size, cy - half_size, cy + half_size, ALTITUDE_M, SPACING_M, face_forward=True)

dataset_metadata = {
    "dataset_name": OUT_DIR,
    "creation_time": datetime.datetime.now().isoformat(),
    "total_views": len(poses),
    "pattern_type": "lawnmower",
    "pattern_center": {"x": cx, "y": cy, "z": -ALTITUDE_M},
    "area_size_meters": AREA_SIZE,
    "spacing_meters": SPACING_M,
    "altitude_meters": ALTITUDE_M,
    "camera_name": CAM_NAME,
    "world_origin_gps": {"lat": WORLD_ORIGIN_LAT, "lon": WORLD_ORIGIN_LON},
    "mode": "Computer Vision (deterministic)"
}

with open(os.path.join(OUT_DIR, "dataset_info.json"), 'w') as f:
    json.dump(dataset_metadata, f, indent=2)

all_metadata = []
for i, pose in tqdm(enumerate(poses), total=len(poses), desc="capturing views"):
    client.simSetVehiclePose(pose, True)
    time.sleep(0.03)
    
    png = client.simGetImage(CAM_NAME, airsim.ImageType.Scene)
    timestamp = datetime.datetime.now()
    
    image_path = os.path.join(images_dir, f"view_{i:03d}.png")
    airsim.write_file(image_path, png)
    
    metadata = save_metadata(client, pose, timestamp, i, metadata_dir)
    all_metadata.append(metadata)
    
    coords = metadata['coordinates']['position']
    
    print(f"view {i:03d}: pos({coords['x_meters']:.3f}, {coords['y_meters']:.3f}, {coords['z_meters']:.3f}) gps({coords['gps']['latitude']:.6f}, {coords['gps']['longitude']:.6f})")

with open(os.path.join(metadata_dir, "all_metadata.json"), 'w') as f:
    json.dump(all_metadata, f, indent=2)

