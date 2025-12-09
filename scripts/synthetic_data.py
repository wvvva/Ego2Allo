import json
import numpy as np
import random
from tqdm import tqdm

category = ['height', 'orientation', 'multi-objects', 'location']

fixed_camera = {"position": [0, 0, 0], "orientation": [0, 0, 0], "color": "black"}

color_dict = [
    ['blue',   [0, 0, 1]],
    # ['red',    [1, 0, 0]],
    ['green',  [0, 1, 0]],
    ['yellow', [1, 1, 0]],
    ['purple', [1, 0, 1]],
    ['orange', [1, 0.5, 0]],
    ['brown',  [0.5, 0.25, 0]],
    ['gray',   [0.5, 0.5, 0.5]],
]

def height(file, camera, color_dict, num_object = 2, xy_range_low = -20, xy_range_high = 20, min_diff = 0.5):
    json_ori = {}
    json_ques = {}

    with open(file, 'r') as f:
        json_ori = json.load(f)
        if not json_ori:
            json_ori = []

    json_ques['camera'] = camera

    colors = random.sample(color_dict, num_object)

    low_coor = {}
    low_pos = []
    low_pos.append(np.random.uniform(xy_range_low, xy_range_high))
    low_y = np.random.uniform(xy_range_low, xy_range_high)
    low_pos.append(low_y)
    low_pos.append(np.random.uniform(xy_range_low, 0))
    low_coor['position'] = low_pos
    low_coor['color'] = colors[0][0]
    low_coor['orientation'] = np.random.uniform(xy_range_low, xy_range_high, 3).tolist()

    json_ques['object_1'] = low_coor

    high_coor = {}
    high_pos = []
    high_pos.append(np.random.uniform(xy_range_low, xy_range_high))
    high_pos.append(np.random.uniform(low_y + min_diff, xy_range_high))
    high_pos.append(np.random.uniform(xy_range_low, 0))
    high_coor['position'] = high_pos
    high_coor['color'] = colors[1][0]
    high_coor['orientation'] = np.random.uniform(xy_range_low, xy_range_high, 3).tolist()


    json_ques['object_2'] = high_coor

    for obj in range(2, num_object):
        obj_coor = {}

        pos = []

        pos.append(np.random.uniform(xy_range_low, xy_range_high))
        pos.append(np.random.uniform(xy_range_low, xy_range_high))
        pos.append(np.random.uniform(xy_range_low, 0))

        obj_coor['position'] = pos
        obj_coor['color'] = colors[obj][0]
        obj_coor['orientation'] = np.random.uniform(xy_range_low, xy_range_high, 3).tolist()

        json_ques[f'object_{obj}'] = obj_coor

    json_ques['category'] = "height"
    A = random.sample([1, 2], 1)[0]
    B = 2 if A == 1 else 1
    json_ques['options'] = {"A": json_ques[f'object_{A}']['color'], "B": json_ques[f'object_{B}']['color']}
    json_ques['question'] = f"Which one is higher, the {json_ques['object_1']['color']} box or the {json_ques['object_2']['color']} box?"
    json_ques['answer'] = "B" if B == 2 else "A"

    json_ori.append(json_ques)

    with open(file, 'w') as f:
        json.dump(json_ori, f, indent = 2)
    

def multi_object(file, camera, color_dict, sub_cate = None, num_object = 3, xy_range_low=-20, xy_range_high=20):
    with open(file, "r") as f:
        json_ori = json.load(f)
        if not json_ori:
            json_ori = []

    json_ques = {"camera": camera}

    colors = random.sample(color_dict, num_object)

    for obj in range(num_object):
        obj_coor = {}

        pos = []

        pos.append(np.random.uniform(xy_range_low, xy_range_high))
        pos.append(np.random.uniform(xy_range_low, xy_range_high))
        pos.append(np.random.uniform(xy_range_low, 0))

        obj_coor['position'] = pos
        obj_coor['color'] = colors[obj][0]
        obj_coor['orientation'] = np.random.uniform(xy_range_low, xy_range_high, 3).tolist()

        json_ques[f'object_{obj}'] = obj_coor

    if sub_cate is None:
        subcats = [
            "closer_to",
            "facing",
            "parallel",
            "same_direction",
            "viewpoint_towards_object",
        ]
        category = random.choice(subcats)
    else:
        category = sub_cate
        
    json_ques['category'] = "multi-object"
    json_ques["subcategory"] = category

    def dist(p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))
    
    if category == "closer_to":
        # Consider the real-world 3D locations of the objects. Which is closer to the desk, the bed or the fridge?
        obj1, obj2, obj3= random.sample(range(num_object), 3)
        d_AB = dist(json_ques[f"object_{obj1}"]["position"], json_ques[f"object_{obj2}"]["position"])
        d_AC = dist(json_ques[f"object_{obj1}"]["position"], json_ques[f"object_{obj3}"]["position"])
        answer = "A" if d_AB < d_AC else "B"

        json_ques["options"] = {"A": json_ques[f"object_{obj2}"]["color"], "B": json_ques[f"object_{obj3}"]["color"]}
        color1 = json_ques[f'object_{obj1}']["color"]
        color2 = json_ques[f'object_{obj2}']["color"]
        color3 = json_ques[f'object_{obj3}']["color"]
        json_ques["question"] = f"Which box is closer to the {color1} box, the {color2} one or the {color3} one?"
        json_ques["answer"] = answer

    elif category == "facing":
        # Consider the real-world 3D locations and orientations of the objects. Which object is the woman facing towards, the computer screen or the flower?
        obj1, obj2, obj3 = random.sample(range(num_object), 3)

        p1 = np.array(json_ques[f"object_{obj1}"]["position"])
        p2 = np.array(json_ques[f"object_{obj2}"]["position"])

        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm == 0:
            direction = np.array([0, 0, 1])
        else:
            direction = direction / norm

        json_ques[f"object_{obj1}"]["orientation"] = direction.tolist()

        A = random.sample([obj2, obj3], 1)[0]
        B = obj3 if A ==obj2 else obj2
        json_ques["options"] = {"A": json_ques[f"object_{A}"]["color"], "B": json_ques[f"object_{B}"]["color"]}
        color1 = json_ques[f'object_{obj1}']["color"]
        color2 = json_ques[f'object_{obj2}']["color"]
        color3 = json_ques[f'object_{obj3}']["color"]
        json_ques["question"] = f"Which box is the {color1} one facing towards, the {color2} one or the {color3} one?"
        json_ques["answer"] = "A" if A ==obj2 else "B"

    elif category == "parallel":
        # Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the red car and the street sign that says "Lakeside AV", parallel of perpendicular to each other?
        obj1, obj2 = random.sample(range(num_object), 2)

        relation = random.choice(["parallel", "perpendicular"])

        ori_A = np.array(json_ques[f"object_{obj1}"]["orientation"])
        ori_A /= np.linalg.norm(ori_A)

        if relation == "parallel":
            json_ques[f"object_{obj2}"]["orientation"] = json_ques[f"object_{obj1}"]["orientation"]
        else:
            ori_B = ori_A
            if np.linalg.norm(ori_B) < 1e-6:
                ori_B = np.array([ori_A[1], -ori_A[0], 0])  
            ori_B /= np.linalg.norm(ori_B)
            json_ques[f"object_{obj2}"]["orientation"] = ori_B.tolist()

        json_ques["options"] = {"A": "parallel", "B": "perpendicular"}
        color1 = json_ques[f'object_{obj1}']["color"]
        color2 = json_ques[f'object_{obj2}']["color"]
        json_ques["question"] = f"Is the {color2} box parallel or perpendicular to the {color1} one?"
        json_ques["answer"] = "A" if relation == "parallel" else "B"

    elif category == "same_direction":
        # Consider the real-world 3D orientations of the objects. Are the yellow sign and the brown menu board facing same or similar directions, or very different directions?
        same_dir = random.sample([True, False], 1)
        obj1, obj2 = random.sample(range(num_object), 2)
        if same_dir:
            json_ques[f"object_{obj2}"]["orientation"] = json_ques[f"object_{obj1}"]["orientation"]

        json_ques["options"] = {"A": "True", "B": "False"}
        color1 = json_ques[f'object_{obj1}']["color"]
        color2 = json_ques[f'object_{obj2}']["color"]
        json_ques["question"] = f"Does the {color1} box and {color2} box face the same direction?"
        json_ques["answer"] = "A" if same_dir else "B"

    elif category == "viewpoint_towards_object":
        # Consider the real-world 3D locations and orientations of the objects. Which side of the white truck is facing the mini stop sign?
        obj1, obj2 = random.sample(range(num_object), 2)

        pos_A = np.array(json_ques[f"object_{obj1}"]["position"])
        ori_A = np.array(json_ques[f"object_{obj1}"]["orientation"])
        ori_A /= np.linalg.norm(ori_A)

        up = np.array([0, 1, 0])
        if abs(np.dot(ori_A, up)) > 0.9:
            up = np.array([1, 0, 0])
        right_A = np.cross(up, ori_A)
        right_A /= np.linalg.norm(right_A)

        side = random.choice(["left", "right"])
        # offset_distance = np.random.uniform(3.0, 6.0)
        if side == "left":
            pos_B = pos_A + right_A
        else:
            pos_B = pos_A - right_A

        json_ques[f"object_{obj2}"]["position"] = pos_B.tolist()

        json_ques["options"] = {"A": "left", "B": "right"}
        color1 = json_ques[f'object_{obj1}']["color"]
        color2 = json_ques[f'object_{obj2}']["color"]
        json_ques["question"] = f"Which side of the {color1} box is facing the {color2}?"
        json_ques["answer"] = "A" if side == "left" else "B"

    json_ori.append(json_ques)
    with open(file, "w") as f:
        json.dump(json_ori, f, indent=2)

    return json_ques

def location(file, camera, color_dict, subcat = None, num_object = 2, xy_range_low=-20, xy_range_high=20):
    with open(file, "r") as f:
        json_ori = json.load(f)
        if not json_ori:
            json_ori = []

    json_ques = {"camera": camera}

    colors = random.sample(color_dict, num_object)

    for obj in range(num_object):
        obj_coor = {}

        pos = []

        pos.append(np.random.uniform(xy_range_low, xy_range_high))
        pos.append(np.random.uniform(xy_range_low, xy_range_high))
        pos.append(np.random.uniform(xy_range_low, 0))

        obj_coor['position'] = pos
        obj_coor['color'] = colors[obj][0]
        obj_coor['orientation'] = np.random.uniform(xy_range_low, xy_range_high, 3).tolist()

        json_ques[f'object_{obj}'] = obj_coor

    if not subcat :
        category = random.choice(["above", "closer_to_camera"])
    else:
        category = subcat

    json_ques['category'] = "location"
    json_ques["subcategory"] = category

    if category == "above":
        obj1, obj2 = random.sample(range(num_object), 2)

        p1 = np.array(json_ques[f"object_{obj1}"]["position"]).copy()

        # Start from p1 but add small horizontal offset and ensure vertical separation
        x_offset = np.random.uniform(-3, 3)
        z_offset = np.random.uniform(-3, 3)

        if p1[1] + 5 < xy_range_high:
            y2 = float(np.random.uniform(p1[1] + 5, xy_range_high))
        else:
            y2 = float(np.random.uniform(xy_range_low, p1[1] - 5))

        # Assign new position for object2
        json_ques[f"object_{obj2}"]["position"] = [
            float(p1[0] + x_offset),
            y2,
            float(p1[2] + z_offset)
        ]
        
        A = random.sample([obj1, obj2], 1)[0]
        B = obj2 if A == obj1 else obj1

        color1 = json_ques[f"object_{obj1}"]["color"]
        color2 = json_ques[f"object_{obj2}"]["color"]
        
        json_ques["options"] = {"A": json_ques[f"object_{A}"]["color"], "B": json_ques[f"object_{B}"]["color"]}
        json_ques["question"] = f"Which box is above the other, the {color1} one or the {color2} one?"
        json_ques["answer"] = "B" if A == obj1 else "A"

    elif category == "closer_to_camera":
        obj1, obj2 = random.sample(range(num_object), 2)

        pos_A = np.array(json_ques[f"object_{obj1}"]["position"])
        pos_B = np.array(json_ques[f"object_{obj2}"]["position"])

        answer = "A" if pos_A[2] > pos_B[2] else "B"

        json_ques[f"object_{obj1}"]["position"] = pos_A.tolist()
        json_ques[f"object_{obj2}"]["position"] = pos_B.tolist()

        color1 = json_ques[f"object_{obj1}"]["color"]
        color2 = json_ques[f"object_{obj2}"]["color"]

        json_ques["options"] = {"A": json_ques[f"object_{obj1}"]["color"], "B": json_ques[f"object_{obj2}"]["color"]}
        json_ques["question"] = f"Which object is closer to the camera, the {color1} one or the {color2} one?"
        json_ques["answer"] = answer

    # print(json_ques)
    json_ori.append(json_ques)
    # print(json_ori)
    with open(file, "w") as f:
        json.dump(json_ori, f, indent=2)

    return json_ques


def orientation(file, camera, color_dict, subcat=None, num_object=2, xy_range_low=-20, xy_range_high=20):
    with open(file, "r") as f:
        json_ori = json.load(f)
        if not json_ori:
            json_ori = []

    json_ques = {"camera": camera}

    colors = random.sample(color_dict, num_object)
    for obj in range(num_object):
        obj_coor = {}
        pos = [
            np.random.uniform(xy_range_low, xy_range_high),
            np.random.uniform(xy_range_low, xy_range_high),
            np.random.uniform(xy_range_low, 0)
        ]
        obj_coor["position"] = pos
        obj_coor["color"] = colors[obj][0]
        obj_coor["orientation"] = np.random.uniform(-1, 1, 3).tolist()
        json_ques[f"object_{obj}"] = obj_coor

    if not subcat:
        category = random.choice(["viewpoint"])
    else:
        category = subcat

    json_ques["category"] = "orientation"
    json_ques["subcategory"] = category

    def normalize(v):
        v = np.array(v)
        return v / (np.linalg.norm(v) + 1e-9)

    if category == "viewpoint":
        obj1 = random.sample(range(num_object), 1)[0]

        cam_pos = np.array([0.0, 0.0, 0.0])
        pos_A = np.array(json_ques[f"object_{obj1}"]["position"])

        # Randomly assign a 3D front orientation for the object
        ori_front = np.random.randn(3)
        ori_front /= np.linalg.norm(ori_front) + 1e-9

        # Define object local axes (right = cross(front, up))
        up = np.array([0, 1, 0])
        right = np.cross(ori_front, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])  # fallback for degenerate case
        right /= np.linalg.norm(right)

        back = -ori_front
        left = -right

        # Vector from object to camera
        v_cam = cam_pos - pos_A
        v_cam /= np.linalg.norm(v_cam) + 1e-9

        # Determine which side’s normal points most toward camera
        side_scores = {
            # "front": float(np.dot(ori_front, v_cam)),
            # "back": float(np.dot(back, v_cam)),
            "left": float(np.dot(left, v_cam)),
            "right": float(np.dot(right, v_cam)),
        }
        facing_side = max(side_scores, key=side_scores.get)

        # Save orientation and question
        json_ques[f"object_{obj1}"]["orientation"] = ori_front.tolist()
        color1 = json_ques[f"object_{obj1}"]["color"]

        json_ques["options"] = {
            "A": "left",
            "B": "right"
        }
        json_ques["question"] = f"Which side of the {color1} box is facing the camera?"
        json_ques["answer"] = "A" if facing_side == "left" else "B"
        json_ques["subcategory"] = "viewpoint"

    # ----------------------------------
    # Save and return
    # ----------------------------------
    json_ori.append(json_ques)
    with open(file, "w") as f:
        json.dump(json_ori, f, indent=2)

    return json_ques


if __name__ == "__main__":
    subcats = [
        "closer_to",
        "facing",
        "parallel",
        "same_direction",
        "viewpoint_towards_object",
    ]
    # for s in subcats:
    for i in tqdm(range(5000)):
        multi_object("/jet/home/ydinga/Ego2Allo/multi_1.json", fixed_camera, color_dict)

    # subcats = [
    #     "above", "closer_to_camera"
    # ]
    # # for s in subcats:
    # for i in tqdm(range(2000)):
    #     location("/jet/home/ydinga/Ego2Allo/location_1.json", fixed_camera, color_dict)

    # subcats = ["viewpoint"]
    # # for s in subcats:
    # for i in tqdm(range(1000)):
    #     orientation("/jet/home/ydinga/Ego2Allo/orien_1.json", fixed_camera, color_dict)

    # for i in tqdm(range(1000)):
    #     height("/jet/home/ydinga/Ego2Allo/height_1.json", fixed_camera, color_dict)