import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_base_id(filename):
    return filename.replace('_pre_disaster.png', '') \
                   .replace('_post_disaster.png', '')

def load_ids(path):
    with open(path, "r") as f:
        ids = [line.strip() for line in f.readlines()]
    return ids

def load_json(path):
    with open(path) as f:
        return json.load(f)
    

def extract_buildings(json_data):
    buildings = []

    for obj in json_data["features"]["xy"]:
        props = obj.get("properties", {})
        wkt = obj.get("wkt", None)

        if wkt is None:
            continue

        damage = props.get("subtype", None)

        # ignorar si no tiene label (por seguridad)
        if damage is None or damage == "un-classified":
            continue

        buildings.append({
            "wkt": wkt,
            "damage": damage
        })

    return buildings

def wkt_to_bbox(wkt):
    try:
        coords = wkt.replace("POLYGON ((", "").replace("))", "")
        points = coords.split(",")

        xs, ys = [], []

        for p in points:
            x, y = map(float, p.strip().split())
            xs.append(x)
            ys.append(y)

        xmin, ymin = min(xs), min(ys)
        xmax, ymax = max(xs), max(ys)

        return int(xmin), int(ymin), int(xmax), int(ymax)

    except:
        return None
    
def crop_image(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    return img.crop((xmin, ymin, xmax, ymax))


def is_valid_bbox(bbox, min_size=20):
    xmin, ymin, xmax, ymax = bbox
    return (xmax - xmin > min_size) and (ymax - ymin > min_size)