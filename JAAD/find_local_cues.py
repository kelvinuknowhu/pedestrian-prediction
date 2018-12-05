from PIL import Image
import numpy as np
import sys
import json
import os


# Resize pedestrian bounding box to height * 2 (this is the region in which we look for cues)
def expandBbox(topLeft, bottomRight):
	width = bottomRight[0] - topLeft[0]
	height = bottomRight[1] - topLeft[1]
	centerX = topLeft[0] + width // 2
	centerY = topLeft[1] + height // 2
	
	# Don't exceed image bounds when expanding
	newTopLeftX = max(centerX - height, 0)
	newTopLeftY = max(centerY - height, 0)
	newBottomRightX = min(centerX + height, 1920)
	newBottomRightY = min(centerY + height, 1080)
	return newTopLeftX, newTopLeftY, newBottomRightX, newBottomRightY

def featuresInBbox(bbox, mask, mask_type, percentage=0.05):
	if mask_type == 'semantic':
		if mask is None:
			return {
				"road": False,
				"sidewalk": False
			}
		
		topLeftX, topLeftY, bottomRightX, bottomRightY = bbox
		patch = mask[topLeftY : bottomRightY, topLeftX : bottomRightX]
		labels, counts = np.unique(patch, return_counts=True)
		
		threshold = int(patch.size * percentage)
		labels = labels[counts >= threshold]
		
		return {
			"road": 0 in labels,
			"sidewalk": 1 in labels,
		}	
	elif mask_type == 'instance':
		topLeftX, topLeftY, bottomRightX, bottomRightY = bbox
		patch = mask[topLeftX : bottomRightX, topLeftY : bottomRightY]
		labels, counts = np.unique(patch, return_counts=True)
		
		threshold = int(patch.size * percentage)
		labels = labels[counts >= threshold]
		
		return {
			"traffic sign": 50 in labels,
			"vehicle": 100 in labels,
			"traffic light": 150 in labels
		}
	else:
		return {}

	
# python find_local_cues.py <mask type> <dataset directory> <mask directory> 
if __name__ == '__main__':
	"""
	dataDir: dataset directory must include:
	- pedestrian_dataset_folds/: 
		has directories fold1/ to fold5/, and fold_dict.json
	maskDir: mask directory must include:
	- video_0001/, video_0002/, etc:
		has frame masks for each video in which at least one pedestrian is present
	"""
	maskType: str = sys.argv[1]
	dataDir: str = sys.argv[2]
	maskDir: str = sys.argv[3]
	assert maskType == 'semantic' or maskType == 'instance'

	
	fold_dict_filename = '/pedestrian_dataset_folds/fold_dict.json'
	with open(dataDir + fold_dict_filename, 'r') as f:
		fold_dict = json.load(f)
	missingFrames = 0
	hasFrames = 0
	for json_filename in fold_dict:
		fold = fold_dict[json_filename]
		json_path = dataDir + "/" + fold + "/" + json_filename
		with open(json_path, 'r') as f:
			ped_json = json.load(f)

		videoDir = maskDir + "/" + ped_json['video']
		frames = ped_json["frame_data"]
		for i, frame in enumerate(frames):
			index = frame["frame_index"]
			bbox = expandBbox(frame["bb_top_left"], frame["bb_bottom_right"])
			mask_path = videoDir + "/" + str(index) + ".png"
			if os.path.exists(mask_path):
				hasFrames += 1
				mask = np.array(Image.open(mask_path))
			else: 
				missingFrames += 1
				mask = None
			features = featuresInBbox(bbox, mask, maskType)
			print(features)
			# Add features to JSON
			# ped_json["frame_data"][i].update(features)
		
		# os.makedirs(maskDir + "/" + fold, exist_ok=True)
		# with open(maskDir + "/" + fold + "/" + json_filename, "w") as f:
		# 	json.dump(ped_json, f)
	print("Processed frames:", hasFrames)
	print("Missing frames:", missingFrames)
