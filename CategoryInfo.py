"""
# one I choose
object_detection_categories_dict = {"unknown": 0, "bathtub": 1, "bed": 2, "shelf": 3, "chair": 4,
                                    "counter": 5, "table": 6, "door": 7, "cabinet": 8, "garbage_bin": 9,
                                    "lamp": 10, "monitor": 11, "pillow": 12,
                                    "sofa": 13, "toilet": 14, "sink": 15, "dresser": 16}
"""
# Object detection dicts
# original 19 object detection category
# use these for both nyudv2 and sun rgbd
original_object_detection_categories_dict = {"unknown": 0, "bathtub": 1, "bed": 2, "bookshelf": 3, "box": 4, "chair": 5, "counter": 6, "desk": 7, "door": 8,
                                    "dresser": 9, "garbage-bin": 10, "lamp": 11, "monitor": 12, "night-stand": 13, "pillow": 14, "sink": 15,
                                    "sofa": 16, "table": 17, "television": 18, "toilet": 19}

# Instance segmentation dicts - 18 classes including background
#NYUDV2, missing monitor and garbage-bin
NYUDv2_instance_segmentation_categories_dict = {"unknown": 0, "bathtub": 1, "bed": 2, "bookshelf": 3, "box": 4, "chair": 5,
                                               "counter": 6, "desk": 7, "door": 8, "dresser": 9, "lamp": 10, "night-stand": 11,
                                               "pillow": 12, "sink": 13, "sofa": 14, "table": 15, "television": 16, "toilet": 17}

# SUN RGBD - 14 classes including background
# removed bookshelf, cabinet, counter, dresser
SUN_RGBD_instance_segmentation_categories_dict = {"unknown": 0, "bathtub": 1, "bed": 2, "chair": 3,
                                    "table": 4, "door": 5, "garbage-bin": 6,
                                    "lamp": 7, "monitor": 8, "pillow": 9,
                                    "sofa": 10, "toilet": 11, "sink": 12, "desk": 13}


# Semantic segmentation dicts

#Note that for NYUDv2, there are some categories from the 37 class version that will never appear, these are
# 20, 24, 25, 28, 32 -> missing floor mat, fridge, tv, shower-curtain, night-stand
# however in the instance segmentation we add back tv and night stand
# so for panoptic s


semantic_37_dict = {0: "unknown", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
                    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves", 16:"curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floor-mat",
                    21:"clothes", 22: "ceiling", 23: "books", 24: "fridge", 25: "tv", 26: "paper", 27: "towel", 28: "shower-curtain", 29: "box", 30: "whiteboard",
                    31:"person", 32: "night-stand", 33:"toilet", 34: "sink", 35: "lamp", 36: "bathtub", 37: "bag"}

semantic_40_dict = {0: "unknown", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window", 10:"bookshelf",
                    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves", 16:"curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floor_mat",
                    21:"clothes", 22: "ceiling", 23:"books", 24: "fridge", 25: "tv", 26:"paper", 27: "towel", 28: "shower-curtain", 29: "box", 30: "whiteboard",
                    31:"person", 32: "night_stand", 33: "toilet", 34: "sink", 35: "lamp", 36:"bathtub", 37: "bag", 38:"other-struct", 39: "other-furniture", 40: "other-prop"}


# Panoptic Segmentation Dicts
SUN_RGBD_ps_dict = {0: "unknown", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
                    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves", 16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floor-mat",
                    21: "clothes", 22: "ceiling", 23: "books", 24: "fridge", 25: "tv", 26: "paper", 27: "towel", 28: "shower-curtain", 29: "box", 30: "whiteboard",
                    31: "person", 32: "night-stand", 33: "toilet", 34: "sink", 35: "lamp", 36: "bathtub", 37: "bag", 38: "garbage-bin"}


# remove floor mat, fridge, shower curtain from 37 categories
# 34 categories
# additionally I removed paper, shelves, books, bag
NYUDv2_ps_dict = {0: "unknown", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
                    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "curtain", 16: "dresser", 17: "pillow", 18: "mirror",
                    19: "clothes", 20: "ceiling", 21: "tv", 22: "towel", 23: "box", 24: "whiteboard",
                    25: "person", 26: "night-stand", 27: "toilet", 28: "sink", 29: "lamp", 30: "bathtub"}


map_37_to_30 = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 16:15, 17:16, 18:17, 19:18, 21:19,
22:20, 25:21, 27:22, 29:23, 30:24, 31:25, 32:26, 33:27, 34:28, 35:29, 36:30}

# note garbage bin is not in the semantic 37 dict
SUN_RGBD_is_to_seg_map = {1: 36, 2: 4, 3: 5, 4: 7, 5: 8, 6: 38, 7: 35, 8: 25, 9: 18, 10: 6, 11: 33, 12: 34, 13: 14}

NYUDv2_is_to_seg_map = {1: 30, 2: 4, 3: 10, 4: 23, 5: 5, 6: 12, 7: 14, 8: 8, 9: 16, 10: 29, 11: 26, 12: 17, 13: 28, 14: 6, 15: 7, 16: 21, 17: 27}
