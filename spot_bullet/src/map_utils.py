import pybullet as pb
import sys
import os

def getMap(filename):

    fileMap = open("spot_bullet/src/" + filename, "r")

    map = fileMap.read().split("\n")
    fileMap.close()

    height = len(map)
    width = len(map[0])

    return map

def createWall(filename):
    objects = {}
    c = 0
    map = getMap(filename)
    height = len(map)
    width = len(map[0])
    for i in range(height):
        for j in range(width):
            if map[i][j] == "-":
                wall_id = pb.loadURDF("./spotmicro/block.urdf", basePosition = [height - i + 0.5,  (width - 1 - j) - int(width/2),0], useFixedBase = True)
                objects["wall{}".format(c)] = wall_id
                c += 1
                
            elif map[i][j] == "|":
                wall_id = pb.loadURDF("./spotmicro/block.urdf", basePosition = [height - i + 0.5, (width - 1 - j) - int(width/2),0], useFixedBase = True, baseOrientation = [0,0,1,1])
                objects["wall{}".format(c)] = wall_id
                c += 1

            elif map[i][j] == "┌":
                wall_id = pb.loadURDF("./spotmicro/blockL.urdf", basePosition = [height - i + 0.5, (width - 1 - j) - int(width/2),0], useFixedBase = True, baseOrientation = [0,0,-3,0])
                objects["wall{}".format(c)] = wall_id
                c += 1

            elif map[i][j] == "┘":
                wall_id = pb.loadURDF("./spotmicro/blockL.urdf", basePosition = [height - i + 0.5, (width - 1 - j) - int(width/2),0], useFixedBase = True)
                objects["wall{}".format(c)] = wall_id
                c += 1

            elif map[i][j] == "└":
                wall_id = pb.loadURDF("./spotmicro/blockL.urdf", basePosition = [height - i + 0.5, (width - 1 - j) - int(width/2),0], useFixedBase = True, baseOrientation = [0,0,-1,1])
                objects["wall{}".format(c)] = wall_id
                c += 1
            
            elif map[i][j] == "┐":
                wall_id = pb.loadURDF("./spotmicro/blockL.urdf", basePosition = [height - i + 0.5, (width - 1 - j) - int(width/2),0], useFixedBase = True, baseOrientation = [0,0,1,1])
                objects["wall{}".format(c)] = wall_id
                c += 1

            elif map[i][j] == "+":
                wall_id = pb.loadURDF("./spotmicro/blockC.urdf", basePosition = [height - i + 0.5, (width - 1 - j) - int(width/2),0], useFixedBase = True)
                objects["wall{}".format(c)] = wall_id
                c += 1

            
    return objects