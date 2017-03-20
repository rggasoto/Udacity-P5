import numpy as np
from PersistentObstacle import PersistentObstacle
class Car:
    def __init__(self,centr_size=[],bbox=[],PersistentTime = 10,timestamp=0):#Persists for 10 frames
        assert len(centr_size) ==2 or len(bbox)==2,"Car must either have center and size argument or a bouding box argument!"
        self.centr = PersistentObstacle(PersistentTime)
        self.size=  PersistentObstacle(PersistentTime)
        centr = np.zeros((1,2))
        size = np.zeros((1,2))
        if len(centr_size)==2:
            centr_size = np.array(centr_size)
            centr = centr_size[0]
            size = centr_size[1]
        if len(bbox)==2:
            bbox = np.array(bbox)
            size,centr = self.getSizeCenter(bbox)
        self.centr.getPersistentDistance(timestamp,centr)
        self.size.getPersistentDistance(timestamp,size)
        self.appearedFrames = 1
        self.maxSize = self.getArea()
        self.appendbox = None

    def appendBox(self,box):
        if self.appendbox is None:
            self.appendbox = np.array(box);
        else:
            self.appendbox[0][0] = min(self.appendbox[0][0],box[0][0])
            self.appendbox[0][1] = min(self.appendbox[0][1],box[0][1])
            self.appendbox[1][0] = max(self.appendbox[1][0],box[1][0])
            self.appendbox[1][1] = max(self.appendbox[1][1],box[1][1])

    def getArea(self):
        b = self.size.distance
        return b[0]*b[1]

    def getBbox(self):
        # print ('attempt to get bbox:',self.appearedFrames, self.maxSize,self.centr.distance,self.size.distance)
        bbox = (tuple(np.int32(self.centr.distance - self.size.distance/2)),tuple(np.int32(self.centr.distance + self.size.distance/2)))
        # print(bbox,self.centr.distance,self.size.distance/2)
        if (self.appearedFrames > 15 and self.maxSize > (2048)):# or self.maxSize > (4098):
            return bbox
    def getSizeCenter(self,bbox):
        bbox = np.array(bbox)
        size = bbox[1]-bbox[0]
        centr = bbox[0]+ self.size.distance/2
        #print(size,centr)
        #input()
        return size,centr

    def getLength(self):
        return max(list(self.size.distance))

    def matchbbox(self,bbox):
        size,centr = self.getSizeCenter(bbox)
        #print('matching  distance: ',np.linalg.norm(self.centr.distance-centr))
        return np.linalg.norm(self.centr.distance-centr) < max(list(self.size.distance))/2 + 10

    def update(self,timestamp):
        bbox = self.appendbox
        size = None
        centr = None
        #print (bbox)
        if not bbox is None:
            size,centr = self.getSizeCenter(bbox)
            # print (size,centr)
            self.appearedFrames +=1
        # print('update car with ',size,centr)
        self.centr.getPersistentDistance(timestamp,centr)
        self.size.getPersistentDistance(timestamp,size)
        self.maxSize = max(self.maxSize,self.getArea())
        if self.centr.isreset:
            self.appearedFrames = 0;
        self.appendbox = None
