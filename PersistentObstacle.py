import numpy as np
class PersistentObstacle:
    """PersistentObstacle: Provides a framework for estimating the obstacle
    dynamics given its last reported distance, velocity, acceleration, up to a
    determined timeout. When it expires, report -1 (no obstacle) """
    def __init__(self, PersistentTime):
        self.T = PersistentTime
        #Timestamp used for timeout
        self.validtimestamp = -1
        #Timestamp used for interval between calls
        self.timestamp = -1
        self.speed = np.array([0,0])
        self.distance = np.array([0,0])
        self.accel = np.array([0,0])
        self.dt = -1
        self.isreset = True


    def reset(self):
        """Resets the Persistency """
        #Timestamp used for timeout
        self.validtimestamp = -1
        #Timestamp used for interval between calls
        self.timestamp = -1
        self.speed = np.zeros([0,0])
        self.distance = np.array([0,0])
        self.accel = np.array([0,0])
        self.isreset = True

    def getPersistentDistance(self,timestamp,D=None,ego_v=np.array([0,0]),V=None,A=None):
        #if not reset
        # print(D)
        if not self.isreset:
            # print(timestamp,self.validtimestamp)
            #if time out
            if timestamp - self.validtimestamp > self.T:
                self.reset()
            else:
                #time interval since last measurement
                self.dt = timestamp - self.timestamp

                self.timestamp = timestamp
                # print (self.distance,self.speed,self.accel)
                # input()
                if V is None:
                    V = self.estimateSpeed(D,ego_v)
                if A is None:
                    A = self.estimateAccel(V,D)
                #update self distance
                if D is None or (D < 0).any(): #Invalid measurement - Estimate distance
                    self.distance = self.distance #+ (V-ego_v)*self.dt #+ 0.5*A*self.dt**2
                else: #Valid measurement, use real measurement Distance`
                    self.distance = D;#(self.distance + D*4)/5
                    #Classify as valid reading, reset timeout
                    self.validtimestamp = timestamp

                print (self.distance,self.speed,self.accel)
                #print(D)
                #input()
                #Move on to compute Acceleration
                # print (self.speed, V)
                # input()

                #Update self speed (at this point it's either a valid value or -1)
                self.speed = V
                self.accel = A
        else:

            if (not D is None):
                #first valid reading. Can only know Distance
                self.isreset = False
                self.timestamp = timestamp
                self.validtimestamp = timestamp
                self.distance = D
                #assume speed equals ego vehicle, acc = 0
                if V is None:
                    self.speed = ego_v
                else:
                    self.speed = V
                if A is None:
                    self.acceleration = np.array([0,0])
                else:
                    self.acceleration = A

        return self.distance,self.speed,self.accel


    def estimateSpeed(self,D,ego_v):
        v = self.speed + self.accel*self.dt #best estimate is dynamics from acc
        # input("c0")
        if (not D is None) and (D > 0).any():
            # input("c1")
            if (self.distance >0).any():
                # input("c2")
                #have at least two measurements, can estimate relative speed
                v = (D - self.distance)/self.dt
                #Sanity Check - if relative speed is too big , it's a new obstacle,
                #start treating as V equal ego v
                if (abs(v) > 10).any(): #TODO find suitable value for sanity check threshold,
                #treating objects running faster than 10m/s as inconsistent
                    v = self.speed
                return v + ego_v
        return v

    def estimateAccel(self,V,D):
        a = self.accel #best accel guess is previous acceleration
        if (not D is None) and (D > 0).any() : #We assume vehicles only move forward
            #if self.speed >= 0: #Have at least three measurements, can compute estimate accel
            a = (V - self.speed)
            a = a/self.dt
            #Sanity check, if acc is greater than 10m/sˆ2, ignore
            print(a)
            if (abs(a) > 10).any():
                a = self.accel
        return a
