import time
from svea_charging.third_party.btree import NodeStatus, Node, ActionNode, Sequence, Fallback #imported behaviour tree implementation from Elisa Bin's repo: https://github.com/elisabin/btree



class Btree:
    def __init__(self, defaultBatteryLevel=20.0, communicationStatus = True):
        self.batteryLevel = float(defaultBatteryLevel) #percentage, we could change this
        self.communicationStatus = communicationStatus
        self.chargerVisible = False
        self.vehicleAligned = False
        self.chargingActive = False
        self.chargingError = False
        self.tickingFreqency = 10.0 #Hz
        self.period = 1.0 / self.tickingFrequency
        self.batteryLevelUpperThreshold = 80.0 #percentage
        self.batteryLevelLowerThreshold = 20.0 #percentage
        self.max_ticks = 100 #is this necessary?? ASK KAJ. max number of ticks before we stop the process, can be changed depending on how long we want to run the process for

        # the tree should be built according to our logic, and should use the functions defined below as leaf nodes
        self.tree = Sequence(
            Sequence(
                Fallback(
                    ActionNode(self.communicationOk),
                    ActionNode(self.alertCommunicationError),
                    name="communicationGuard",
                ),
                Fallback(
                    ActionNode(self.atChargingArea),
                    ActionNode(self.alertChargingAreaError),
                    name="chargingAreaCheck",
                ),
            ),
            Sequence(
                Fallback(
                    ActionNode(self.inProximityArea),
                    ActionNode(self.driveToProximityArea),
                    name="proximityAreaCheck",
                ),
                Fallback(
                    ActionNode(self.vehicleDocked),
                    ActionNode(self.dockVehicle),
                    name="docking",
                ),
            ),
            Sequence(
                Fallback(
                    ActionNode(self.chargingProcessOk),
                    ActionNode(self.alertChargingError),
                    name="chargingProcessCheck",
                ),
                Sequence(
                    ActionNode(self.doneCharging),
                    ActionNode(self.undockVehicle),
                    name="undocking",
                ),
            ),
            name = "root"
        )



    def tick(self):
        self.tree.tick()
        time.sleep(self.period) #should this sleep here? idk if this is the approach. ASK KAJ and Elisa

    
    @property
    def state(self):
        return self.tree.currentRunningNode.name


    #condition
    def communicationOk(self):
        #check if communication with the car is ok
        if self.communicationStatus:
            return NodeStatus.SUCCESS
        else:
            return NodeStatus.FAILURE
        
    #action
    def alertCommunicationError(self):
        #alert the user that communication with the car is not ok, check with Kaj what is the best way to do handle this
        print("ALERT: Communication issues.")
        return NodeStatus.FAILURE #How should we handle this? Should we return FAILURE here, or SUCCESS since we have alerted the user? ASK KAJ and Elisa

    #condition
    def atChargingArea(self):
        #check if vehicle is in CA (Charging Area)
        if True: #should change this to check if we are in the charging area
            return NodeStatus.SUCCESS
        else:
            print("Vehicle is not in the charging area.")
            return NodeStatus.FAILURE

    #action
    def alertChargingAreaError(self):
        #alert the user that the vehicle is not in the charging area
        print("ALERT: Vehicle is not in the charging area.")
        return NodeStatus.FAILURE #How should we handle this? Should we return FAILURE here, or SUCCESS since we have alerted the user? ASK KAJ and Elisa


    #condition
    def inProximityArea(self):
        #check if vehicle is in PA (Proximity Area)
        if True: #should change this to check if we are in the proximity area
            return NodeStatus.SUCCESS
        else:
            print("Vehicle is not in the proximity area.")
            return NodeStatus.FAILURE

    #action
    def driveToProximityArea(self):
        #some action that would drive the vehicle to the proximity area
        try:
            print("Driving to the proximity area...")
            #should add code here to drive to the proximity area
            return NodeStatus.RUNNING
        except Exception as e:
            print(f"Error while driving to the proximity area: {e}")
            return NodeStatus.FAILURE
        

    #condition
    def vehicleDocked(self):
        #check if the vehicle is docked and have contact with the charger
        if self.vehicleDockedStatus:
            return NodeStatus.SUCCESS
        else:
            print("Vehicle is not docked with the charger yet.")
            return NodeStatus.FAILURE

    #action
    def dockVehicle(self):
        #some action that would dock the vehicle
        try:
            if self.vehicleDockedStatus:
                print("Vehicle is now docked with the charger.")
                return NodeStatus.SUCCESS
            print("Docking with the charger...")
            #should add docking code here, and set self.vehicleDockedStatus to True when we are docked with the charger
            return NodeStatus.RUNNING
        except Exception as e:
            print(f"Error while docking with the charger: {e}")
            return NodeStatus.FAILURE

    #condition
    def chargingProcessOk(self):
        #check if the charging process is going ok
        if self.chargingError: # check with Kaj and Nils. add another method which enables chargingError when anomalies are detected.
            return NodeStatus.FAILURE
        else:
            return NodeStatus.SUCCESS

    #action
    def alertChargingError(self):
        #alert the user that the charging process is not going ok
        print("ALERT: Charging process issues detected.")
        return NodeStatus.FAILURE #How should we handle this? Should we return FAILURE here, or SUCCESS since we have alerted the user? ASK KAJ
    
    #condition
    def doneCharging(self):
        #check if the battery is sufficiently charged
        if self.batteryLevel >= self.batteryLevelUpperThreshold:
            return NodeStatus.SUCCESS
        else:
            print("Battery is not sufficiently charged yet.")
            return NodeStatus.FAILURE

    def undockVehicle(self):
        #some action that would undock the vehicle
        try:
            if not self.vehicleDockedStatus:
                print("Vehicle is now undocked from the charger.")
                return NodeStatus.SUCCESS
            print("Undocking from the charger...")
            #should add undocking code here, and set self.vehicleDockedStatus to False when we are undocked from the charger
            return NodeStatus.RUNNING
        except Exception as e:
            print(f"Error while undocking from the charger: {e}")
            return NodeStatus.FAILURE
        

    def run(self):
        for tick in range(self.max_ticks):
            print(f"--- TICK {tick} ---")
            status = self.tree.run()               # one full evaluation
            print(f"* Tree returned: {status}\n")
            time.sleep(self.period)
            if status == NodeStatus.SUCCESS:
                print("Charging process completed successfully.")
                break
            elif status == NodeStatus.FAILURE:
                print("Charging process failed.")
                break
