from __future__ import annotations

from dataclasses import dataclass

from svea_charging.third_party.btree.btree import (
    ActionNode,
    Fallback,
    NodeStatus,
    Sequence,
)


@dataclass
class MissionBlackboard:
    battery_level: float = 20.0
    battery_current: float = -1.0
    battery_voltage: float = 12.0
    communication_ok: bool = True
    charger_visible: bool = False
    line_visible: bool = False
    charging_active: bool = False
    charging_error: bool = False
    dist_to_station: float | None = None
    aruco_distance: float | None = None
    switch_distance_m: float = 2.8
    dock_distance_m: float = 1.6
    active_controller: str = "stanley"
    mission_phase: str = "approach"
    last_tree_status: str = NodeStatus.RUNNING
    last_running_node: str = "startup"


class ChargingMissionTree:
    """
    Minimal behaviour tree for controller handoff during charging approach.

    The tree is intentionally small so it can become the logic owner today and
    grow with new leaves later. It currently owns:
    - controller selection: Stanley -> line follower
    - proximity-based switching
    - simple docking completion
    - communication guard
    """
   
    def __init__(self, blackboard: MissionBlackboard):
        self.bb = blackboard
        self.tree = Sequence(
            Fallback(
                ActionNode(self.communication_ok, "communication_ok"),
                ActionNode(self.handle_communication_error, "handle_communication_error"),
                name="communication_guard",
            ),
            Fallback(
                ActionNode(self.is_near_docking_zone, "is_near_docking_zone"),
                ActionNode(self.run_stanley_approach, "run_stanley_approach"),
                name="approach_phase",
            ),
            Fallback(
                ActionNode(self.is_docked, "is_docked"),
                ActionNode(self.run_line_follower_docking, "run_line_follower_docking"),
                name="docking_phase",
            ),
            name="charging_mission",
        )

    def tick(self) -> str:
        status = self.tree.run()
        self.bb.last_tree_status = status
        self.bb.last_running_node = self._current_running_node_name()
        return status

    @property
    def state(self) -> str:
        return self.bb.last_running_node

    def communication_ok(self) -> str:
        return NodeStatus.SUCCESS if self.bb.communication_ok else NodeStatus.FAILURE

    def handle_communication_error(self) -> str:
        self.bb.active_controller = "idle"
        self.bb.mission_phase = "communication_error"
        return NodeStatus.FAILURE

    def is_near_docking_zone(self) -> str:
        
        distance = self.bb.aruco_distance
        if distance is None:
            return NodeStatus.FAILURE
        if distance <= self.bb.switch_distance_m:
            
            self.bb.mission_phase = "docking"
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

    def run_stanley_approach(self) -> str:
        self.bb.active_controller = "stanley"
        self.bb.mission_phase = "approach"
        return NodeStatus.RUNNING

    def is_docked(self) -> str:
        aruco_distance = self.bb.aruco_distance
        if aruco_distance is None:
            return NodeStatus.FAILURE
        if self.bb.battery_current > -0.5:
            self.bb.active_controller = "idle"
            self.bb.mission_phase = "docked"
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

    def run_line_follower_docking(self) -> str:
        self.bb.active_controller = "line_follower"
        self.bb.mission_phase = "docking"

        if self.bb.charger_visible and self.bb.line_visible:
            return NodeStatus.RUNNING

        return NodeStatus.FAILURE

    def _current_running_node_name(self) -> str:
        current = getattr(self.tree, "currentRunningNode", None)
        if current is None:
            return self.bb.mission_phase
        return getattr(current, "name", self.bb.mission_phase)
