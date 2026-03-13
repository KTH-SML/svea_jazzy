"""
Behaviour-tree primitives
~~~~~~~~~~~~~~~~~~~~~~~~~

Drop-in replacement for the classes we discussed earlier, now packaged so
you can `import btree` from any script or test.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class NodeStatus:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class Node(ABC):
    currentRunningNode: Node | None = None
    @abstractmethod
    def run(self) -> str: ...


class ActionNode(Node):
    """
    Leaf node that simply calls the supplied Python function and expects it
    to return one of the NodeStatus strings.
    """

    def __init__(self, action, name: str | None = None):
        if not callable(action):
            raise TypeError("ActionNode expects a callable")
        self.action = action
        self.name = name or action.__name__

    def run(self) -> str:
        status = self.action()
        #added by Elias
        if status == NodeStatus.RUNNING:
            self.currentRunningNode = self
        else:
            self.currentRunningNode = None

        print(f"  {self.name}: {status}")
        return status


class Composite(Node):
    """
    Convenience superclass for control-flow nodes (Sequence, Fallback, …).

    * Validates that at least one child is provided.
    * Accepts *either* ready-made Nodes *or* bare callables and wraps the
      latter in ActionNodes automatically.
    * Provides the tiny `_trace()` helper for uniform debug prints.
    """

    def __init__(self, *children, name: str | None = None):
        if not children:
            raise ValueError(f"{self.__class__.__name__} needs at least one child")

        # convert raw functions → ActionNode so everything is a Node
        self.children = tuple(
            c if isinstance(c, Node) else ActionNode(c) for c in children
        )
        self.name = name or self.__class__.__name__

    # shared utility for readable debug output
    def _trace(self, msg: str) -> None:
        print(f"{self.name}: {msg}")


class Fallback(Composite):
    """“OR” node – succeeds when the **first** child succeeds."""

    def run(self) -> str:
        self._trace("start (OR)")
        for child in self.children:
            result = child.run()

            if result == NodeStatus.RUNNING:
                self._trace("=> RUNNING")
                self.currentRunningNode = child
                return NodeStatus.RUNNING
            else:
                self.currentRunningNode = None

            if result == NodeStatus.SUCCESS:
                self._trace("=> SUCCESS")
                return NodeStatus.SUCCESS

            # result was FAILURE → try next child

        self._trace("=> FAILURE")
        return NodeStatus.FAILURE


class Sequence(Composite):
    """“AND” node – fails when the **first** child fails."""

    def run(self) -> str:
        self._trace("start (AND)")
        for child in self.children:
            result = child.run()

            if result == NodeStatus.RUNNING:
                self._trace("=> RUNNING")
                self.currentRunningNode = child
                return NodeStatus.RUNNING
            else:
                self.currentRunningNode = None
            
            if result == NodeStatus.FAILURE:
                self._trace("=> FAILURE")
                return NodeStatus.FAILURE

            # result was SUCCESS → continue with next child

        self._trace("=> SUCCESS")
        return NodeStatus.SUCCESS
