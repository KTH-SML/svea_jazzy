#!/usr/bin/env python3
"""
example.py
----------

Ticks a small behaviour-tree at 10 Hz.  The ‘warm_up’ leaf returns
RUNNING for its first *three* invocations, so the console output will
explicitly show the RUNNING state before the tree eventually settles on
SUCCESS.

Run with:  python example.py
"""
import time
import random

from btree import NodeStatus, ActionNode, Sequence, Fallback

# ---------------------------------------------------------------------------
#  Leaf-node functions                                                       |
# ---------------------------------------------------------------------------

def warm_up():
    """
    First three ticks → RUNNING, afterwards forever → SUCCESS.

    Demonstrates a task that needs multiple ticks to finish.
    """
    if not hasattr(warm_up, "_count"):
        warm_up._count = 0
    if warm_up._count < 3:
        warm_up._count += 1
        return NodeStatus.RUNNING
    return NodeStatus.SUCCESS


def random70():
    """Return SUCCESS with 70 % probability, otherwise FAILURE."""
    return NodeStatus.SUCCESS if random.random() < 0.7 else NodeStatus.FAILURE


# ---------------------------------------------------------------------------
#  Build the behaviour tree                                                  |
# ---------------------------------------------------------------------------

tree = Fallback(
    Sequence(                          # must warm-up AND then pass random70
        ActionNode(warm_up,  "Warm-up (3 ticks)"),
        ActionNode(random70, "Random 70 %"),
        name="WorkSequence"
    ),
    ActionNode(lambda: NodeStatus.SUCCESS, "Always-Success Fallback"),
    name="Root"
)

# ---------------------------------------------------------------------------
#  Main loop – tick at 10 Hz                                                 |
# ---------------------------------------------------------------------------

HZ = 10.0
PERIOD = 1.0 / HZ
print("Behaviour-tree example running at 10 Hz – press Ctrl-C to stop\n")

try:
    tick = 0
    while True:
        print(f"--- TICK {tick} ---")
        status = tree.run()               # one full evaluation
        print(f"* Tree returned: {status}\n")
        tick += 1
        time.sleep(PERIOD)
except KeyboardInterrupt:
    print("\n[example] Stopped by user.")
