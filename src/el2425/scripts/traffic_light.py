#! /usr/bin/env python3

# imports 
import random

from rclpy.clock import Clock, Duration
from std_msgs.msg import String, Float32

from svea_core import rosonic as rx
from svea_core.interfaces import ShowMarker

class traffic_light(rx.Node):

    ## Constants

    GREEN_TIME  = 5.                # [s]
    YELLOW_TIME = 2.                # [s]
    RED_TIME    = 5.                # [s]

    # dictionary for states
    TRANSITIONS = {
        'Rd': 'YG', # Red           -> Yellow-Green
        'YG': 'Gr', # Yellow-Green  -> Green
        'Gr': 'YR', # Green         -> Yellow-Red
        'YR': 'Rd', # Yellow-Red    -> Red
    }

    ## Parameters

    rate = rx.Parameter(10)
    alpha = rx.Parameter(1.0)

    pos_x = rx.Parameter(0.0)
    pos_y = rx.Parameter(0.0)

    ## Publishers

    state_pub = rx.Publisher(String, '~/state')
    time_pub = rx.Publisher(Float32, '~/time_left')

    ## Interfaces

    marker = ShowMarker()

    ## Methods

    def on_startup(self):

        assert 0 <= self.alpha <= min(self.RED_TIME, self.GREEN_TIME)

        self.switch('Rd')  # initial state is red
        self._loop_tmr = self.create_timer(1.0 / self.rate, self.loop)
    
    def loop(self):

        now = Clock().now()

        # calculate time remaining for current state
        time_left = (self._time0 + self._delta - now).nanoseconds / 1e9
        
        # if time has elapsed, then switch to next state
        if time_left <= 0:
            self.switch()

        # publish current state and time remaining
        msg = String()
        msg.data = self._state
        self.state_pub.publish(msg)

        msg = Float32()
        msg.data = time_left
        self.time_pub.publish(msg)

        # visualization for the lights
        self.visualize_traffic_lights()

    def switch(self, to=None):
        """
        Change the traffic light to the next state
        """
        self._state = (to if to is not None else
                       self.TRANSITIONS[self._state])

        delta = (self.RED_TIME      if self._state == 'Rd' else
                 self.GREEN_TIME    if self._state  == 'Gr' else
                 self.YELLOW_TIME)
        if self._state in ('Rd', 'Gr'):
            delta += random.uniform(-self.alpha, +self.alpha)
        
        # delta determines how long each state lasts
        self._delta = Duration(seconds=delta)

        self._time0 = Clock().now()

    def visualize_traffic_lights(self):
        """Publish the positions of traffic lights for visualization in RViz."""
        
        color = ('red' if self._state == 'Rd' else
                 'green' if self._state == 'Gr' else
                 'yellow' if self._state in ('YG', 'YR') else
                 'black')

        self.marker.place([self.pos_x, self.pos_y, 1.0], color=color)

if __name__ == '__main__':

    traffic_light.main()
