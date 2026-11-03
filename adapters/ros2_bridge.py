"""EXPERIMENTAL ROS 2 bridge. Not run: no ROS 2 installation was available when
this was written, so treat it as a starting point, not a tested adapter.

Subscribes to /robot_agent/instruction (std_msgs/String) and publishes the
closed-loop result as JSON on /robot_agent/result (std_msgs/String). The same
typed tools back the MCP server (tools/), so there is one robot interface.

    ros2 run <your_pkg> ros2_bridge      # or: python -m adapters.ros2_bridge
    ros2 topic pub --once /robot_agent/instruction std_msgs/msg/String "{data: 'put the red block in the bin'}"
"""

import json


def main():
    import rclpy  # imported lazily: ROS 2 is optional
    from rclpy.node import Node
    from std_msgs.msg import String

    from tools.session import RobotSession

    class RobotAgentNode(Node):
        def __init__(self):
            super().__init__("robot_agent")
            self.session = RobotSession()
            self.pub = self.create_publisher(String, "/robot_agent/result", 10)
            self.create_subscription(String, "/robot_agent/instruction", self.on_instruction, 10)

        def on_instruction(self, msg):
            result = self.session.run_instruction(msg.data)
            self.pub.publish(String(data=json.dumps(result)))

    rclpy.init()
    node = RobotAgentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
