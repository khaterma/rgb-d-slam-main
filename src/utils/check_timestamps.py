"""
A short script to check wether the timestamps for msg reception by bag are in sync with the header timestamps of the
msgs. Computes the maximum error in seconds & nanoseconds. If divergence is too big might need to rewrite bag with
message header time stamps.
"""

from rospy import Duration
from rosbag import Bag
from pathlib import Path
import tqdm
from utils.data_directory_from_json import get_bag_name

bag_name = get_bag_name()

bag = Bag(Path(__file__).parent.parent / 'rosbags' / bag_name, 'r')

print(f"Checking reception and sent timestamps of messages from bag {bag_name}")
i = 0
all_errors_positive = True
all_msgs_w_header = True
topic_wo_header = None
max_error = Duration(0, 0)
with tqdm.tqdm(total = bag.get_message_count()) as pbar:
    for topic, msg, rec_time in bag.read_messages():
        num_messages_wo_header = 0

        # if topic in ['/tf', '/tf_static']:
        if topic in ['/tf']:
            for t in msg.transforms:
                error = rec_time - t.header.stamp
                if error > max_error:
                    max_error = error
        elif topic in ['/tf_static']:
            continue
        elif msg._has_header:
            error = rec_time - msg.header.stamp
            if error > max_error:
                max_error = error
        else:
            topic_wo_header = topic
            all_msgs_w_header = False
        pbar.update(1)

print(f"Max delay from message sent to message received (secs, nsecs): {max_error.secs}, {max_error.nsecs}")
print(f"All messages apart from /tf_static have header / timestamps: {all_msgs_w_header}, {topic_wo_header}")
