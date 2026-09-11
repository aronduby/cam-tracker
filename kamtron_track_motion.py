#!/usr/bin/env python3
"""
Pan the Kamtron 826 (MIPC-protocol) camera left/right to follow motion,
within a limited horizontal field of view.

There's no true "motion tracking" feature exposed by the camera's own
protocol, and no absolute-position query either -- control_ptz() only
takes RELATIVE moves, and there's no feedback on where the camera
actually is. So this script:

  1. Keeps track of where it THINKS the camera is pointed (current_x),
     starting from a known reference point (the camera's home/leftmost
     position).
  2. Grabs snapshots in a loop and does simple frame-differencing to find
     motion (this is the same technique used in kamtron_ptz.py's
     --wait-stopped, just inverted: watching FOR change instead of
     waiting for it to stop).
  3. When motion is found, computes how far left/right of center it is
     in the frame, and issues a small relative pan command to nudge the
     camera toward it.
  4. Clamps current_x to a [--pan-min, --pan-max] range so the camera
     doesn't wander outside your desired field of view.

IMPORTANT - you must calibrate --pan-min/--pan-max yourself:

  The camera's PTZ units are NOT degrees, and nobody's published the
  conversion factor (even the library's own author left it as a TODO).
  To figure out what range corresponds to 180 degrees for your camera:

    1. Run kamtron_ptz.py with --home to send it to its leftmost extreme.
    2. Use kamtron_ptz.py --x <value> --wait-stopped repeatedly, watching
       the live view, to find how large an --x value it takes to sweep
       across the 180-degree arc you want covered.
    3. Pass that value here as --pan-max (with --pan-min 0, since we
       treat "home" as position 0 / the left edge of the range).

Install dependencies first:
    pip install mipc-camera-client pillow numpy pyyaml

Usage:
    python3 kamtron_track_motion.py --host 192.168.1.180 --user admin \\
        --password admin --pan-max 800

    # Same, but also tilt to a fixed vertical position (e.g. eye-level)
    # before tracking starts -- tracking itself only ever pans, it never
    # re-adjusts tilt once running.
    python3 kamtron_track_motion.py --host 192.168.1.180 --user admin \\
        --password admin --pan-max 800 --y 150

Config files:
    All options can be loaded from a YAML file instead of the command
    line (see myconfig.yaml.example for the option names and defaults):

        python3 kamtron_track_motion.py --config myconfig.yaml

    Any option also passed on the command line overrides the value from
    the config file, e.g.:

        python3 kamtron_track_motion.py --config myconfig.yaml --gain 200
"""

import argparse
import io
import sys
import time

try:
    from mipc_camera_client import MipcCameraClient
    from PIL import Image
    import numpy as np
    import yaml
except ImportError:
    sys.exit("Missing dependency. Install it first with:\n    pip install -r requirements.txt")


# Options that configure config-file handling itself, not tracking
# behavior -- these are never read from a config file.
CONFIG_META_DESTS = ("help", "config")


def grab_gray_array(cam: "MipcCameraClient", size):
    img = Image.open(io.BytesIO(cam.get_image())).convert("L").resize(size)
    return np.asarray(img, dtype=np.float32)


def find_motion_offset(prev: "np.ndarray", curr: "np.ndarray", pixel_threshold: float, min_motion_fraction: float):
    """
    Compare two grayscale frames and, if there's enough of a change,
    return a normalized horizontal offset in [-1.0, 1.0] for where the
    motion is centered (-1 = far left of frame, 0 = center, 1 = far right).
    Returns None if there isn't enough motion to act on.
    """
    diff = np.abs(curr - prev)
    mask = diff > pixel_threshold

    total_pixels = mask.size
    motion_pixels = int(mask.sum())
    if motion_pixels < min_motion_fraction * total_pixels:
        return None

    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return None

    width = mask.shape[1]
    center_col = float(cols.mean())
    offset = (center_col - width / 2.0) / (width / 2.0)
    return offset


def build_parser():
    parser = argparse.ArgumentParser(description="Pan a Kamtron/MIPC camera to follow motion")

    parser.add_argument("--config", metavar="FILE",
                         help="Load option values from this YAML file. Options also given "
                              "directly on the command line take precedence over values "
                              "from the file.")

    parser.add_argument("--host", help="Camera IP/hostname (required, directly or via --config)")
    parser.add_argument("--user", help="Camera login username (required, directly or via --config)")
    parser.add_argument("--password", help="Camera login password (required, directly or via --config)")

    parser.add_argument("--pan-min", type=int, default=0,
                         help="Leftmost allowed pan position, in control_ptz units (default 0 = home)")
    parser.add_argument("--pan-max", type=int,
                         help="Rightmost allowed pan position, in control_ptz units -- "
                              "calibrate this yourself, see the module docstring "
                              "(required, directly or via --config)")
    parser.add_argument("--y", type=int, default=0,
                         help="Tilt (vertical) position to move to on startup, relative to home, "
                              "in control_ptz units (default 0 = don't tilt away from home). "
                              "This is set once at startup and held fixed -- tracking only pans.")
    parser.add_argument("--speed-x", type=int, default=60, help="Pan motor speed (default 60)")
    parser.add_argument("--speed-y", type=int, default=60, help="Tilt motor speed, used only for "
                         "the startup move to --y (default 60)")

    parser.add_argument("--gain", type=float, default=120.0,
                         help="How many pan units to move per full-frame-width of motion offset "
                              "(default 120 -- raise for bigger steps, lower for gentler tracking)")
    parser.add_argument("--deadzone", type=float, default=0.15,
                         help="Ignore motion within this fraction of center (0-1) to avoid jitter (default 0.15)")
    parser.add_argument("--min-motion-fraction", type=float, default=0.02,
                         help="Fraction of the frame that must change to count as real motion, "
                              "not noise (default 0.02 = 2%% of pixels)")
    parser.add_argument("--pixel-threshold", type=float, default=25.0,
                         help="Per-pixel brightness change (0-255) to count as 'changed' (default 25)")

    parser.add_argument("--poll-interval", type=float, default=0.5,
                         help="Seconds between snapshots while watching for motion (default 0.5)")
    parser.add_argument("--cooldown", type=float, default=2.0,
                         help="Minimum seconds between pan commands, so the motor isn't spammed "
                              "and frames have time to settle after a move (default 2.0)")
    parser.add_argument("--settle-time", type=float, default=1.5,
                         help="Seconds to wait after issuing a pan command before trusting the "
                              "next frame diff again (avoids mistaking the pan itself for motion)")

    parser.add_argument("--recenter-after", type=int, default=10,
                         help="After this many consecutive poll rounds with no motion at all, pan "
                              "back to the middle of [--pan-min, --pan-max]. Set to 0 to disable "
                              "(default 10)")

    parser.add_argument("--no-home", action="store_true",
                         help="Don't send the camera home at startup, and don't move to the "
                              "initial (center, --y) position either -- assume it's already "
                              "positioned correctly and skip straight to tracking")
    parser.add_argument("--home-settle-time", type=float, default=6.0,
                         help="Seconds to wait after the startup home command before starting to "
                              "track (a full-range move takes longer than a small nudge)")
    parser.add_argument("--initial-position-settle-time", type=float, default=3.0,
                         help="Seconds to wait after moving to the initial position (center of "
                              "[--pan-min, --pan-max], --y) before starting to track (default 3.0)")

    return parser


def configurable_actions(parser):
    """All argparse actions that can be set from a config file."""
    return [a for a in parser._actions if a.dest not in CONFIG_META_DESTS]


def load_config_file(parser, path):
    """Read a YAML config file and return a dict of {argparse dest: value}."""
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        sys.exit(f"Could not parse config file {path}: {e}")

    data = data or {}
    if not isinstance(data, dict):
        sys.exit(f"Config file {path} must contain a YAML mapping of option names to values")

    actions_by_key = {a.dest.replace("_", "-"): a for a in configurable_actions(parser)}

    file_values = {}
    for key, value in data.items():
        action = actions_by_key.get(str(key).replace("_", "-"))
        if action is None:
            sys.exit(f"Unknown option '{key}' in {path} "
                      f"(expected one of: {', '.join(sorted(actions_by_key))})")
        # YAML already parses plain ints/floats/bools natively, but coerce
        # in case a value was quoted as a string (e.g. pan-max: "800").
        if isinstance(value, str) and callable(action.type):
            value = action.type(value)
        file_values[action.dest] = value

    return file_values


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        file_values = load_config_file(parser, args.config)
        parser.set_defaults(**file_values)
        args = parser.parse_args()  # reparse so explicit CLI flags still win over the file

    missing = [flag for flag, val in (
        ("--host", args.host), ("--user", args.user),
        ("--password", args.password), ("--pan-max", args.pan_max),
    ) if val is None or val == ""]
    if missing:
        sys.exit(f"Missing required argument(s): {', '.join(missing)} "
                  "(pass them on the command line, or put them in a --config file)")

    if args.pan_max <= args.pan_min:
        sys.exit("--pan-max must be greater than --pan-min")

    frame_size = (160, 120)

    print(f"Connecting to {args.host} as {args.user} ...")
    cam = MipcCameraClient(args.host)
    cam.login(args.user, args.password)

    center_x = (args.pan_min + args.pan_max) // 2
    current_x = center_x

    if not args.no_home:
        print("Homing camera to leftmost position...")
        cam.control_ptz(tilt_x=-360, tilt_y=-360, speed_x=args.speed_x, speed_y=args.speed_y)
        time.sleep(args.home_settle_time)

        print(f"Moving to initial position (pan={center_x}, y={args.y})...")
        cam.control_ptz(tilt_x=center_x, tilt_y=args.y, speed_x=args.speed_x, speed_y=args.speed_y)
        time.sleep(args.initial_position_settle_time)
    else:
        print(f"Skipping home step, assuming camera is already at pan position {current_x} and y={args.y}")

    print(f"Tracking motion. Pan range [{args.pan_min}, {args.pan_max}], center={center_x}. Ctrl+C to stop.")

    prev = grab_gray_array(cam, frame_size)
    last_move_time = 0.0
    no_motion_streak = 0

    try:
        while True:
            time.sleep(args.poll_interval)
            curr = grab_gray_array(cam, frame_size)

            # Right after a move, the frame changed because WE panned it,
            # not because something moved in the scene -- don't act on
            # that, just resync our baseline frame and keep going.
            now = time.monotonic()
            if now - last_move_time < args.settle_time:
                prev = curr
                continue

            offset = find_motion_offset(
                prev, curr,
                pixel_threshold=args.pixel_threshold,
                min_motion_fraction=args.min_motion_fraction,
            )
            prev = curr

            if offset is None:
                no_motion_streak += 1

                if args.recenter_after > 0 and no_motion_streak >= args.recenter_after:
                    if current_x != center_x:
                        print(f"No motion for {no_motion_streak} rounds -> recentering "
                              f"from {current_x} to {center_x}")
                        cam.control_ptz(tilt_x=center_x - current_x, tilt_y=0,
                                         speed_x=args.speed_x, speed_y=0)
                        current_x = center_x
                        last_move_time = time.monotonic()
                    no_motion_streak = 0

                continue

            # Any detected motion (even if too small/central to act on) counts
            # as "something's happening" and resets the idle counter.
            no_motion_streak = 0

            if abs(offset) < args.deadzone:
                print(f"Motion detected near center (offset={offset:+.2f}), no move needed")
                continue

            if now - last_move_time < args.cooldown:
                continue  # still cooling down from the last move

            step = int(offset * args.gain)
            target_x = max(args.pan_min, min(args.pan_max, current_x + step))
            actual_step = target_x - current_x

            if actual_step == 0:
                print(f"Motion at offset={offset:+.2f} but already at pan limit ({current_x})")
                continue

            print(f"Motion at offset={offset:+.2f} -> panning by {actual_step} "
                  f"(target position {target_x}/{args.pan_max})")
            cam.control_ptz(tilt_x=actual_step, tilt_y=0, speed_x=args.speed_x, speed_y=0)
            current_x = target_x
            last_move_time = time.monotonic()

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
