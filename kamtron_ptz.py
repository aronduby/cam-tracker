#!/usr/bin/env python3
"""
Pan/tilt control for the Kamtron 826 (and other "MIPC"-protocol) IP cameras.

This wraps the `mipc-camera-client` library, which was reverse-engineered
from the camera's own web UI (the same /ccm/*.js endpoints your camera
exposes at http://192.168.1.180/ccm/...).

Install the dependencies first:
    pip install mipc-camera-client pillow numpy

Usage examples:
    # Nudge right and down
    python3 kamtron_ptz.py --host 192.168.1.180 --user admin --password admin --x 200 --y 80

    # Send it back to its home/leftmost position
    python3 kamtron_ptz.py --host 192.168.1.180 --user admin --password admin --home

    # Take a snapshot while you're at it
    python3 kamtron_ptz.py --host 192.168.1.180 --user admin --password admin --snapshot out.jpg

    # Move, then don't return control until the picture stops changing
    # (there's no "movement finished" signal in this protocol, so this
    # watches consecutive snapshots for pixel differences instead)
    python3 kamtron_ptz.py --host 192.168.1.180 --user admin --password admin --x 300 --wait-stopped
"""

import argparse
import io
import sys
import time

try:
    from mipc_camera_client import MipcCameraClient
except ImportError:
    sys.exit(
        "Missing dependency. Install it first with:\n"
        "    pip install mipc-camera-client"
    )


def wait_until_stopped(
    cam: "MipcCameraClient",
    poll_interval: float = 0.4,
    threshold: float = 6.0,
    stable_frames: int = 3,
    timeout: float = 20.0,
    verbose: bool = True,
) -> float:
    """
    Poll snapshots and wait until the picture stops changing between frames.

    There's no documented "motor finished moving" signal from the camera
    itself (the PTZ control ack just confirms the command was received,
    not that the move completed), so this watches the actual image instead:
    once several consecutive frames come back nearly identical, the camera
    is assumed to have physically stopped.

    Returns the number of seconds it took to settle. Raises TimeoutError
    if it never settles within `timeout` seconds (could mean it's still
    moving, or something else is changing in the frame, e.g. IR/exposure).
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        sys.exit(
            "wait_until_stopped needs Pillow and numpy. Install with:\n"
            "    pip install pillow numpy"
        )

    def grab_gray_array(size=(160, 120)):
        img = Image.open(io.BytesIO(cam.get_image())).convert("L").resize(size)
        return np.asarray(img, dtype=np.float32)

    start = time.monotonic()
    prev = grab_gray_array()
    consecutive_stable = 0

    while time.monotonic() - start < timeout:
        time.sleep(poll_interval)
        curr = grab_gray_array()
        diff = float(np.abs(curr - prev).mean())
        prev = curr

        if verbose:
            print(f"  frame diff: {diff:.2f}")

        if diff < threshold:
            consecutive_stable += 1
            if consecutive_stable >= stable_frames:
                elapsed = time.monotonic() - start
                if verbose:
                    print(f"Settled after {elapsed:.1f}s")
                return elapsed
        else:
            consecutive_stable = 0

    raise TimeoutError(
        f"Camera did not settle within {timeout}s (still moving, or "
        f"something else in frame is changing — try raising --threshold)"
    )


def main():
    parser = argparse.ArgumentParser(description="Pan/tilt a Kamtron/MIPC IP camera")
    parser.add_argument("--host", required=True, help="Camera IP, e.g. 192.168.1.180")
    parser.add_argument("--user", required=True, help="Camera username (as used in the web UI)")
    parser.add_argument("--password", required=True, help="Camera password")

    parser.add_argument("--x", type=int, default=0, help="Relative horizontal movement (pan)")
    parser.add_argument("--y", type=int, default=0, help="Relative vertical movement (tilt)")
    parser.add_argument("--speed-x", type=int, default=80, help="Pan speed (default 80)")
    parser.add_argument("--speed-y", type=int, default=50, help="Tilt speed (default 50)")
    parser.add_argument(
        "--home", action="store_true",
        help="Reset to the camera's home / leftmost-extreme position instead of a relative move",
    )
    parser.add_argument("--snapshot", metavar="FILE", help="Also save a JPEG snapshot to this path")
    parser.add_argument(
        "--wait-stopped", action="store_true",
        help="After moving, poll snapshots until the picture stops changing (see wait_until_stopped)",
    )
    parser.add_argument(
        "--threshold", type=float, default=6.0,
        help="Mean pixel-difference below which two frames count as 'the same' (default 6.0, 0-255 scale)",
    )
    parser.add_argument(
        "--settle-timeout", type=float, default=20.0,
        help="Give up waiting for the camera to settle after this many seconds (default 20)",
    )

    args = parser.parse_args()

    print(f"Connecting to {args.host} as {args.user} ...")
    cam = MipcCameraClient(args.host)
    cam.login(args.user, args.password)

    moved = False
    if args.home:
        print("Sending camera to home/reset position...")
        cam.control_ptz(tilt_x=-360, tilt_y=-360, speed_x=args.speed_x, speed_y=args.speed_y)
        moved = True
    elif args.x or args.y:
        print(f"Moving pan={args.x} tilt={args.y} at speed ({args.speed_x}, {args.speed_y})...")
        cam.control_ptz(tilt_x=args.x, tilt_y=args.y, speed_x=args.speed_x, speed_y=args.speed_y)
        moved = True
    else:
        print("No movement requested (pass --x/--y or --home).")

    if moved and args.wait_stopped:
        print("Waiting for the picture to stop changing...")
        try:
            wait_until_stopped(cam, threshold=args.threshold, timeout=args.settle_timeout)
        except TimeoutError as e:
            print(f"Warning: {e}")

    if args.snapshot:
        print(f"Saving snapshot to {args.snapshot} ...")
        with open(args.snapshot, "wb") as f:
            f.write(cam.get_image())

    print("Done.")


if __name__ == "__main__":
    main()
