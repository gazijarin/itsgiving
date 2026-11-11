#!/usr/bin/env bash
# Run this once before starting its_giving_v2.py
# Creates a named virtual camera that Zoom can find

# Remove any existing loopback first (Iriun may have grabbed one)
sudo modprobe -r v4l2loopback 2>/dev/null || true

# Load with settings Zoom needs:
#   exclusive_caps=1  → apps see it as a proper camera (required for Chrome/Zoom)
#   card_label        → friendly name shown in Zoom's camera list
#   video_nr=10       → use /dev/video10 so it doesn't clash with real cameras
sudo modprobe v4l2loopback \
    devices=1 \
    video_nr=10 \
    card_label="ItsGiving Cam" \
    exclusive_caps=1

echo ""
echo "Virtual camera ready at /dev/video10"
echo "Now run:  python its_giving_v2.py --camera 5"
echo "Then in Zoom: Settings → Video → Camera → pick 'ItsGiving Cam'"
