#!/bin/bash
# Fly's private network between apps is IPv6, and the image's own relay listens on IPv4 only: one
# more relay, from the private network to the gateway's paper API (which takes local connections
# only). Nothing here is public: the app has no public ports.
(
  while true; do
    socat TCP6-LISTEN:4006,fork,reuseaddr TCP4:127.0.0.1:4002
    sleep 2
  done
) &
exec /home/ibgateway/scripts/run.sh
