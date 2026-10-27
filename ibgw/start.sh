#!/bin/bash
# Fly's private network between apps is IPv6, and the image's own relay listens on IPv4 only: one
# more relay, from the private network to the gateway's API (which takes local connections only):
# the paper account's on 4002, the real one's on 4001 (TRADING_MODE, set by the deploy workflow).
# Nothing here is public: the app has no public ports.
API=4002
[ "$TRADING_MODE" = "live" ] && API=4001
(
  while true; do
    socat TCP6-LISTEN:4006,fork,reuseaddr TCP4:127.0.0.1:$API
    sleep 2
  done
) &
exec /home/ibgateway/scripts/run.sh
