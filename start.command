#!/bin/bash
# Mac: double-click to start the bot. The dashboard opens in your browser.
# To carry on from the server: run it with the export ZIP: ./start.command polybot-export.zip
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
if [ -n "$1" ]; then
  python3 -m polybot --import "$1"
else
  python3 -m polybot
fi
