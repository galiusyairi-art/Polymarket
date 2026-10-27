# IB Gateway for Atlas: Interactive Brokers' gateway (gnzsnz/ib-gateway: IBC logs in by itself and
# keeps it up), on its own small Fly app next to Polybot's. The paper account or the real one (fly.toml).
FROM ghcr.io/gnzsnz/ib-gateway:stable
COPY --chmod=755 start.sh /home/ibgateway/start-fly.sh
CMD ["/home/ibgateway/start-fly.sh"]
