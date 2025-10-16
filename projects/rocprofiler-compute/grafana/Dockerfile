# -----------------------------------------------------------------------
# NOTE:
# Dependencies are not included as part of ROCm Compute Profiler.
# It's the user's responsibility to accept any licensing implications
# before building the project
# -----------------------------------------------------------------------

FROM ubuntu:22.04
WORKDIR /app

USER root

ENV DEBIAN_FRONTEND noninteractive
ENV TZ "US/Chicago"
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 20.12.2

ADD plugins/rocprofiler-compute_plugin /var/lib/grafana/plugins/rocprofiler-compute_plugin

# Install Grafana and MongoDB Community Edition
# Note: Grafana install is stubbed to 10.4.3
RUN apt-get update && \
    apt-get install -y adduser libfontconfig1 musl wget && \
    wget -q https://dl.grafana.com/enterprise/release/grafana-enterprise_10.4.3_amd64.deb && \
    dpkg -i grafana-enterprise_10.4.3_amd64.deb && \
    apt-get install -y gnupg curl && \
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg  --dearmor && \
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-7.0.list && \
    apt-get update && \
    apt-get install -y mongodb-org

RUN mkdir /usr/local/nvm && \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
    . $NVM_DIR/nvm.sh && \
    nvm install $NODE_VERSION && \
    nvm alias default $NODE_VERSION && \
    nvm use default

ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

RUN npm --version && \
    node --version

# Install Grafana plugins
RUN apt-get install -y tzdata systemd apt-utils npm vim net-tools && \
    /usr/sbin/grafana-cli plugins install michaeldmoore-multistat-panel && \
    /usr/sbin/grafana-cli plugins install ae3e-plotly-panel && \
    /usr/sbin/grafana-cli plugins install natel-plotly-panel && \
    /usr/sbin/grafana-cli plugins install grafana-image-renderer && \
    /usr/sbin/grafana-cli plugins install aceiot-svg-panel && \
    chown root:grafana /etc/grafana && \
    cd /var/lib/grafana/plugins/rocprofiler-compute_plugin && \
    npm install && \
    npm run build && \
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    sed -i "s/  bindIp.*/  bindIp: 0.0.0.0/" /etc/mongod.conf && \
    mkdir -p /var/lib/grafana && \
    touch /var/lib/grafana/grafana.lib && \
    chown grafana:grafana /var/lib/grafana/grafana.lib

# Overwrite grafana ini file
COPY grafana.ini /etc/grafana

# Switch Grafana port to 4000
RUN sed -i "s/^;http_port = 3000/http_port = 4000/" /etc/grafana/grafana.ini && \
    sed -i "s/^http_port = 3000/http_port = 4000/" /usr/share/grafana/conf/defaults.ini

# Starts mongo and grafana-server at startup
COPY docker-entrypoint.sh /docker-entrypoint.sh

ENTRYPOINT [ "/docker-entrypoint.sh" ]
