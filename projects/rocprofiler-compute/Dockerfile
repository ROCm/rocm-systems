# -----------------------------------------------------------------------
# NOTE:
# Dependencies are not included as part of Omniperf.
# It's the user's responsibility to accept any licensing implications 
# before building the project
# -----------------------------------------------------------------------

FROM ubuntu:20.04
WORKDIR /app

USER root

ENV DEBIAN_FRONTEND noninteractive
ENV TZ "US/Chicago"

ADD grafana_plugins/svg_plugin /var/lib/grafana/plugins/custom-svg
ADD grafana_plugins/omniperfData_plugin /var/lib/grafana/plugins/omniperfData_plugin

RUN apt-get update && \
    apt-get install -y apt-transport-https software-properties-common  adduser libfontconfig1 wget curl && \
    wget https://dl.grafana.com/enterprise/release/grafana-enterprise_8.3.4_amd64.deb &&\
    dpkg -i grafana-enterprise_8.3.4_amd64.deb &&\
    echo "deb https://packages.grafana.com/enterprise/deb stable main" | tee -a /etc/apt/sources.list.d/grafana.list && \
    echo "deb [signed-by=/usr/share/keyrings/yarnkey.gpg] https://dl.yarnpkg.com/debian stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get install gnupg && \
    wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc -O server-5.0.asc &&\
    apt-key add server-5.0.asc && \
    echo "deb [trusted=yes arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org.list && \
    wget -q -O - https://packages.grafana.com/gpg.key | apt-key add - && \
    curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | gpg --dearmor | tee /usr/share/keyrings/yarnkey.gpg > /dev/null && \
    apt-get update && \
    apt-get install -y mongodb-org                                      && \
    apt-get install -y tzdata systemd apt-utils npm vim net-tools  	&& \
    mkdir -p /nonexistent                                               && \
    /usr/sbin/grafana-cli plugins install michaeldmoore-multistat-panel && \
    /usr/sbin/grafana-cli plugins install ae3e-plotly-panel             && \
    /usr/sbin/grafana-cli plugins install natel-plotly-panel            && \
    /usr/sbin/grafana-cli plugins install grafana-image-renderer        && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash -           && \
    apt-get install -y yarn nodejs                                      && \
    chown root:grafana /etc/grafana                                     && \
    cd /var/lib/grafana/plugins/omniperfData_plugin                       && \
    npm install                                                         && \
    npm run build                                                       && \
    apt-get autoremove -y                                               && \
    apt-get autoclean -y                                                && \
    cd /var/lib/grafana/plugins/custom-svg                              && \
    yarn install                                                        && \
    yarn build                                                          && \
    yarn autoclean                                                      && \
    sed -i "s/  bindIp.*/  bindIp: 0.0.0.0/" /etc/mongod.conf           && \
    mkdir -p /var/lib/grafana						&& \
    touch /var/lib/grafana/grafana.lib					&& \
    chown grafana:grafana /var/lib/grafana/grafana.lib			&& \
    rm /app/grafana-enterprise_8.3.4_amd64.deb /app/server-5.0.asc

# Overwrite grafana ini file
COPY docker/grafana.ini /etc/grafana

# switch Grafana port to 4000
RUN sed -i "s/^;http_port = 3000/http_port = 4000/" /etc/grafana/grafana.ini && \
    sed -i "s/^http_port = 3000/http_port = 4000/" /usr/share/grafana/conf/defaults.ini

# starts mongo and grafana-server at startup
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh

ENTRYPOINT [ "/docker-entrypoint.sh" ]