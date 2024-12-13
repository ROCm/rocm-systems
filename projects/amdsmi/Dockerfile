    # Use ubuntu 20.04 base image
    FROM compute-artifactory.amd.com:5000/rocm-base-images/ubuntu-20.04-bld:2024112501

    # Set environment variables
    ENV BUILD_FOLDER=/src/build
    ENV DEB_BUILD="amd-smi-lib*99999-local_amd64.deb"
    ENV DEB_BUILD_TEST='amd-smi-lib-tests*99999-local_amd64.deb'

    # Set up the working directory
    WORKDIR /src

    # Copy the source code into the container
    COPY . /src

    # Run the build and install commands
    RUN rm -rf ${BUILD_FOLDER} && \
        mkdir -p ${BUILD_FOLDER} && \
        cd ${BUILD_FOLDER} && \
        cmake .. -DBUILD_TESTS=ON -DENABLE_ESMI_LIB=ON && \
        make -j $(nproc) VERBOSE=1 && \
        make package && \
        sudo apt install -y ${BUILD_FOLDER}/${DEB_BUILD} && \
        sudo ln -s /opt/rocm/bin/amd-smi /usr/local/bin

    # Verify installation
    RUN python3 -m pip list | grep amd && \
        python3 -m pip list | grep pip && \
        python3 -m pip list | grep setuptools

    # Set the entrypoint
    ENTRYPOINT ["/bin/bash"]
