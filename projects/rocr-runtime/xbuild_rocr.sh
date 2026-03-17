
#!/bin/bash

full=${full:-0}

export HIP_DIR=$(readlink -f ../hip)
echo "HIP_DIR: $HIP_DIR"
mkdir -p build
pushd build

if [[ ${full} -eq 1 ]]; then
  rm -rf *
  cmake -DCMAKE_INSTALL_PREFIX=/tf/rocr-runtime-install  -DCMAKE_BUILD_TYPE=Release ..
fi

make install -j VERBOSE=1
popd
