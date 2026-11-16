#!/usr/bin/env bash
# Fetch the third-party code and data that some experiments need, at the exact versions we used. Nothing here is
# required for the core experiments (E0, E1, E2, E4, E8, E9, E12, E15), which only need requirements.txt and data/.
#   bash scripts/setup_third_party.sh deepgate2     released DeepGate2 code + weights, and its training set   (E3, E10)
#   bash scripts/setup_third_party.sh abc           ABC at a fixed commit, built from source                  (E17)
#   bash scripts/setup_third_party.sh all
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p third_party logs
what=${1:-all}

deepgate2() {
  if [ ! -d third_party/python-deepgate ]; then
    git clone https://github.com/Ironprop-Stone/python-deepgate.git third_party/python-deepgate || return 1
  fi
  git -C third_party/python-deepgate checkout 173db7529cefc97f7b9b2b3fa97ec1bd5773754d || return 1
  if [ ! -d third_party/dg2_train ]; then                      # DeepGate2's own training set, used only by E10
    mkdir -p third_party/dg2_train && (cd third_party/dg2_train &&
      wget -q https://github.com/Ironprop-Stone/python-deepgate/releases/download/dataset/train.zip && unzip -q train.zip) || return 1
  fi
  echo "DeepGate2 code and data ready"
}
abc() {
  if [ ! -x third_party/abc/abc ]; then
    rm -rf third_party/abc
    git clone https://github.com/berkeley-abc/abc.git third_party/abc || return 1
    git -C third_party/abc checkout f34d3b2899c0ca631b3629f174a720fd776d667b || return 1
    (cd third_party/abc && make -j"$(nproc)" ABC_USE_NO_READLINE=1 ABC_USE_PTHREADS=0 abc > ../../logs/abc_build.log 2>&1) \
      || { echo "ABC build failed, see logs/abc_build.log (usually gcc/g++/make are missing)"; return 1; }
  fi
  third_party/abc/abc -q "version"
}
epfl() {                                                       # EPFL benchmark AIGs, read by tools/gen_large.py (E13)
  [ -d third_party/epfl_benchmarks ] || git clone https://github.com/lsils/benchmarks.git third_party/epfl_benchmarks || return 1
  echo "EPFL benchmarks ready (EPFL_DIR=third_party/epfl_benchmarks)"
}
case $what in
  deepgate2) deepgate2 ;;
  abc) abc ;;
  epfl) epfl ;;
  all) deepgate2 && abc && epfl ;;
  *) echo "usage: bash scripts/setup_third_party.sh deepgate2|abc|epfl|all"; exit 1 ;;
esac
