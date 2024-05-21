ROOT=$(pwd)
BIN=$ROOT/bin
SRC=$ROOT/src
BUILD=$ROOT/build

rm -rf $BIN
mkdir -p $BIN
rm -rf $BUILD
mkdir -p $BUILD

# Environment
# CUDA
export CUDA_PATH=/usr/local/cuda-12.4 # CHANGE
export PATH=${CUDA_PATH}/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
# Chapel
export CHPL_PATH=$HOME/libs/chapel # CHANGE
source $CHPL_PATH/util/setchplenv.bash
export CHPL_LLVM=bundled
export CHPL_TASKS=qthreads
export CHPL_LOCALE_MODEL=gpu
# Kokkos
export KOKKOS_SRC=$HOME/libs/kokkos-4.3.01 # CHANGE
# oneAPI
export ONEAPI_PATH=/opt/intel/oneapi # CHANGE
source $ONEAPI_PATH/setvars.sh

# Chapel bench build
cd $SRC/chapel
make CHPL_LOCALE_MODEL=gpu CHPL_GPU=nvidia CHPL_GPU_ARCH=sm_60
mv ./chapel-reduced $BIN
cd $ROOT

# Kokkos bench build
# Issues: Kokkos src must be under this project's src folder
mkdir -p $BUILD/kokkos
cd $BUILD/kokkos
cmake -DKOKKOS_SRC=$KOKKOS_SRC -DMODEL=Kokkos -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_ARCH_VOLTA72=On $SRC # CHANGE: Didn't find corresponding Kokkos_ARCH_VOLTAXX for 3070ti
make
mv ./Reduced $BIN/kokkos-reduced
cd $ROOT

# OpenMP (Target)
mkdir -p $BUILD/omp
cd $BUILD/omp
cmake -DMODEL=OpenMP-target -DOMP_TARGET=Intel -DCMAKE_CXX_COMPILER=icpx $SRC
make
mv ./Reduced $BIN/omp-reduced
cd $ROOT


