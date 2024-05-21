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
export CHPL_PATH=$HOME/libs/chapel # CHANGE: 2.1
source $CHPL_PATH/util/setchplenv.bash
export CHPL_LLVM=bundled
export CHPL_TASKS=qthreads
export CHPL_LOCALE_MODEL=gpu
# Kokkos
export KOKKOS_SRC=$HOME/libs/kokkos-4.3.01 # CHANGE: any version is OK
# oneAPI
export ONEAPI_PATH=/opt/intel/oneapi # CHANGE: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
source $ONEAPI_PATH/setvars.sh
# SYCL
export SYCL_PATH=$HOME/libs/sycl # CHANGE: https://github.com/intel/llvm/releases
export PATH=$SYCL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$SYCL_PATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$SYCL_PATH/lib:LIBRARY_PATH
# RAJA
export RAJA_PATH=$HOME/libs/raja

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

# RAJA
# Issues: errors during make
mkdir -p $BUILD/raja
cd $BUILD/raja
cmake -DMODEL=RAJA -DRAJA_SRC=$RAJA_PATH -DENABLE_CUDA=On -DCMAKE_CUDA_ARCHITECTURES=60 -DCUDA_ARCH=sm_60 -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH $SRC
make
cd $ROOT


# SYCL
# Issues: ./bin/onedpl-reduced: error while loading shared libraries: libsycl.so.8: cannot open shared object file: No such file or directory
mkdir -p $BUILD/sycl
cd $BUILD/sycl
cmake -DMODEL=SYCL -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-fsycl -fsycl-unnamed-lambda' $SRC
make
mv ./Reduced $BIN/sycl-reduced
cd $ROOT

# oneDPL
# Issues: ./bin/onedpl-reduced: error while loading shared libraries: libsycl.so.8: cannot open shared object file: No such file or directory
mkdir -p $BUILD/onedpl
cd $BUILD/onedpl
cmake -DMODEL=oneDPL -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS='-fsycl -fsycl-unnamed-lambda' $SRC
make
mv ./Reduced $BIN/onedpl-reduced
cd $ROOT