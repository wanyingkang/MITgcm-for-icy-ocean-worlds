#! /bin/bash
# this should be customerized to your cluster XX
if [[ ${SLURM_JOB_PARTITION} == *"hdr"* ]]; then
  module purge
  module load intel/2021.4.0_rhel8
  module load openmpi/3.1.6
elif [[ ${SLURM_JOB_PARTITION} == *"fdr"* || ${SLURM_JOB_PARTITION} == *"edr"* ]]; then
  module add intel/2017.0.1
  module add openmpi
fi

export GENERIC='on'
export MITGCMRT="MITROOTXXMITROOT/MITgcm"
export OPTFILE="${MITGCMRT}/tools/build_options/linux_amd64_ifort11"
export MyExp=`pwd`

# 1. obtain input arguments
export CASENAME=$1   
export RESOL=$2 # can be znncs(32,96,501) or znntwod(96,960) or znnx2y96...
export NCORES=$3
export TODO=$4
export NCPUS=$NCORES
export NCPU_PERNODE=32
export NNODES=$(((NCORES-1) / NCPU_PERNODE+1))
export NZ=`echo $RESOL | grep -o -E '[0-9]+' | head -1`
export NX=`echo $RESOL | grep -o -E '[0-9]+' | head -2 | tail -1`
export RESOLUTION=`echo $RESOL | grep -o -E '[A-Za-z]+' | tail -1``echo $RESOL | grep -o -E '[0-9]+' | tail -1`
export CURRENTITER=0
export EACHITER=$((2592000)) # 103680 iter = 360 day, if timestep: 300s
export TOTALITER=$((EACHITER*2))
 # mth openmp related
  export KMP_STACKSIZE=400m
  nTx=`cat input_now/eedata|grep "nTx"|tr -dc '0-9'`
  nTy=`cat input_now/eedata|grep "nTy"|tr -dc '0-9'`
  export OMP_NUM_THREADS=$(( nTx * nTy ))
sizefile=${MyExp}/SIZE_hs/SIZE.h.${RESOLUTION}.${NCORES}p
export OUTPUT=/net/fs06/d0/wanying/data_${CASENAME}

echo CASENAME: $CASENAME
echo NZ: $NZ
echo RESOLUTION: $RESOLUTION
echo NCORES: $NCORES
echo NNODES: $NNODES
echo sizefile: $sizefile

# 2. set up case directory
if [[ $TODO == *"re"* ]]; then
    mv $CASENAME/README .
fi
if [[ $TODO == *"clean"* || $TODO == *"re"* ]]; then
    rm -rf $CASENAME
    rm -rf $OUTPUT
fi
if [[ $TODO == *"compile"* ]]; then
    mkdir $CASENAME
    cd $CASENAME
    #cp $sizefile ../code_now/SIZE.h
    cp ../code_now/SIZE.h $sizefile
    sed -i "s/_NR_/$NZ/g" ../code_now/SIZE.h
    sed -i "s/_NX_/$NX/g" ../code_now/SIZE.h
    cp -r ../code_now ./code
    cp -r ../input_now ./input
    cp ${MyExp}/grids/grid${RESOLUTION}/* ./input/
    mv ../README .
    mkdir build
    mkdir $OUTPUT
    ln -s $OUTPUT .
    cp ../newrundir . 
    chmod +x newrundir
    cp deleterun $OUTPUT
    chmod +s $OUTPUT/deleterun
fi
    
# 3. compile case
if [[ $TODO == *"compile"* ]]; then
    cd build
    ln -s ../code/SIZE.h SIZE.h
    if [[ $OMP_NUM_THREADS == 1 ]]; then
      # mpi only
      ${MITGCMRT}/tools/genmake2 -mods ../code -of ${OPTFILE} -mpi
    else
      # mpi + openmp (mth)
      ${MITGCMRT}/tools/genmake2 -mods ../code -of ${OPTFILE} -mpi -omp
    fi
    make depend
    cd ../
fi
if [[ $TODO == *"build"* ]]; then
    cd build
    make
    cd ../
fi

# 4. copy input files to run director
if [[ $TODO == *"compile"* ]]; then
    ln -s ${MyExp}/${CASENAME}/input/* $OUTPUT/
    ln -s ${MyExp}/${CASENAME}/build/SIZE.h $OUTPUT/
    ln -s ${MyExp}/${CASENAME}/build/mitgcmuv $OUTPUT/
    #ln -s ${MyExp}/grids/grid${RESOLUTION}/* $OUTPUT/
fi

# 5. configure run.sub
if [[ $TODO == *"compile"* ]]; then
  cp ${MyExp}/run.sub_template $OUTPUT/run.sub
  sed -i "s/_CASENAME_/${CASENAME}/g" $OUTPUT/run.sub
  sed -i "s/_NNODES_/${NNODES}/g" $OUTPUT/run.sub
  sed -i "s/_NCPUS_/${NCPUS}/g" $OUTPUT/run.sub
  sed -i "s/_CURRENTITER_/${CURRENTITER}/g" $OUTPUT/run.sub
  sed -i "s/_EACHITER_/${EACHITER}/g" $OUTPUT/run.sub
  sed -i "s/_TOTALITER_/${TOTALITER}/g" $OUTPUT/run.sub
  if [[ ${SLURM_JOB_PARTITION} == *"hdr"* ]]; then
    sed -i "s/_QUEUE_/hdr/g" $OUTPUT/run.sub
  elif [[ ${SLURM_JOB_PARTITION} == *"fdr"* || ${SLURM_JOB_PARTITION} == *"edr"* ]]; then
    sed -i "s/_QUEUE_/edr,fdr/g" $OUTPUT/run.sub
  fi
  BRIEFNAME=`echo ${CASENAME}|rev|cut -d '_' -f 1|rev`
  sed -i "s/_BRIEFNAME_/${BRIEFNAME}/g" $OUTPUT/run.sub

  cp ${MyExp}/climatology.m $OUTPUT/
  sed -i "s/_CASENAME_/${CASENAME}/g" $OUTPUT/climatology.m
  sed -i "s/_EACHITER_/${EACHITER}/g" $OUTPUT/climatology.m
  sed -i "s/_CURRENTITER_/-1/g" $OUTPUT/climatology.m
  export DT=`grep -oP 'deltaT=\s*\K\d+' ./input/data`
  sed -i "s/_DT_/${DT}/g" $OUTPUT/climatology.m
  cp ${MyExp}/cleandata $OUTPUT/
  chmod +x $OUTPUT/cleandata
  cp ${MyExp}/deleterun $OUTPUT/
  chmod +x $OUTPUT/deleterun

  cp ${MyExp}/jupyter_template.ipynb ${MyExp}/${CASENAME}/
fi

