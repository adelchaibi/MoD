#!/bin/bash

# Copyright (c) 2019-2020, Intel Corporation, All Rights Reserved.
#
# This software and the related documents are Intel copyrighted materials.
# Your use of them is governed by the express license under which they were
# provided to you (License). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this
# software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the
# License.
#
#make sure we are sourced. 
([[ -n $ZSH_EVAL_CONTEXT && $ZSH_EVAL_CONTEXT =~ :file$ ]] || 
 [[ -n $KSH_VERSION && $(cd "$(dirname -- "$0")" &&
    printf '%s' "${PWD%/}/")$(basename -- "$0") != "${.sh.file}" ]] || 
 [[ -n $BASH_VERSION ]] && (return 0 2>/dev/null)) && sourced=1 || sourced=0

if [ $sourced -eq 0 ]; then
  echo "Incorrect usage. $(basename ${BASH_SOURCE[0]}) is not meant to be executed directly, use source instead: "
  echo " source $(basename ${BASH_SOURCE[0]})"
  exit -1
fi

if [ -n OCL_ICD_FILENAMES ]; then
  echo "opt/intel/oneapi/setvars.sh sourced - proceeding"
else
  echo "Incorrect usage. You need to source /opt/intel/oneapi/setvars.sh before running this script"
  exit -1
fi

#
#   And make the necessary path updates
#
export DRIVERLOC=/usr/lib/x86_64-linux-gnu
export OCL_ICD_FILENAMES=$OCL_ICD_FILENAMES:$DRIVERLOC/intel-opencl/libigdrcl.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DRIVERLOC
export PATH=$PATH:/opt/intel/oneapi:$DRIVERLOC

