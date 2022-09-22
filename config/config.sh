source /etc/profile.d/modules.sh
source /opt/intel/oneapi/setvars.sh
module load intel/igt-embargo/587590eb1
module load intel/oneapi/2022.2.0  # <=== This oneAPI version works with the prebuilt wheels
source /opt/intel/oneapi/PVC_setup.sh   # <=== this has to be the last line, otherwise the PVC variables may be overridden by some other scripts!

