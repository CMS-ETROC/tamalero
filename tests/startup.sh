#! /bin/bash

#### Predefined variables and functions ####
RED='\033[1;31m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
NC='\033[0m'

ID="210308B0B4F5"
FIRMWARE="v`python -c 'import tamalero; print(tamalero.__fw_version__)'`"
PSU="192.168.2.3"
CH="ch2"
KCU="192.168.0.11"
HELP="false"
POWER_CYCLE="false"

function info() { echo -e "${BLUE}${@}${NC}"; }
function error() { echo -e "${RED}${@}${NC}"; }
function success() { echo -e "${GREEN}${@}${NC}"; }

get_firmware_zip() {
    version=$1
    project="etl_test_fw"
    projectid="107856"

    file=$project-$version.zip
    url=$(curl  "https://gitlab.cern.ch/api/v4/projects/${projectid}/releases/$version" | jq '.description' | sed -n "s|.*\[$project.zip\](\(.*\)).*|\1|p")
    wget $url
    unzip $file
}

USAGE()
{
	info "Usage: 
		startup
       	      Options:	
		[-i | --id ID]              Unique ID of CI KCU
		[-f | --firmware FIRMWARE]  Firmware version of KCU
		[-p | --psu PSU:CH]         IP address and channel(s) of Power Supply Unit (will trigger power cycle)
		[-k | --kcu KCU]            IP address of Xilinx KCU
		[-c | --cycle]		    Power cycle PSU (not necessary if -p is set)
		[-h | --help]               Show this screen"
}

FIRMWARE=v`/usr/bin/env python3 -c "import tamalero; print(tamalero.__fw_version__)"`

#### Parse and check options and flags ####
PARSED_ARGS=$(getopt -a -n "startup" --options i:f:p:k:ch --longoptions "id:,firmware:,psu:,kcu:,cycle,help" -- "$@")
VALID_ARGS=$?
if [ "${VALID_ARGS}" -ne 0 ]; then
	USAGE
fi

info "PARSED_ARGUMENTS is ${PARSED_ARGS}"
eval set -- "${PARSED_ARGS}"

while true; do
	case "$1" in
		-i | --id ) ID="$2"; shift 2;;
		-f | --firmware ) FIRMWARE="$2"; shift 2;;
		-p | --psu )
			old_ifs="$IFS"
			IFS=":"
			read -r PSU CH <<< "$2"
			POWER_CYCLE="true"
			IFS="$old_ifs"
			shift 2;;
		-k | --kcu ) KCU="$2"; shift 2;;
		-c | --cycle ) POWER_CYCLE="true"; shift;;
		-h | --help ) HELP="true"; shift;;
		--) shift; break;;
	        * ) error "Unexpected option: $1 - this should not happen"; USAGE; return 1;;
	esac
done

if [ "${HELP}" == "true" ]; then
	USAGE
	return 0
fi

#### Startup ####
if [ -z "${TAMALERO_BASE}" ]; then
	error "Must source setup tamalero first."
	return 1
fi

VIVADO=$(command -v vivado || command -v vivado_lab)
if [[ -z "${VIVADO}" ]]; then
	error "ERROR: Vivado not found in path. Must source Vivado first."
        return 1
fi

cd $TAMALERO_BASE

### delete previous FW downloads
rm -rf etl_test_fw-v*

get_firmware_zip "${FIRMWARE}"
cd ./etl_test_fw-${FIRMWARE}
source ./program.sh "${ID}" noflash
cd -

# power cycle the PSUs with cocina
if [ "${POWER_CYCLE}" == "true" ]; then
	info "Power cycle of PSUs with cocina..."
	/usr/bin/env python3 power_cycle.py --ip "${PSU}" --ch "${CH}"
fi

# run test_tamalero with power up
info "Running test_tamalero..."
/usr/bin/env python3 test_tamalero.py --kcu "${KCU}" --power_up --verbose --adcs --strict --configuration emulator
EXIT=$?
if [ ${EXIT} -ne 0 ]; then
	error "Failure when running test_tamalero.py; exit code is ${EXIT}. Blocking merge."
	return ${EXIT}
else
	success "Success! test_tamalero.py exit with code ${EXIT}"
fi
