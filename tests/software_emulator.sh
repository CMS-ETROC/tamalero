#! /bin/bash

#### Predefined variables and functions ####
RED='\033[1;31m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
NC='\033[0m'

function info() { echo -e "${BLUE}${@}${NC}"; }
function error() { echo -e "${RED}${@}${NC}"; }
function success() { echo -e "${GREEN}${@}${NC}"; }

# run test_tamalero with power up
info "Running test_ETROC..."
/usr/bin/env python3 test_ETROC.py
EXIT=$?
if [ ${EXIT} -ne 0 ]; then
	error "Failure when running test_ETROC.py; exit code is ${EXIT}. Blocking merge."
	return ${EXIT}
else
	success "Success! test_ETROC.py exit with code ${EXIT}"
fi
