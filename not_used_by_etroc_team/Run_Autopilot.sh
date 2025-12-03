#!/bin/bash

# 1) Number of runs
# 2) Bias voltage
# 3) Threshold offset
# 4) Number of events
# 5) The number of the PCb board, use "-" to separate boards id in multilayer setup
# 6) Wirebonded/Bump-bonded wb or bb
# 7) Beam energy
# 8) tracker
# 9) ETROC Power mode: i1,i2,i3,i4
# 9) Multilayer setup

# LOOP={10..12..1}
bias_V=$2 # V
offset=$3 # vth
n_events=$4
board_number=$5
bond=$6
energy=$7
isTrack=$8
powerMode=$9  # I1 (high) to I4 (low)
isMulti=${10}
run_number=`cat ../ScopeHandler/Lecroy/Acquisition/next_run_number.txt`

echo "Starting configuration for $run_number on KCU . Turn beam OFF!!"
/usr/bin/python3 telescope.py  --configuration cern_1 --kcu 192.168.0.10 --offset $3 --delay 14 
#/usr/bin/python3 poke_board.py --configuration modulev1 --etrocs 0 --kcu 192.168.0.10 --dark_mode --mask telescope_config_data/cern_test_2024-05-13-20-07-26/noise_width_module_106_etroc_2.yaml
/usr/bin/python3 poke_board.py --configuration modulev1 --etrocs 2 --kcu 192.168.0.10 --dark_mode

if [ "$isTrack" = true ]
then
    echo "You are starting a telescope run. Have you entered the run number $run_number on telescope? And turn BEAM ON now"
    read dummy
fi  

for i in $(seq 1 $1)
do
    merging_dir="../ScopeHandler/ScopeData/LecroyMerged/"
    
    echo "___________________________________ "$i
    run_number=`cat ../ScopeHandler/Lecroy/Acquisition/next_run_number.txt`
    echo "Run number: $run_number"
    # /usr/bin/python3 poke_board.py --configuration modulev0b --etrocs 0  --kcu 192.168.0.10 --rb 1 --bitslip
    # /usr/bin/python3 poke_board.py --configuration modulev1 --etrocs 2  --kcu 192.168.0.10 --rb 2 --bitslip
    # /usr/bin/python3 poke_board.py --configuration modulev1 --etrocs 2  --kcu 192.168.0.10 --rb 1 --bitslip
    ./autopilot.sh $n_events $offset
    temperature=$(/usr/bin/python3 poke_board.py --configuration modulev1 --etrocs 2 --rbs 0 --modules 1 --kcu 192.168.0.10 --temperature)

    sleep 7s
    kcu=`cat ./running_ETROC_acquisition.txt`
    scope=`cat ../ScopeHandler/Lecroy/Acquisition/running_acquisition.txt`
    conversion=`cat ../ScopeHandler/Lecroy/Conversion/ongoing_conversion.txt`
    merging=`cat ../ScopeHandler/Lecroy/Merging/ongoing_merging.txt`
    echo $kcu
    echo $scope
    echo $conversion
    echo $merging

    while [ $kcu == "True" ] || [ $scope == "True" ] || [ $conversion == "True" ] || [ $merging == "True" ]; do
        echo "Waiting..."
        sleep 1s
        kcu=`cat ./running_ETROC_acquisition.txt`
        scope=`cat ../ScopeHandler/Lecroy/Acquisition/running_acquisition.txt`
        conversion=`cat ../ScopeHandler/Lecroy/Conversion/ongoing_conversion.txt`
        merging=`cat ../ScopeHandler/Lecroy/Merging/ongoing_merging.txt`
        echo $kcu
        echo $scope
        echo $conversion
        echo $merging
    done

    test_successful=`test "$merging_dir/run_$run_number.root"`

    printf "$run_number,$bias_V,$offset,$n_events,$board_number,$bond,$energy,`date -u`,$isTrack,$powerMode,$temperature, $isMulti \n">>./run_log_SPS_Oct2024.csv

done
