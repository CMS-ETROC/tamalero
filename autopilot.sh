echo $3
next_file_index="../ScopeHandler/Lecroy/Acquisition/next_run_number.txt"
index=`cat $next_file_index`
echo $index

#i/usr/bin/python3 telescope.py -- --kcu 192.168.0.10 --offset $2 --delay 32
echo -n "True" > running_ETROC_acquisition.txt
(/usr/bin/python3 daq.py --l1a_rate 0 --ext_l1a --kcu 192.168.0.10 --rb 0 --run $index --lock "/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/Lecroy/Acquisition/running_acquisition.txt") &
(sleep 15
/usr/bin/python3 /home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/Lecroy/Acquisition/acquisition_wrapper.py --nevents $1)
/usr/bin/python3 data_dumper.py --input "ETROC_output/output_run_${index}_rb0.dat" --rbs 0 --skip_trigger_check
# remember: 0,1,2
#/usr/bin/python3 root_dumper.py --input ${index} #_rb0 # Run in other shell with root configured
echo -n "False" > running_ETROC_acquisition.txt
echo -n "True" > merging.txt
echo -n "True" >../ScopeHandler/Lecroy/Acquisition/merging.txt
