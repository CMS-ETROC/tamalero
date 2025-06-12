index=$1
#echo "start" > /home/daq/ETROC2_Test_Stand/module_test_sw/ETROC_Status.txt

cd /home/daq/ETROC2_Test_Stand/module_test_sw/
source setup.sh

echo "ETL Starting Run $1"
#wait 30

/usr/bin/python3 daq.py --l1a_rate 0 --ext_l1a --kcu 192.168.0.10 --rb 0 --run $index --lock "/home/daq/ETROC2_Test_Stand/module_test_sw/ETROC_Status.txt"

/usr/bin/python3 poke_board.py --configuration modulev1 --etrocs 2 --rbs 0 --modules 0 --kcu 192.168.0.10 --temperature --time --run $index >> temp_log.txt

echo "ETL Done With Run $1"
