ports=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null)

function get_driver() {
	port=$1
	udevadm info -a -p  $(udevadm info -q path -n $port) | grep DRIVERS | head -n 1 | awk 'BEGIN{FS="=="} /cp210x/ {gsub("\"","",$2);print $2}' 
}

for i in $ports; do
	driver=$(get_driver $i)
	if [ "$driver" = "cp210x" ]; then 
		echo $i
	fi;
done
