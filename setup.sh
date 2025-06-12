export TAMALERO_BASE=$PWD
export LD_LIBRARY_PATH=/opt/cactus/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONIOENCODING=utf8
echo "Set PYTHONPATH"
echo $PYTHONPATH
echo "Set LD_LIBRARY"
echo $LD_LIBRARY_PATH

# making some of the directories that are (currently) needed
mkdir -p output
mkdir -p ETROC_output
mkdir -p results
mkdir -p fit_results
