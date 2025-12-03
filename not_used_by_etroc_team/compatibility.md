DAQ Software and Firmware Compatibility issues 

Can jump to step 3 for quick fix. 

1. Overview: How the System Works

This DAQ software is designed to be flexible and work with multiple versions of the KCU firmware. To achieve this, it uses a dynamic loading mechanism for the hardware address map (the XML files).

The process is as follows:

When the Python script starts, it establishes a minimal connection to the KCU.

It reads a unique version hash (e.g., tamalero/address_table/2481716) directly from a register on the FPGA. This hash identifies the specific firmware version that is currently running.

The script then uses this hash to find and load the corresponding set of XML files from a version-specific sub-folder within tamalero/address_table/.

This ensures that the software's "address map" always matches the hardware's "physical layout."

2. How to Identify the Active Firmware Version
When you run any DAQ script (like tmp_qinj_run.newSMU.py), look at the first few lines of the output. The script will print a debug message that explicitly tells you which XML folder it is loading. This folder name is the active firmware version hash.

Example Output:

Bash

roy@xinghuang:~/yf_temp/tamalero/address_table$ /usr/bin/python /home/roy/yf_temp/tamalero/tmp_qinj_run.newSMU.py -o results_smu
ETROC COSMIC RUN TEST - HARDWARE INITIALIZATION
IPBus address: chtcp-2.0://localhost:10203?target=192.168.0.10:50001
DEBUG: uHAL is loading address table from: /home/roy/yf_temp/tamalero/address_table/2481716/etl_test_fw.xml

In this example, the active firmware version is 2481716.

3. IMPORTANT: Required Code Changes for Trigger Configuration
Firmware development is an iterative process. As a result, the names of the self-trigger configuration registers have changed between different firmware versions. You must ensure the Python script uses the correct register names for the active firmware.

The relevant function in the script is configure_trigger_system().

For Newer Firmware
This firmware version uses functionally descriptive register names. The configure_trigger_system() function should look like this:

Python

# In the DAQ script (e.g., tmp_qinj_run.newsMu.py)

def configure_trigger_system(rb):
    # ...
    # Uses names like TRIG_ENABLE_MASK, TRIG_DATA_SIZE, etc.

    rb.kcu.hw.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK", TRIGGER_ENABLE_MASK)
    rb.kcu.hw.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_DATA_SIZE", TRIGGER_DATA_SIZE)
    rb.kcu.hw.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_DLY_SEL", TRIGGER_DELAY_SEL)

    # ...


For Older Firmware 
This firmware version used numbered registers that were parts of a larger trigger mask. For this version, the configure_trigger_system() function should look like this:

Python

# In DAQ script (e.g., tmp_qinj_run.newsMu.py)

def configure_trigger_system(rb):
    # ...
    # Uses names like TRIG_ENABLE_MASK_0, TRIG_ENABLE_MASK_1, etc.
    rb.kcu.hw.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_0", TRIGGER_ENABLE_MASK)
    rb.kcu.hw.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_1", TRIGGER_DATA_SIZE)
    rb.kcu.hw.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_3", TRIGGER_DELAY_SEL)
    
    # ...

4. Summary Checklist
When switching between different hardware setups or after a new firmware is flashed, follow these steps:

Run DAQ script once.

Look at the initial DEBUG: uHAL is loading address table from: ... message to identify the active firmware hash (e.g., 2481716).

Open DAQ script (e.g., tmp_qinj_run.newsMu.py).

Navigate to the configure_trigger_system() function.

Ensure the rb.kcu.hw.write_node(...) calls use the correct set of register names corresponding to the active firmware version as shown above. If they don't match, edit the script and save it.