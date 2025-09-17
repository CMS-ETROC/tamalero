from tamalero.KCU import KCU
import os
import uhal

KCU_IP = "192.168.0.10"

print("Connecting to KCU to read its firmware version hash...")
try:

    ipb_path = f"chtcp-2.0://localhost:10203?target={KCU_IP}:50001"

    generic_xml_path = os.path.expandvars("$TAMALERO_BASE/address_table/generic/etl_test_fw.xml")
    
    
    kcu_tmp = KCU(
        name="tmp_kcu",
        ipb_path=ipb_path,
        adr_table=generic_xml_path
    )

    firmware_version_hash = kcu_tmp.get_xml_sha()
    final_xml_path = os.path.expandvars(f"$TAMALERO_BASE/address_table/{firmware_version_hash}/etl_test_fw.xml")

    print(f"\nFirmware on KCU is reporting its version hash as:")
    print(f"  -> {firmware_version_hash}")
    print(f"\nPython application will attempt to load the XML files from this directory:")
    print(f"  -> {os.path.dirname(final_xml_path)}")
    print("\n" + "="*60 + "\n")

    if os.path.isdir(os.path.dirname(final_xml_path)):
        print(f"The directory '.../{firmware_version_hash}/' was found on your computer.")
    else:
        print(f"The directory '.../{firmware_version_hash}/' was NOT found")

except uhal._core.exception as e:
    print(f"\nError occurred while communicating with the KCU: {e}")
    print("   Please check your KCU_IP, connection, and whether the ControlHub is running.")
except Exception as e:
    print(f"\nAn unexpected Python error occurred: {e}")