import random

from tamalero.KCU import KCU
from tamalero.ReadoutBoard import ReadoutBoard

from tamalero.SCA import SCA_CONTROL

if __name__ == '__main__':

    kcu = KCU(name="my_device",
              ipb_path=None,
              adr_table="module_test_fw/address_tables/etl_test_fw.xml",
              dummy=True)

    rb_0 = ReadoutBoard(0, trigger=True, kcu=kcu)

    rb_0.DAQ_LPGBT.dump(20)

    from tamalero.utils import get_temp
    
    # Read ADC channel 7 on DAQ lpGBT
    adc_7 = random.random()

    # Read ADC channel 29 on GBT-SCA
    adc_in29 = random.random()

    # Check what the lpGBT DAC is set to
    v_ref = 1.0
    print ("\nV_ref is set to: %.3f V"%v_ref)

    if v_ref>0:
        print ("\nTemperature on RB RT1 is: %.3f C"%get_temp(adc_7, v_ref, 10000, 25, 10000, 3900))
        print ("Temperature on RB RT2 is: %.3f C"%get_temp(adc_in29, v_ref, 10000, 25, 10000, 3900))
