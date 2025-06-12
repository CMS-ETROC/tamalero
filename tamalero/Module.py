#!/usr/bin/env python3
import os
from tamalero.utils import load_yaml
from tamalero.colors import red, green
from tamalero.ETROC import ETROC
from tamalero.Monitoring import Lock
from time import sleep

class Module:
    def __init__(self, rb, i=1, strict=False, enable_power_board=False, moduleid=0, poke=False, hard_reset=False, ext_vref=False, verbose=False):
        # don't like that this also needs a RB
        # think about a better solution
        self.config = rb.configuration['modules'][i]
        self.breed = rb.config
        #self.config = load_yaml(map_file)[f'm{i}']
        #self.regs = load_yaml(os.path.expandvars('$TAMALERO_BASE/address_table/ETROC2.yaml'))
        #self.regs_em = ['disScrambler', 'singlePort', 'mergeTriggerData', 'triggerGranularity']
        self.i = i
        self.rb = rb
        self.id = moduleid

        if enable_power_board:
            self.enable_power_board()
            sleep(0.1)  # enough time to let the ETROC power up?

        self.ETROCs = []
        all_good = True
        for j in range(len(self.config['addresses'])):
            if self.config['i2c']['master']=='fake':
                self.ETROCs.append(
                    ETROC(
                        #usefake=,
                        master=self.config['i2c']['master'],
                        i2c_channel=self.config['i2c']['channel'],
                        elinks={k: self.config['elinks'][j][k] for k in range(len(self.config['elinks'][j]))},
                        i2c_adr = self.config['addresses'][j],
                        reset = None,
                        breed = 'software'
                    ))
            else:
                try:
                    if verbose: print(f"Working on ETROC {j} of module {self.id}")
                    self.ETROCs.append(
                        ETROC(
                            rb          = rb,
                            master      = self.config['i2c']['master'],
                            i2c_channel = self.config['i2c']['channel'],
                            elinks={k: self.config['elinks'][j][k] for k in range(len(self.config['elinks'][j]))},
                            #elinks      = self.config['elinks'][j],
                            i2c_adr     = self.config['addresses'][j],
                            strict      = strict,
                            reset = self.config['reset'],
                            breed = self.breed,
                            vref = self.config['vref'][j],
                            vref_pd = self.config['disable_vref_gen'][j] if ext_vref is False else True,
                            vtemp = self.config['vtemp'][j],
                            chip_id = (self.id << 2) | j,  # this gives every ETROC a unique ID, based on module ID and ETROC number on the module
                            no_init = poke,
                            hard_reset = hard_reset,
                            no_hard_reset_on_init = (j != 0),
                        ))
                    all_good &= self.ETROCs[-1].get_elink_status(summary=True)
                except RuntimeError:
                    print("Couldn't add ETROC", j)

        self.connected = any([etroc.is_connected() for etroc in self.ETROCs])

        if not all_good and self.connected and not poke:
            for etroc in self.ETROCs:
                etroc.default_config()
            for etroc in self.ETROCs:
                etroc.default_config(no_reset=True)


    #def configure(self):
    #    if self.connected:
    #        self.wr_reg('singlePort', 0)
    #        self.wr_reg('mergeTriggerData', 0)
    #        self.wr_reg('disScrambler', 1)

    #def I2C_write(self, reg=0x0, val=0x0):
    #    self.I2C_master.I2C_write(
    #        reg=reg,
    #        val=val,
    #        master=self.config['i2c']['channel'],
    #        slave_addr=0x72,  # NOTE this will need to change in the future
    #    )

    #def I2C_read(self, reg=0x0):
    #    return self.I2C_master.I2C_read(
    #        reg=reg,
    #        master=self.config['i2c']['channel'],
    #        slave_addr=0x72,  # NOTE this will need to change in the future
    #    )

    #def wr_reg(self, reg, val):
    #    adr   = self.regs[reg]['regadr']
    #    shift = self.regs[reg]['shift']
    #    mask  = self.regs[reg]['mask']

    #    orig_val = self.I2C_read(adr)
    #    new_val = ((val<<shift)&mask) | (orig_val&(~mask))

    #    self.I2C_write(adr, new_val)

    #def rd_reg(self, reg):
    #    adr = self.regs[reg]['regadr']
    #    mask = self.regs[reg]['mask']
    #    shift = self.regs[reg]['shift']

    #    return (self.I2C_read(adr)&mask) >> shift

    def get_power_board_status(self):
        return self.rb.SCA.read_gpio(self.config['power_board'])

    def enable_power_board(self):
        return self.rb.SCA.set_gpio(self.config['power_board'], 1)

    def disable_power_board(self):
        return self.rb.SCA.set_gpio(self.config['power_board'], 0)

    def get_power_good(self):
        if self.rb.config.count('modulev0') and self.rb.ver<3:
            return self.rb.SCA.read_gpio(self.config['pgood'])
        else:
            return False

    def get_locked_links(self):
        self.locked = {0:[], 1:[]}
        self.unlocked = {0:[], 1:[]}
        for etroc in self.ETROCs:
            etroc.get_elink_status()
            for i in [0,1]:
                if i in etroc.links_locked:
                    for j, link in enumerate(etroc.elinks[i]):
                        if etroc.links_locked[i][j]:
                            self.locked[i].append(link)
                        else:
                            self.unlocked[i].append(link)
        return {'locked': self.locked, 'unlocked': self.unlocked}

    
    def show_status(self):
        self.get_locked_links()
        len_corrector_0 = len(' '.join([str(x) for x in self.locked[0] + self.unlocked[0]]))
        len_corrector_1 = len(' '.join([str(x) for x in self.locked[1] + self.unlocked[1]]))
        if len_corrector_0 == 0: len_corrector_0 = -1
        if len_corrector_1 == 0: len_corrector_1 = -1

        print ('┏━┳━' + 25*'━' + '━┳━┓')
        print ('┃○┃ ' + 25*' ' + ' ┃○┃')
        print ('┃ ┃ ' + '{:10}{:<5}{:4}{:<6}'.format("Module:", self.i, "ID:", self.id) + ' ┃ ┃' )
        # this was useful for the emulator, but version numbers haven't
        # been updated anyway...
        # no FW version for the actual ETROCs
        #ver = self.ETROCs[0].get_ver()
        #print ('┃ ┃ ' + '{:16}{:9}'.format("Firmware ver.",ver) + ' ┃ ┃' )
        pb_status, pb_col = ('on', green) if self.get_power_good() else ('off', red)
        print ('┃ ┃ ' + pb_col('{:16}{:9}'.format("Power board is:",pb_status)) + ' ┃ ┃' )
        #print ('┃ ┃ ' + 25*' ' + ' ┃ ┃')
        col = green if self.ETROCs[0].connected else red
        prefix = '' if self.ETROCs[0].connected else "Not "
        print ('┃ ┃ ' + col('{:25}'.format(prefix+"Connected:")) + ' ┃ ┃' )
        print ('┃ ┃ ' + col(' {:12}{:12}'.format('i2c master:', self.ETROCs[0].master)) + ' ┃ ┃')
        print ('┃ ┃ ' + col(' {:12}{:<12}'.format('channel:', self.ETROCs[0].i2c_channel )) + ' ┃ ┃')

        print ('┃ ┃ ' + '{:25}'.format("lpGBT 1 links:") + ' ┃ ┃' )
        stats = [ green(str(l)) for l in self.locked[0] ] + [red(str(l)) for l in self.unlocked[0]]
        print ('┃ ┃ ' + (' {}'*len(self.locked[0]+self.unlocked[0])).format(*stats) + (24-len_corrector_0)*' ' + ' ┃ ┃' )

        print ('┃ ┃ ' + '{:25}'.format("lpGBT 2 links:") + ' ┃ ┃' )
        stats = [ green(str(l)) for l in self.locked[1] ] + [red(str(l)) for l in self.unlocked[1]]
        print ('┃ ┃ ' + (' {}'*len(self.locked[1]+self.unlocked[1])).format(*stats) + (24-len_corrector_1)*' ' + ' ┃ ┃' )


        print ('┃○┃ ' + 25*' ' + ' ┃○┃')
        print ('┗━┻━' + 25*'━' + '━┻━┛')

    def monitor(self):
        with Lock(self.rb.SCA) as l:
            status = 0
            if self.ETROCs[0].is_connected(): status += 1
            for etroc in self.ETROCs:
                etroc.get_elink_status()
                if etroc.daq_locked:
                    status += 1

            self.rb.SCA.set_gpio(self.rb.SCA.gpio_mapping[self.config['status']]['pin'], 0)
            sleep(0.25)
            for i in range(status):
                self.rb.SCA.set_gpio(self.rb.SCA.gpio_mapping[self.config['status']]['pin'], 1)
                sleep(0.25)
                self.rb.SCA.set_gpio(self.rb.SCA.gpio_mapping[self.config['status']]['pin'], 0)
                sleep(0.25)

            if status > 0:
                self.rb.SCA.set_gpio(self.rb.SCA.gpio_mapping[self.config['status']]['pin'], 1)
            else:
                self.rb.SCA.set_gpio(self.rb.SCA.gpio_mapping[self.config['status']]['pin'], 0)
