import yaml
import subprocess
from datetime import datetime, timezone

def get_git_version(repo_path=None):
    """
    Get current git commit hash with dirty flag.

    Args:
        repo_path: Path to git repository. If None, uses current directory.
    """
    try:
        git_cmd = ['git']
        if repo_path:
            git_cmd.extend(['-C', str(repo_path)])

        git_cmd.extend(['rev-parse', 'HEAD'])
        commit = subprocess.check_output(
            git_cmd,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()[:8]  # Short hash

        # Only check dirty flag if no repo_path specified (i.e., current DAQ repo)
        if repo_path is None:
            dirty_cmd = ['git', 'status', '--porcelain']
            dirty = subprocess.check_output(
                dirty_cmd,
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()

            return f"{commit}{'_dirty' if dirty else ''}"
        else:
            return commit

    except:
        return "unknown"



def save_run_metadata(system, config, max_run_time, firmware_path=None, note="", charge_injection_mode=False):
    """
    Save complete run configuration metadata to YAML file.

    Args:
        system: ETROCSystem instance with hardware connections
        config: DAQConfig instance with run settings
        max_run_time: Maximum run time in minutes
        firmware_path: Path to firmware git repository (optional)
        note: User-provided run note
        charge_injection_mode: Whether this is a charge injection run
    """
    print("\n5. Saving run metadata...")

    # Prepare metadata structure
    metadata = {
        'run_info': {
            'run_name': config.outdir.name,
            'run_type': 'charge_injection' if charge_injection_mode else 'beam',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'note': note,
            'daq_version': get_git_version(),
            'firmware_version': get_git_version(firmware_path) if firmware_path else "not_specified",
        },

        'hardware': {
            'kcu_ip': config.kcu_ip,
            'readout_board': {
                'id': config.readout_board_id,
                'version': system.rb.ver,
                'config': config.readout_board_config,
            },
        },

        'trigger_config': {
            'enable_mask': config.trigger_enable_mask,
            'trigger_bit_size': config.trigger_data_size,
            'delay_sel': config.trigger_delay_sel,
            'combination_logic': config.trigger_logic,
            'logic_description': 'OR' if config.trigger_logic == 0 else 'AND',
        },

        'acquisition_settings': {
            'max_run_time_minutes': max_run_time,
            'chunk_size': config.chunk_size,
            'max_file_size_bytes': config.max_file_size_bytes,
        },

        'etroc_chips': {},
    }

    # Add charge injection specific settings if applicable
    if charge_injection_mode:
        metadata['charge_injection'] = {
            'charge_fc': config.charge_fc,
            'test_pixels': config.test_pixels,
            'qinj_count': config.qinj_count,
        }

    # Populate each ETROC chip configuration
    for i, etroc in enumerate(system.etroc_chips):
        chip_name = config.etroc_names[i]

        if etroc is None:
            metadata['etroc_chips'][chip_name] = {
                'status': 'not_connected',
            }
            continue

        # Get the board config for this chip
        board_config = config.boards[i]

        chip_metadata = {
            'status': 'connected' if system.connected_names[i] else 'not_responding',
            'i2c_address': f"0x{board_config.i2c_id:02X}",
            'elink_id': board_config.elink_id,
            'threshold_offset': board_config.th_offset,
            'L1A_Delay': board_config.l1a_delay,
        }

        # Get applied DAC values for each pixel
        # We need to reconstruct what was applied in apply_configuration()
        pixels_dac = {}

        # Determine which pixels were configured
        if charge_injection_mode:
            pixels_to_read = config.test_pixels
        else:
            pixels_to_read = [
                (r, c)
                for r in range(config.pixel_row)
                for c in range(config.pixel_col)
            ]

        # Read the DAC register values from the chip
        for row, col in pixels_to_read:
            try:
                # Read actual DAC value from hardware
                dac_value = etroc.rd_reg('DAC', row=row, col=col)
                pixel_key = f"({row},{col})"
                pixels_dac[pixel_key] = int(dac_value)
            except Exception as e:
                print(f"   Warning: Could not read DAC for {chip_name} pixel ({row},{col}): {e}")
                pixels_dac[f"({row},{col})"] = None

        chip_metadata['pixels'] = {
            'dac_values': pixels_dac,
        }

        metadata['etroc_chips'][chip_name] = chip_metadata

    # Save to YAML file
    config.outdir.mkdir(parents=True, exist_ok=True)
    output_path = config.outdir / 'run_metadata.yaml'
    try:
        with open(output_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        print(f"   Metadata saved to: {output_path}")
    except Exception as e:
        print(f"   ERROR: Failed to save metadata: {e}")
        raise

    return metadata
