import yaml
import subprocess
from datetime import datetime, timezone

def get_git_version():
    """Get current git commit hash with dirty flag."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()[:8]  # Short hash

        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()

        return f"{commit}{'_dirty' if dirty else ''}"
    except:
        return "unknown"


def save_run_metadata(system, config, cal_mgr, note="", charge_injection_mode=False):
    """
    Save complete run configuration metadata to YAML file.

    Args:
        system: ETROCSystem instance with hardware connections
        config: DAQConfig instance with run settings
        cal_mgr: CalibrationManager instance (to access applied DAC values)
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
            'software_version': get_git_version(),
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
            'data_size': config.trigger_data_size,
            'delay_sel': config.trigger_delay_sel,
            'combination_logic': config.trigger_logic,
            'logic_description': 'OR' if config.trigger_logic == 0 else 'AND',
        },

        'acquisition_settings': {
            'max_run_time_minutes': config.max_run_time,
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
            'pixel_dimensions': {
                'rows': config.pixel_row,
                'cols': config.pixel_col,
            },
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
                dac_value = etroc.rd_reg('DAC', row=row, col=col, broadcast=False)
                pixel_key = f"({row},{col})"
                pixels_dac[pixel_key] = int(dac_value)
            except Exception as e:
                print(f"   Warning: Could not read DAC for {chip_name} pixel ({row},{col}): {e}")
                pixels_dac[f"({row},{col})"] = None

        chip_metadata['pixels'] = {
            'count': len(pixels_dac),
            'dac_values': pixels_dac,
        }

        metadata['etroc_chips'][chip_name] = chip_metadata

    # Save to YAML file
    output_path = config.outdir / 'run_metadata.yaml'
    try:
        with open(output_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        print(f"   Metadata saved to: {output_path}")
    except Exception as e:
        print(f"   ERROR: Failed to save metadata: {e}")
        raise

    return metadata