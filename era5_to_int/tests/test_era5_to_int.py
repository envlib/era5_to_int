"""
Tests for era5_to_int.

Uses ERA5 test data from /home/mike/data/wrf/tests/test_data/era5/.
"""
import pathlib
import struct
import uuid

import numpy as np
import pytest
from typer.testing import CliRunner

from era5_to_int.era5_to_int import (
    app,
    days_in_month,
    intdate_to_string,
    datetime_to_string,
    string_to_yyyymmddhh,
    begin_6hourly,
    end_6hourly,
    begin_daily,
    end_daily,
    begin_monthly,
    end_monthly,
    find_era5_file,
    find_time_index,
    MetVar,
)

ERA5_ROOT = pathlib.Path('/home/mike/data/wrf/tests/test_data/era5')

runner = CliRunner()


# ======================================================================
# Utility functions
# ======================================================================

class TestDaysInMonth:
    def test_january(self):
        assert days_in_month(2023, 1) == 31

    def test_february_non_leap(self):
        assert days_in_month(2023, 2) == 28

    def test_february_leap(self):
        assert days_in_month(2024, 2) == 29

    def test_february_century_non_leap(self):
        assert days_in_month(1900, 2) == 28

    def test_february_400_year_leap(self):
        assert days_in_month(2000, 2) == 29


class TestIntdateToString:
    def test_basic(self):
        assert intdate_to_string(2023021200) == '2023-02-12_00:00:00'

    def test_nonzero_hour(self):
        assert intdate_to_string(2023021218) == '2023-02-12_18:00:00'


class TestDatetimeToString:
    def test_basic(self):
        from datetime import datetime
        dt = datetime(2023, 2, 12, 6)
        assert datetime_to_string(dt) == '2023-02-12_06'


class TestStringToYyyymmddHh:
    def test_basic(self):
        yyyy, mm, dd, hh = string_to_yyyymmddhh('2023-02-12_06')
        assert (yyyy, mm, dd, hh) == (2023, 2, 12, 6)


class TestDateRounding:
    def test_begin_6hourly(self):
        assert begin_6hourly(2023, 2, 12, 7) == '2023021206'

    def test_end_6hourly(self):
        assert end_6hourly(2023, 2, 12, 7) == '2023021211'

    def test_begin_daily(self):
        assert begin_daily(2023, 2, 12, 15) == '2023021200'

    def test_end_daily(self):
        assert end_daily(2023, 2, 12, 15) == '2023021223'

    def test_begin_monthly(self):
        assert begin_monthly(2023, 2, 12, 15) == '2023020100'

    def test_end_monthly(self):
        assert end_monthly(2023, 2, 12, 15) == '2023022823'


# ======================================================================
# File lookup
# ======================================================================

class TestFindEra5File:
    @pytest.fixture
    def temp_var(self):
        return MetVar('TT', 'T', 'e5.oper.an.pl.128_130_t.ll025sc.{}_{}.nc',
                       begin_daily, end_daily)

    def test_finds_pressure_level_file(self, temp_var):
        path = find_era5_file(temp_var, '2023-02-12_00', ERA5_ROOT)
        assert path.exists()
        assert 'e5.oper.an.pl.128_130_t' in path.name

    def test_missing_file_raises(self, temp_var):
        with pytest.raises(RuntimeError, match='Could not find file'):
            find_era5_file(temp_var, '2099-01-01_00', ERA5_ROOT)

    def test_finds_invariant_file(self):
        var = MetVar('LANDSEA', 'LSM',
                     'e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc',
                     begin_monthly, end_monthly, isInvariant=True)
        path = find_era5_file(var, '2023-02-12_00', ERA5_ROOT)
        assert path.exists()


class TestFindTimeIndex:
    def test_finds_existing_time(self):
        path = ERA5_ROOT / 'e5.oper.an.pl' / '202302' / 'e5.oper.an.pl.128_130_t.ll025sc.2023021200_2023021223.nc'
        idx = find_time_index(path, '2023-02-12_06')
        assert idx == 6  # hour 6 in a daily file starting at hour 0

    def test_missing_time_returns_minus_one(self):
        path = ERA5_ROOT / 'e5.oper.an.pl' / '202302' / 'e5.oper.an.pl.128_130_t.ll025sc.2023021200_2023021223.nc'
        idx = find_time_index(path, '2023-01-01_00')
        assert idx == -1


# ======================================================================
# CLI — Help
# ======================================================================

class TestCliHelp:
    def test_help_exits_zero(self):
        result = runner.invoke(app, ['--help'])
        assert result.exit_code == 0

    def test_help_mentions_era5(self):
        output = result = runner.invoke(app, ['--help'])
        assert 'era5' in result.output.lower() or 'ERA5' in result.output


# ======================================================================
# CLI — End-to-end conversion
# ======================================================================

class TestCliConvert:
    def test_single_timestep(self, tmp_path):
        """Convert a single 6-hourly timestep and verify output file is created."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, [
            str(ERA5_ROOT),
            '2023-02-12_12',
            '2023-02-12_12',
            '-h', '6',
        ])
        assert result.exit_code == 0, result.output

        # Should produce ERA5:2023-02-12_12
        out_file = tmp_path / 'ERA5:2023-02-12_12'
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_output_has_correct_fields(self, tmp_path):
        """Verify the intermediate file contains expected WPS fields."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, [
            str(ERA5_ROOT),
            '2023-02-12_12',
            '2023-02-12_12',
            '-h', '6',
        ])
        assert result.exit_code == 0, result.output

        out_file = tmp_path / 'ERA5:2023-02-12_12'
        fields = _read_int_fields(out_file)

        field_names = {f[0] for f in fields}

        # Pressure-level fields
        assert 'TT' in field_names
        assert 'UU' in field_names
        assert 'VV' in field_names
        assert 'HGT' in field_names  # GHT → HGT in WPSUtils
        assert 'RH' in field_names
        assert 'SPECHUMD' in field_names

        # Surface fields
        assert 'PSFC' in field_names
        assert 'PMSL' in field_names
        assert 'SKINTEMP' in field_names
        assert 'LANDSEA' in field_names
        assert 'SOILHGT' in field_names

    def test_output_projection_latlon(self, tmp_path):
        """Verify the intermediate file uses LATLON projection (iproj=0)."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, [
            str(ERA5_ROOT),
            '2023-02-12_12',
            '2023-02-12_12',
            '-h', '6',
        ])
        assert result.exit_code == 0, result.output

        out_file = tmp_path / 'ERA5:2023-02-12_12'
        fields = _read_int_fields(out_file)

        # All fields should be LATLON (iproj=0)
        for name, xlvl, nx, ny, iproj in fields:
            assert iproj == 0, f'{name} at {xlvl} has iproj={iproj}, expected 0'

    def test_variable_filter(self, tmp_path):
        """--variables flag should limit output to selected variables."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, [
            str(ERA5_ROOT),
            '2023-02-12_12',
            '2023-02-12_12',
            '-h', '6',
            '-v', 'PSFC,PMSL',
        ])
        assert result.exit_code == 0, result.output

        out_file = tmp_path / 'ERA5:2023-02-12_12'
        fields = _read_int_fields(out_file)
        field_names = {f[0] for f in fields}

        assert 'PSFC' in field_names
        assert 'PMSL' in field_names
        # Pressure-level fields should not be present
        assert 'TT' not in field_names or all(f[1] == 200100.0 for f in fields if f[0] == 'TT')

    def test_multiple_timesteps(self, tmp_path):
        """Converting a range produces multiple output files."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, [
            str(ERA5_ROOT),
            '2023-02-12_12',
            '2023-02-12_18',
            '-h', '6',
            '-v', 'PSFC',
        ])
        assert result.exit_code == 0, result.output

        assert (tmp_path / 'ERA5:2023-02-12_12').exists()
        assert (tmp_path / 'ERA5:2023-02-12_18').exists()

    def test_skip_vars_excludes_field(self, tmp_path):
        """--skip-vars removes the named field from the output."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, [
            str(ERA5_ROOT),
            '2023-02-12_12',
            '2023-02-12_12',
            '-h', '6',
            '-s', 'SST,SEAICE',
        ])
        assert result.exit_code == 0, result.output

        out_file = tmp_path / 'ERA5:2023-02-12_12'
        fields = _read_int_fields(out_file)
        field_names = {f[0] for f in fields}

        assert 'SST' not in field_names
        assert 'SEAICE' not in field_names
        # Other fields still written
        assert 'PSFC' in field_names
        assert 'SKINTEMP' in field_names

    def test_skip_vars_unknown_errors(self, tmp_path):
        """--skip-vars with an unknown WPS variable exits 1."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, [
            str(ERA5_ROOT),
            '2023-02-12_12',
            '2023-02-12_12',
            '-h', '6',
            '-s', 'NOTAREALVAR',
        ])
        assert result.exit_code == 1
        assert 'NOTAREALVAR' in result.output

    def test_skip_vars_composes_with_variables(self, tmp_path):
        """--variables and --skip-vars compose: include list minus skip list."""
        import os
        os.chdir(tmp_path)

        result = runner.invoke(app, [
            str(ERA5_ROOT),
            '2023-02-12_12',
            '2023-02-12_12',
            '-h', '6',
            '-v', 'PSFC,PMSL,SST',
            '-s', 'SST',
        ])
        assert result.exit_code == 0, result.output

        out_file = tmp_path / 'ERA5:2023-02-12_12'
        fields = _read_int_fields(out_file)
        field_names = {f[0] for f in fields}

        assert 'PSFC' in field_names
        assert 'PMSL' in field_names
        assert 'SST' not in field_names


# ======================================================================
# Helpers
# ======================================================================

def _read_fortran_record(f):
    raw = f.read(4)
    if not raw:
        return None
    nbytes = struct.unpack('>I', raw)[0]
    data = f.read(nbytes)
    f.read(4)
    return data


def _read_int_fields(path):
    """Read all field headers from a WPS intermediate file.

    Returns list of (field_name, xlvl, nx, ny, iproj).
    """
    fields = []
    with open(path, 'rb') as f:
        while True:
            rec = _read_fortran_record(f)
            if rec is None:
                break
            rec = _read_fortran_record(f)
            field = rec[60:69].decode().strip()
            xlvl = struct.unpack('>f', rec[140:144])[0]
            nx, ny, iproj = struct.unpack('>iii', rec[144:156])
            _read_fortran_record(f)  # projection
            _read_fortran_record(f)  # wind flag
            _read_fortran_record(f)  # data slab
            fields.append((field, xlvl, nx, ny, iproj))
    return fields
